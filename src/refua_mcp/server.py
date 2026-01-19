from __future__ import annotations

import base64
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from refua.boltz.data.write.mmcif import to_mmcif
from mcp.server.fastmcp import FastMCP
from refua import Boltz2, BoltzGen
from refua.boltz.api import bcif_bytes_from_mmcif, msa_from_a3m

mcp = FastMCP("refua-mcp")

DEFAULT_BOLTZ_CACHE = str(Path("~/.boltz").expanduser())
JOB_HISTORY_LIMIT = 100
JOB_MAX_WORKERS = 1


@dataclass
class JobRecord:
    job_id: str
    tool: str
    status: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


_JOB_LOCK = threading.Lock()
_JOB_STORE: OrderedDict[str, JobRecord] = OrderedDict()
_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=JOB_MAX_WORKERS)


@lru_cache(maxsize=4)
def _get_boltz2(
    cache_dir: str | None,
    device: str | None,
    auto_download: bool,
    use_kernels: bool,
    affinity_mw_correction: bool,
) -> Boltz2:
    if not cache_dir:
        cache_dir = DEFAULT_BOLTZ_CACHE
    return Boltz2(
        cache_dir=cache_dir,
        device=device,
        auto_download=auto_download,
        use_kernels=use_kernels,
        affinity_mw_correction=affinity_mw_correction,
    )


@lru_cache(maxsize=4)
def _get_boltzgen(
    mol_dir: str | None,
    auto_download: bool,
    cache_dir: str | None,
    token: str | None,
    force_download: bool,
) -> BoltzGen:
    return BoltzGen(
        mol_dir=mol_dir,
        auto_download=auto_download,
        cache_dir=cache_dir,
        token=token,
        force_download=force_download,
    )


def _coerce_modifications(mods: Iterable[Any]) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for mod in mods:
        if isinstance(mod, dict):
            if "position" not in mod or "ccd" not in mod:
                raise ValueError("Modification requires position and ccd.")
            result.append((int(mod["position"]), str(mod["ccd"])))
        elif isinstance(mod, (list, tuple)) and len(mod) == 2:
            result.append((int(mod[0]), str(mod[1])))
        else:
            raise ValueError("Modification entries must be dicts or (position, ccd) tuples.")
    return result


def _resolve_output_format(output_path: str | None, output_format: str | None) -> str | None:
    if output_format:
        normalized = output_format.lower()
        if normalized not in {"cif", "bcif"}:
            raise ValueError("output_format must be 'cif' or 'bcif'.")
        return normalized
    if output_path:
        suffix = Path(output_path).suffix.lower()
        if suffix == ".bcif":
            return "bcif"
        if suffix in {".cif", ".mmcif"}:
            return "cif"
    return None


def _resolve_feature_output_format(output_path: str, output_format: str | None) -> str:
    if output_format:
        normalized = output_format.lower()
        if normalized not in {"torch", "npz"}:
            raise ValueError("output_format must be 'torch' or 'npz'.")
        return normalized
    suffix = Path(output_path).suffix.lower()
    if suffix in {".pt", ".pth", ".torch"}:
        return "torch"
    if suffix == ".npz":
        return "npz"
    return "torch"


def _build_fold_complex(
    model: Boltz2,
    *,
    name: str,
    chains: list[dict[str, Any]],
    constraints: list[dict[str, Any]] | None,
) -> Any:
    if not chains:
        raise ValueError("chains must include at least one chain spec.")

    complex_builder = model.fold_complex(name)

    for chain in chains:
        chain_type = str(chain.get("type", "")).lower()
        ids = chain.get("id")
        if not ids:
            raise ValueError("Each chain must include an 'id'.")

        if chain_type == "protein":
            sequence = chain.get("sequence")
            if not sequence:
                raise ValueError("Protein chains require a sequence.")
            mods = _coerce_modifications(chain.get("modifications", []))
            msa = None
            msa_a3m = chain.get("msa_a3m")
            if msa_a3m:
                msa = msa_from_a3m(
                    str(msa_a3m),
                    taxonomy=chain.get("msa_taxonomy"),
                    max_seqs=chain.get("msa_max_seqs"),
                )
            complex_builder.protein(
                ids,
                sequence,
                modifications=mods,
                msa=msa,
                cyclic=bool(chain.get("cyclic", False)),
            )
        elif chain_type == "dna":
            sequence = chain.get("sequence")
            if not sequence:
                raise ValueError("DNA chains require a sequence.")
            mods = _coerce_modifications(chain.get("modifications", []))
            complex_builder.dna(
                ids,
                sequence,
                modifications=mods,
                cyclic=bool(chain.get("cyclic", False)),
            )
        elif chain_type == "rna":
            sequence = chain.get("sequence")
            if not sequence:
                raise ValueError("RNA chains require a sequence.")
            mods = _coerce_modifications(chain.get("modifications", []))
            complex_builder.rna(
                ids,
                sequence,
                modifications=mods,
                cyclic=bool(chain.get("cyclic", False)),
            )
        elif chain_type == "ligand":
            smiles = chain.get("smiles")
            ccd = chain.get("ccd")
            if (smiles is None) == (ccd is None):
                raise ValueError("Ligands require exactly one of smiles or ccd.")
            complex_builder.ligand(ids, ccd=ccd, smiles=smiles)
        else:
            raise ValueError(f"Unknown chain type: {chain_type}")

    for constraint in constraints or []:
        constraint_type = str(constraint.get("type", "")).lower()
        if constraint_type == "bond":
            complex_builder.bond(constraint["atom1"], constraint["atom2"])
        elif constraint_type == "pocket":
            complex_builder.pocket(
                constraint["binder"],
                constraint["contacts"],
                max_distance=float(constraint.get("max_distance", 6.0)),
                force=bool(constraint.get("force", False)),
            )
        elif constraint_type == "contact":
            complex_builder.contact(
                constraint["token1"],
                constraint["token2"],
                max_distance=float(constraint.get("max_distance", 6.0)),
                force=bool(constraint.get("force", False)),
            )
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")

    return complex_builder


def _write_structure(
    *,
    output_path: str,
    output_format: str,
    mmcif_text: str | None,
    bcif_bytes: bytes | None,
) -> str:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "bcif":
        if bcif_bytes is None:
            raise ValueError("bcif_bytes is required for BCIF output.")
        path.write_bytes(bcif_bytes)
    else:
        if mmcif_text is None:
            raise ValueError("mmcif_text is required for CIF output.")
        path.write_text(mmcif_text, encoding="utf-8")
    return str(path)


def _summarize_features(features: dict[str, Any]) -> dict[str, list[int]]:
    summary: dict[str, list[int]] = {}
    for key, value in features.items():
        if torch.is_tensor(value):
            summary[key] = list(value.shape)
        elif isinstance(value, np.ndarray):
            summary[key] = list(value.shape)
    return summary


def _save_features(
    *,
    output_path: str,
    output_format: str,
    features: dict[str, Any],
) -> str:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "torch":
        torch.save(features, path)
        return str(path)

    arrays: dict[str, np.ndarray] = {}
    for key, value in features.items():
        if torch.is_tensor(value):
            arrays[key] = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            arrays[key] = value
    np.savez_compressed(path, **arrays)
    return str(path)


def _infer_binder(chains: list[dict[str, Any]]) -> str | None:
    ligand_ids: list[str] = []
    for chain in chains:
        if str(chain.get("type", "")).lower() != "ligand":
            continue
        ids = chain.get("id")
        if isinstance(ids, str):
            ligand_ids.append(ids)
        elif isinstance(ids, (list, tuple)):
            ligand_ids.extend([str(item) for item in ids])
    if len(ligand_ids) == 1:
        return ligand_ids[0]
    return None


def _prune_jobs_locked() -> None:
    if len(_JOB_STORE) <= JOB_HISTORY_LIMIT:
        return
    for job_id, job in list(_JOB_STORE.items()):
        if len(_JOB_STORE) <= JOB_HISTORY_LIMIT:
            break
        if job.status in {"success", "error"}:
            _JOB_STORE.pop(job_id, None)


def _run_job(job_id: str, runner: Callable[[], dict[str, Any]]) -> None:
    with _JOB_LOCK:
        job = _JOB_STORE.get(job_id)
        if job is None:
            return
        job.status = "running"
        job.started_at = time.time()

    try:
        result = runner()
    except Exception:
        err = traceback.format_exc()
        with _JOB_LOCK:
            job = _JOB_STORE.get(job_id)
            if job is None:
                return
            job.status = "error"
            job.error = err
            job.finished_at = time.time()
        return

    with _JOB_LOCK:
        job = _JOB_STORE.get(job_id)
        if job is None:
            return
        job.status = "success"
        job.result = result
        job.finished_at = time.time()


def _submit_job(tool: str, runner: Callable[[], dict[str, Any]]) -> str:
    job_id = uuid.uuid4().hex
    record = JobRecord(
        job_id=job_id,
        tool=tool,
        status="queued",
        created_at=time.time(),
    )
    with _JOB_LOCK:
        _JOB_STORE[job_id] = record
        _prune_jobs_locked()
    _JOB_EXECUTOR.submit(_run_job, job_id, runner)
    return job_id


def _job_snapshot(job_id: str, include_result: bool) -> dict[str, Any]:
    with _JOB_LOCK:
        job = _JOB_STORE.get(job_id)
        if job is None:
            raise ValueError(f"Unknown job id: {job_id}")
        snapshot: dict[str, Any] = {
            "job_id": job.job_id,
            "tool": job.tool,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "result_available": job.status == "success",
        }
        if job.status == "error" and job.error:
            snapshot["error"] = job.error
        if include_result and job.status == "success":
            snapshot["result"] = job.result
        return snapshot


@mcp.tool()
def boltz2_fold_complex(
    chains: list[dict[str, Any]],
    *,
    name: str = "complex",
    constraints: list[dict[str, Any]] | None = None,
    cache_dir: str | None = DEFAULT_BOLTZ_CACHE,
    device: str | None = None,
    auto_download: bool = True,
    use_kernels: bool = True,
    affinity_mw_correction: bool = True,
    predict_overrides: dict[str, Any] | None = None,
    output_path: str | None = None,
    output_format: str | None = None,
    return_mmcif: bool = False,
    return_bcif_base64: bool = False,
    async_mode: bool = False,
) -> dict[str, Any]:
    """Fold a complex with Boltz2 and return mmCIF/BCIF output.

    chains: list of chain specs, each with keys:
      - type: protein | dna | rna | ligand
      - id: string or list of chain ids
      - sequence: required for protein/dna/rna
      - modifications: optional list of {position, ccd}
      - msa_a3m: optional A3M content (protein only)
      - cyclic: optional boolean
      - smiles or ccd: required for ligands

    constraints: optional list with type = bond | pocket | contact.
    Set async_mode to true to enqueue a long-running job and poll with boltz2_job.
    """
    def run() -> dict[str, Any]:
        model = _get_boltz2(
            cache_dir,
            device,
            auto_download,
            use_kernels,
            affinity_mw_correction,
        )
        complex_builder = _build_fold_complex(
            model,
            name=name,
            chains=chains,
            constraints=constraints,
        )

        overrides = dict(predict_overrides or {})
        prediction = complex_builder._predict_structure(**overrides)  # noqa: SLF001

        mmcif_text = None
        bcif_bytes = None

        output_kind = _resolve_output_format(output_path, output_format)
        if output_kind is None and output_path is not None:
            output_kind = "cif"

        if output_kind or return_mmcif or return_bcif_base64:
            mmcif_text = to_mmcif(
                prediction.structure,
                plddts=prediction.plddt,
                boltz2=True,
            )
        if output_kind == "bcif" or return_bcif_base64:
            bcif_bytes = bcif_bytes_from_mmcif(mmcif_text or "")

        output_written = None
        if output_path and output_kind:
            output_written = _write_structure(
                output_path=output_path,
                output_format=output_kind,
                mmcif_text=mmcif_text,
                bcif_bytes=bcif_bytes,
            )

        result: dict[str, Any] = {
            "name": name,
            "confidence_score": prediction.confidence_score,
            "output_path": output_written,
            "output_format": output_kind,
        }
        if return_mmcif and mmcif_text is not None:
            result["mmcif"] = mmcif_text
        if return_bcif_base64 and bcif_bytes is not None:
            result["bcif_base64"] = base64.b64encode(bcif_bytes).decode("ascii")
        return result

    if async_mode:
        job_id = _submit_job("boltz2_fold_complex", run)
        return {"job_id": job_id, "status": "queued"}

    return run()


@mcp.tool()
def boltz2_affinity(
    chains: list[dict[str, Any]],
    *,
    name: str = "complex",
    binder: str | None = None,
    constraints: list[dict[str, Any]] | None = None,
    cache_dir: str | None = DEFAULT_BOLTZ_CACHE,
    device: str | None = None,
    auto_download: bool = True,
    use_kernels: bool = True,
    affinity_mw_correction: bool = True,
    use_structure_prediction: bool = False,
    crop_affinity: bool = False,
    override_method: str | None = None,
    structure_predict_overrides: dict[str, Any] | None = None,
    affinity_predict_overrides: dict[str, Any] | None = None,
    structure_output_path: str | None = None,
    structure_output_format: str | None = None,
    return_mmcif: bool = False,
    return_bcif_base64: bool = False,
    async_mode: bool = False,
) -> dict[str, Any]:
    """Predict Boltz2 affinity for a ligand binder.

    Uses the same chain schema as boltz2_fold_complex. Set binder when multiple ligands
    are present. Set use_structure_prediction to condition affinity on a folded complex.
    Set async_mode to true to enqueue a long-running job and poll with boltz2_job.
    """
    def run() -> dict[str, Any]:
        model = _get_boltz2(
            cache_dir,
            device,
            auto_download,
            use_kernels,
            affinity_mw_correction,
        )
        complex_builder = _build_fold_complex(
            model,
            name=name,
            chains=chains,
            constraints=constraints,
        )

        binder_id = binder or _infer_binder(chains)

        structure_prediction = None
        output_kind = _resolve_output_format(structure_output_path, structure_output_format)
        if output_kind is None and structure_output_path is not None:
            output_kind = "cif"

        if use_structure_prediction or output_kind or return_mmcif or return_bcif_base64:
            structure_prediction = complex_builder._predict_structure(  # noqa: SLF001
                **(structure_predict_overrides or {})
            )

        affinity = complex_builder.get_affinity(
            binder=binder_id,
            use_structure_prediction=use_structure_prediction,
            crop_affinity=crop_affinity,
            override_method=override_method,
            **(affinity_predict_overrides or {}),
        )

        result: dict[str, Any] = {
            "name": name,
            "binder": binder_id,
            "affinity": {
                "ic50": affinity.ic50,
                "binding_probability": affinity.binding_probability,
                "ic50_1": affinity.ic50_1,
                "binding_probability_1": affinity.binding_probability_1,
                "ic50_2": affinity.ic50_2,
                "binding_probability_2": affinity.binding_probability_2,
            },
        }

        if structure_prediction is not None:
            mmcif_text = None
            bcif_bytes = None
            if output_kind or return_mmcif or return_bcif_base64:
                mmcif_text = to_mmcif(
                    structure_prediction.structure,
                    plddts=structure_prediction.plddt,
                    boltz2=True,
                )
            if output_kind == "bcif" or return_bcif_base64:
                bcif_bytes = bcif_bytes_from_mmcif(mmcif_text or "")

            output_written = None
            if structure_output_path and output_kind:
                output_written = _write_structure(
                    output_path=structure_output_path,
                    output_format=output_kind,
                    mmcif_text=mmcif_text,
                    bcif_bytes=bcif_bytes,
                )

            result["structure"] = {
                "confidence_score": structure_prediction.confidence_score,
                "output_path": output_written,
                "output_format": output_kind,
            }
            if return_mmcif and mmcif_text is not None:
                result["structure"]["mmcif"] = mmcif_text
            if return_bcif_base64 and bcif_bytes is not None:
                result["structure"]["bcif_base64"] = base64.b64encode(bcif_bytes).decode(
                    "ascii"
                )

        return result

    if async_mode:
        job_id = _submit_job("boltz2_affinity", run)
        return {"job_id": job_id, "status": "queued"}

    return run()


@mcp.tool()
def boltz2_job(job_id: str, *, include_result: bool = False) -> dict[str, Any]:
    """Check status for a background Boltz2 job (include_result returns output on success)."""
    return _job_snapshot(job_id, include_result)


@mcp.tool()
def boltzgen_antibody_design(
    antigen_path: str,
    *,
    antigen_chain: str = "A",
    binding_range: str | None = "10..40",
    heavy_id: str = "H",
    light_id: str = "L",
    heavy_length: int = 120,
    light_length: int = 110,
    heavy_sequence: str | None = None,
    light_sequence: str | None = None,
    name: str = "antibody_design",
    base_dir: str | None = None,
    mol_dir: str | None = None,
    auto_download: bool = True,
    cache_dir: str | None = None,
    token: str | None = None,
    force_download: bool = False,
    output_path: str | None = None,
    output_format: str | None = None,
) -> dict[str, Any]:
    """Prepare BoltzGen antibody design features from an antigen structure."""

    def resolve_design_sequence(explicit: str | None, length: int, label: str) -> str:
        if explicit:
            return explicit
        if length <= 0:
            raise ValueError(f"{label} length must be positive.")
        return str(length)

    antigen_file = Path(antigen_path).expanduser().resolve()
    if not antigen_file.exists():
        raise FileNotFoundError(f"Antigen file not found: {antigen_file}")

    base_path = Path(base_dir).expanduser().resolve() if base_dir else antigen_file.parent

    gen = _get_boltzgen(mol_dir, auto_download, cache_dir, token, force_download)
    design = gen.design(name, base_dir=base_path).file(
        antigen_file,
        include=[{"chain": {"id": antigen_chain}}],
        binding_types=(
            [{"chain": {"id": antigen_chain, "binding": binding_range}}]
            if binding_range
            else None
        ),
    )

    heavy_seq = resolve_design_sequence(heavy_sequence, heavy_length, "Heavy chain")
    light_seq = resolve_design_sequence(light_sequence, light_length, "Light chain")

    design.protein(heavy_id, heavy_seq).protein(light_id, light_seq)

    features = design.to_features()
    summary = _summarize_features(features)

    output_written = None
    feature_format = None
    if output_path:
        feature_format = _resolve_feature_output_format(output_path, output_format)
        output_written = _save_features(
            output_path=output_path,
            output_format=feature_format,
            features=features,
        )

    return {
        "name": name,
        "feature_keys": sorted(features.keys()),
        "feature_shapes": summary,
        "output_path": output_written,
        "output_format": feature_format,
    }


@mcp.tool()
def boltzgen_peptide_design(
    target_path: str,
    *,
    target_chain: str = "A",
    binding_range: str | None = "10..40",
    peptide_id: str = "P",
    peptide_spec: str = "12",
    peptide_cyclic: bool = False,
    peptide_secondary_structure: str | None = None,
    name: str = "peptide_binder",
    base_dir: str | None = None,
    mol_dir: str | None = None,
    auto_download: bool = True,
    cache_dir: str | None = None,
    token: str | None = None,
    force_download: bool = False,
    output_path: str | None = None,
    output_format: str | None = None,
) -> dict[str, Any]:
    """Prepare BoltzGen peptide binder design features from a target structure."""
    target_file = Path(target_path).expanduser().resolve()
    if not target_file.exists():
        raise FileNotFoundError(f"Target file not found: {target_file}")

    base_path = Path(base_dir).expanduser().resolve() if base_dir else target_file.parent

    gen = _get_boltzgen(mol_dir, auto_download, cache_dir, token, force_download)
    design = gen.design(name, base_dir=base_path).file(
        target_file,
        include=[{"chain": {"id": target_chain}}],
        binding_types=(
            [{"chain": {"id": target_chain, "binding": binding_range}}]
            if binding_range
            else None
        ),
    )

    peptide_kwargs: dict[str, Any] = {}
    if peptide_secondary_structure:
        peptide_kwargs["secondary_structure"] = peptide_secondary_structure

    design.protein(
        peptide_id,
        peptide_spec,
        cyclic=peptide_cyclic,
        **peptide_kwargs,
    )

    features = design.to_features()
    summary = _summarize_features(features)

    output_written = None
    feature_format = None
    if output_path:
        feature_format = _resolve_feature_output_format(output_path, output_format)
        output_written = _save_features(
            output_path=output_path,
            output_format=feature_format,
            features=features,
        )

    return {
        "name": name,
        "feature_keys": sorted(features.keys()),
        "feature_shapes": summary,
        "output_path": output_written,
        "output_format": feature_format,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
