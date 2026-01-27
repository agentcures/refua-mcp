from __future__ import annotations

import base64
import importlib.util
import statistics
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from refua import Boltz2, BoltzGen, Complex, SmallMolecule
    from refua.admet import AdmetPredictor  # type: ignore[reportMissingImports]
else:
    Boltz2 = BoltzGen = Complex = SmallMolecule = Any
    AdmetPredictor = Any

mcp = FastMCP("refua-mcp")

DEFAULT_BOLTZ_CACHE = str(Path("~/.boltz").expanduser())
JOB_HISTORY_LIMIT = 100
JOB_MAX_WORKERS = 1
POLL_MIN_SECONDS = 10
POLL_MAX_SECONDS = 30
POLL_FRACTION = 0.2
ADMET_DEPENDENCIES = ("transformers", "huggingface_hub")


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _admet_available() -> bool:
    return all(_module_available(dep) for dep in ADMET_DEPENDENCIES)


_ADMET_AVAILABLE = _admet_available()


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


def _clamp_seconds(value: float, minimum: int, maximum: int) -> int:
    return int(max(minimum, min(maximum, round(value))))


def _recommend_poll_seconds(estimate_seconds: float | None, queue_position: int) -> int:
    if estimate_seconds is None:
        return min(POLL_MAX_SECONDS, POLL_MIN_SECONDS + queue_position * 5)
    estimate_seconds = max(estimate_seconds, float(POLL_MIN_SECONDS))
    return _clamp_seconds(
        estimate_seconds * POLL_FRACTION,
        POLL_MIN_SECONDS,
        POLL_MAX_SECONDS,
    )


def _median_runtime_seconds_locked() -> float | None:
    runtimes = [
        job.finished_at - job.started_at
        for job in _JOB_STORE.values()
        if job.started_at is not None and job.finished_at is not None
    ]
    if not runtimes:
        return None
    return float(statistics.median(runtimes))


def _queue_position_locked(job_id: str) -> int:
    position = 0
    for existing_id, job in _JOB_STORE.items():
        if existing_id == job_id:
            break
        if job.status in {"queued", "running"}:
            position += 1
    return position


def _queue_depth_locked() -> int:
    return sum(1 for job in _JOB_STORE.values() if job.status == "queued")


@lru_cache(maxsize=4)
def _get_boltz2(
    cache_dir: str | None,
    device: str | None,
    auto_download: bool,
    use_kernels: bool,
    affinity_mw_correction: bool,
) -> Boltz2:
    from refua import Boltz2

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
    from refua import BoltzGen

    return BoltzGen(
        mol_dir=mol_dir,
        auto_download=auto_download,
        cache_dir=cache_dir,
        token=token,
        force_download=force_download,
    )


def _parse_boltz_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    opts = dict(options or {})
    known = {
        "cache_dir",
        "device",
        "auto_download",
        "use_kernels",
        "affinity_mw_correction",
        "predict_args",
        "affinity_predict_args",
    }
    unknown = set(opts) - known
    if unknown:
        raise ValueError(f"Unknown boltz options: {sorted(unknown)}")
    return opts


def _parse_boltzgen_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    opts = dict(options or {})
    known = {"mol_dir", "auto_download", "cache_dir", "token", "force_download"}
    unknown = set(opts) - known
    if unknown:
        raise ValueError(f"Unknown boltzgen options: {sorted(unknown)}")
    return opts


def _parse_admet_options(admet: Any) -> tuple[str, dict[str, Any]]:
    if admet is None:
        return "auto", {}
    if admet is False:
        return "off", {}
    if admet is True:
        return "on", {}
    if isinstance(admet, str):
        mode = str(admet).lower()
        if mode not in {"auto", "on", "off"}:
            raise ValueError("admet must be 'auto', 'on', 'off', a bool, or a dict.")
        return mode, {}
    if isinstance(admet, Mapping):
        opts = dict(admet)
        mode_value = opts.pop("mode", None)
        enabled = opts.pop("enabled", None)
        if enabled is not None:
            mode_value = "on" if bool(enabled) else "off"
        if mode_value is None:
            mode_value = "on"
        mode = str(mode_value).lower()
        if mode not in {"auto", "on", "off"}:
            raise ValueError("admet.mode must be 'auto', 'on', or 'off'.")
        known = {
            "ligands",
            "model_variant",
            "max_new_tokens",
            "include_scoring",
            "task_ids",
        }
        unknown = set(opts) - known
        if unknown:
            raise ValueError(f"Unknown admet options: {sorted(unknown)}")
        return mode, opts
    raise ValueError("admet must be 'auto', 'on', 'off', a bool, a dict, or None.")


def _normalize_admet_ligands(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("admet.ligands cannot be empty.")
        return [str(item) for item in value]
    raise ValueError("admet.ligands must be a string or list of strings.")


def _select_admet_ligands(
    ligand_specs: list[dict[str, Any]],
    requested: list[str] | None,
    alias_map: Mapping[str, str],
) -> list[dict[str, Any]]:
    if requested is None:
        return ligand_specs
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ligand in requested:
        ligand_id = alias_map.get(str(ligand), str(ligand))
        match = next(
            (spec for spec in ligand_specs if spec["ligand_id"] == ligand_id),
            None,
        )
        if match is None:
            raise ValueError(f"Unknown ligand id for admet: {ligand}")
        if ligand_id in seen:
            continue
        seen.add(ligand_id)
        resolved.append(match)
    return resolved


def _normalize_admet_task_ids(
    task_ids: Iterable[Any] | None,
) -> tuple[str, ...] | None:
    if task_ids is None:
        return None
    if isinstance(task_ids, str):
        raise ValueError("task_ids must be a list of strings.")
    normalized = tuple(str(task_id) for task_id in task_ids)
    if not normalized:
        raise ValueError("task_ids cannot be empty.")
    return normalized


def _build_boltz2_from_options(options: Mapping[str, Any] | None) -> Boltz2:
    opts = _parse_boltz_options(options)
    cache_dir = opts.get("cache_dir", DEFAULT_BOLTZ_CACHE)
    device = opts.get("device")
    auto_download = bool(opts.get("auto_download", True))
    use_kernels = bool(opts.get("use_kernels", True))
    affinity_mw_correction = bool(opts.get("affinity_mw_correction", True))
    predict_args = opts.get("predict_args")
    affinity_predict_args = opts.get("affinity_predict_args")

    if predict_args is not None or affinity_predict_args is not None:
        from refua import Boltz2

        return Boltz2(
            cache_dir=cache_dir,
            device=device,
            auto_download=auto_download,
            use_kernels=use_kernels,
            affinity_mw_correction=affinity_mw_correction,
            predict_args=predict_args,
            affinity_predict_args=affinity_predict_args,
        )

    return _get_boltz2(
        cache_dir,
        device,
        auto_download,
        use_kernels,
        affinity_mw_correction,
    )


def _build_boltzgen_from_options(options: Mapping[str, Any] | None) -> BoltzGen:
    opts = _parse_boltzgen_options(options)
    mol_dir = opts.get("mol_dir")
    auto_download = bool(opts.get("auto_download", True))
    cache_dir = opts.get("cache_dir")
    token = opts.get("token")
    force_download = bool(opts.get("force_download", False))
    return _get_boltzgen(mol_dir, auto_download, cache_dir, token, force_download)


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
            raise ValueError(
                "Modification entries must be dicts or (position, ccd) tuples."
            )
    return result


def _coerce_chain_ids(value: Any | None) -> str | tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Chain ids cannot be empty.")
        return tuple(str(item) for item in value)
    raise ValueError("Chain id must be a string or list of strings.")


def _resolve_entity_ids(entity: Mapping[str, Any]) -> str | tuple[str, ...] | None:
    if "id" in entity:
        return _coerce_chain_ids(entity.get("id"))
    if "ids" in entity:
        return _coerce_chain_ids(entity.get("ids"))
    return None


def _resolve_msa(entity: Mapping[str, Any]) -> object | None:
    msa_a3m = entity.get("msa_a3m")
    if not msa_a3m:
        return None
    from refua.boltz.api import msa_from_a3m

    return msa_from_a3m(
        str(msa_a3m),
        taxonomy=entity.get("msa_taxonomy"),
        max_seqs=entity.get("msa_max_seqs"),
    )


@lru_cache(maxsize=128)
def _load_ccd_mol(mol_dir: str, ccd: str) -> Any:
    from refua.boltz.data.mol import load_molecules

    return load_molecules(mol_dir, [ccd])[ccd]


def _resolve_boltz_mol_dir(
    boltz_model: Boltz2 | None,
    boltz_options: Mapping[str, Any],
) -> Path | None:
    if boltz_model is not None:
        return Path(boltz_model.mol_dir)
    cache_dir = boltz_options.get("cache_dir") or DEFAULT_BOLTZ_CACHE
    return Path(cache_dir).expanduser() / "mols"


def _make_ligand(
    *,
    smiles: str | None,
    ccd: str | None,
    mol_dir: Path | None,
) -> SmallMolecule:
    from refua import SmallMolecule

    if (smiles is None) == (ccd is None):
        raise ValueError("Ligands require exactly one of smiles or ccd.")
    if smiles is not None:
        return SmallMolecule.from_smiles(str(smiles))
    if mol_dir is None:
        raise ValueError("CCD ligands require boltz mol_dir assets.")
    mol = _load_ccd_mol(str(mol_dir), str(ccd))
    return SmallMolecule.from_mol(mol, name=str(ccd))


def _build_complex_from_spec(
    *,
    name: str,
    base_dir: str | None,
    entities: list[dict[str, Any]],
    boltz_mol_dir: Path | None,
) -> tuple[Complex, dict[str, str], bool, bool]:
    from refua import Binder, Complex, DNA, Protein, RNA

    if not entities:
        raise ValueError("entities must include at least one entity spec.")

    complex_spec = Complex(name=name, base_dir=base_dir)
    ligand_alias_map: dict[str, str] = {}
    ligand_index = 1
    has_boltz = False
    has_boltzgen = False

    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError("Each entity must be a dict.")
        entity_type = str(entity.get("type", "")).lower()
        if not entity_type:
            raise ValueError("Entity is missing type.")

        if entity_type == "protein":
            sequence = entity.get("sequence")
            if not sequence:
                raise ValueError("Protein entities require a sequence.")
            ids = _resolve_entity_ids(entity)
            complex_spec.add(
                Protein(
                    str(sequence),
                    ids=ids,
                    modifications=_coerce_modifications(
                        entity.get("modifications", [])
                    ),
                    msa=_resolve_msa(entity),
                    binding_types=entity.get("binding_types"),
                    secondary_structure=entity.get("secondary_structure"),
                    cyclic=bool(entity.get("cyclic", False)),
                )
            )
            has_boltz = True
            continue

        if entity_type == "dna":
            sequence = entity.get("sequence")
            if not sequence:
                raise ValueError("DNA entities require a sequence.")
            ids = _resolve_entity_ids(entity)
            complex_spec.add(
                DNA(
                    str(sequence),
                    ids=ids,
                    modifications=_coerce_modifications(
                        entity.get("modifications", [])
                    ),
                    cyclic=bool(entity.get("cyclic", False)),
                )
            )
            has_boltz = True
            continue

        if entity_type == "rna":
            sequence = entity.get("sequence")
            if not sequence:
                raise ValueError("RNA entities require a sequence.")
            ids = _resolve_entity_ids(entity)
            complex_spec.add(
                RNA(
                    str(sequence),
                    ids=ids,
                    modifications=_coerce_modifications(
                        entity.get("modifications", [])
                    ),
                    cyclic=bool(entity.get("cyclic", False)),
                )
            )
            has_boltz = True
            continue

        if entity_type == "binder":
            ids = _resolve_entity_ids(entity)
            spec = entity.get("spec")
            length = entity.get("length")
            if spec is None and length is None:
                spec = entity.get("sequence")
            if length is not None:
                length = int(length)
            complex_spec.add(
                Binder(
                    spec=spec,
                    length=length,
                    ids=ids,
                    binding_types=entity.get("binding_types"),
                    secondary_structure=entity.get("secondary_structure"),
                    cyclic=bool(entity.get("cyclic", False)),
                )
            )
            has_boltzgen = True
            continue

        if entity_type == "ligand":
            ligand = _make_ligand(
                smiles=entity.get("smiles"),
                ccd=entity.get("ccd"),
                mol_dir=boltz_mol_dir,
            )
            complex_spec.add(ligand)
            alias_value = entity.get("id", entity.get("ids"))
            if alias_value is not None:
                if isinstance(alias_value, (list, tuple)):
                    if len(alias_value) != 1:
                        raise ValueError("Ligand id must be a single string.")
                    alias = str(alias_value[0])
                else:
                    alias = str(alias_value)
                expected = f"L{ligand_index}"
                if alias.startswith("L") and alias[1:].isdigit() and alias != expected:
                    raise ValueError(
                        "Ligand id aliases cannot shadow unified ids. "
                        "Omit the alias or use a non-L name."
                    )
                if alias in ligand_alias_map:
                    raise ValueError(f"Duplicate ligand alias: {alias}")
                ligand_alias_map[alias] = expected
            ligand_index += 1
            has_boltz = True
            continue

        if entity_type == "file":
            path_value = entity.get("path")
            if not path_value:
                raise ValueError("File entities require a path.")
            file_path = Path(path_value).expanduser().resolve()
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            complex_spec.file(
                file_path,
                include=entity.get("include"),
                exclude=entity.get("exclude"),
                include_proximity=entity.get("include_proximity"),
                binding_types=entity.get("binding_types"),
                structure_groups=entity.get("structure_groups"),
                design=entity.get("design"),
                not_design=entity.get("not_design"),
                secondary_structure=entity.get("secondary_structure"),
                design_insertions=entity.get("design_insertions"),
                fuse=entity.get("fuse"),
                msa=entity.get("msa"),
                use_assembly=entity.get("use_assembly"),
                reset_res_index=entity.get("reset_res_index"),
                extra=entity.get("extra") or {},
            )
            has_boltzgen = True
            continue

        raise ValueError(f"Unknown entity type: {entity_type}")

    return complex_spec, ligand_alias_map, has_boltz, has_boltzgen


def _map_chain_id(value: Any, alias_map: Mapping[str, str]) -> str:
    return alias_map.get(str(value), str(value))


def _map_atom_ref(value: Any, alias_map: Mapping[str, str]) -> tuple[Any, Any, Any]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        chain, residue, atom = value
        return (_map_chain_id(chain, alias_map), residue, atom)
    raise ValueError("Bond atom references must be 3-item sequences.")


def _map_token_ref(value: Any, alias_map: Mapping[str, str]) -> tuple[Any, Any]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        chain, token = value
        return (_map_chain_id(chain, alias_map), token)
    raise ValueError("Token references must be 2-item sequences.")


def _apply_constraints(
    complex_spec: Complex,
    constraints: list[dict[str, Any]] | None,
    alias_map: Mapping[str, str],
) -> None:
    for constraint in constraints or []:
        constraint_type = str(constraint.get("type", "")).lower()
        if constraint_type == "bond":
            complex_spec.bond(
                _map_atom_ref(constraint["atom1"], alias_map),
                _map_atom_ref(constraint["atom2"], alias_map),
            )
            continue
        if constraint_type == "pocket":
            binder = constraint.get("binder")
            if binder is None:
                raise ValueError("Pocket constraints require a binder.")
            contacts = constraint.get("contacts")
            if not contacts:
                raise ValueError("Pocket constraints require contacts.")
            complex_spec.pocket(
                _map_chain_id(binder, alias_map),
                contacts=[_map_token_ref(contact, alias_map) for contact in contacts],
                max_distance=float(constraint.get("max_distance", 6.0)),
                force=bool(constraint.get("force", False)),
            )
            continue
        if constraint_type == "contact":
            complex_spec.contact(
                _map_token_ref(constraint["token1"], alias_map),
                _map_token_ref(constraint["token2"], alias_map),
                max_distance=float(constraint.get("max_distance", 6.0)),
                force=bool(constraint.get("force", False)),
            )
            continue
        raise ValueError(f"Unknown constraint type: {constraint_type}")


def _resolve_affinity_request(
    affinity: Any,
    alias_map: Mapping[str, str],
) -> tuple[bool, str | None]:
    if affinity is None or affinity is False:
        return False, None
    if affinity is True:
        return True, None
    if isinstance(affinity, dict):
        binder = affinity.get("binder")
        if binder is None:
            return True, None
        return True, _map_chain_id(binder, alias_map)
    raise ValueError("affinity must be a bool or dict with optional binder.")


def _resolve_output_format(
    output_path: str | None, output_format: str | None
) -> str | None:
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
    import numpy as np
    import torch

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
    import numpy as np
    import torch

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
        now = time.time()
        snapshot: dict[str, Any] = {
            "job_id": job.job_id,
            "tool": job.tool,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "result_available": job.status == "success",
        }
        if job.status in {"queued", "running"}:
            queue_position = _queue_position_locked(job_id)
            queue_depth = _queue_depth_locked()
            avg_runtime = _median_runtime_seconds_locked()
            estimate_seconds: float | None = None

            snapshot["queue_position"] = queue_position
            snapshot["queue_depth"] = queue_depth
            if avg_runtime is not None:
                snapshot["average_runtime_seconds"] = avg_runtime
                if job.status == "queued" and queue_position > 0:
                    estimate_seconds = avg_runtime * queue_position
                    snapshot["estimated_start_seconds"] = estimate_seconds
                elif job.status == "running":
                    started_at = job.started_at or job.created_at
                    elapsed = max(0.0, now - started_at)
                    estimate_seconds = max(avg_runtime - elapsed, 0.0)
                    snapshot["estimated_remaining_seconds"] = estimate_seconds

            snapshot["recommended_poll_seconds"] = _recommend_poll_seconds(
                estimate_seconds,
                queue_position,
            )
        if job.status == "error" and job.error:
            snapshot["error"] = job.error
        if include_result and job.status == "success":
            snapshot["result"] = job.result
        return snapshot


@mcp.tool()
def refua_complex(
    entities: list[dict[str, Any]],
    *,
    name: str = "complex",
    base_dir: str | None = None,
    constraints: list[dict[str, Any]] | None = None,
    affinity: bool | dict[str, Any] | None = None,
    action: str = "fold",
    run_boltz: bool | None = None,
    run_boltzgen: bool | None = None,
    boltz: dict[str, Any] | None = None,
    boltzgen: dict[str, Any] | None = None,
    admet: bool | str | dict[str, Any] | None = None,
    structure_output_path: str | None = None,
    structure_output_format: str | None = None,
    return_mmcif: bool = False,
    return_bcif_base64: bool = False,
    feature_output_path: str | None = None,
    feature_output_format: str | None = None,
    async_mode: bool = False,
) -> dict[str, Any]:
    """Run a unified Refua Complex spec.

    entities: list of entity dicts with keys:
      - type: protein | dna | rna | binder | ligand | file
      - protein: sequence (required), id/ids, modifications, msa_a3m, binding_types,
        secondary_structure, cyclic
      - dna/rna: sequence (required), id/ids, modifications, cyclic
      - binder: spec or length (required), id/ids, binding_types, secondary_structure, cyclic
      - ligand: smiles or ccd (required), optional id alias (maps to L1/L2 order)
      - file: path (required) plus include/exclude/binding_types/etc.

    constraints: list of {type: bond|pocket|contact}. Ligand references use L1/L2
    (or the provided ligand id alias). DNA/RNA entities are Boltz2-only.
    action can be "fold" or "affinity".
    admet: run optional ADMET predictions for ligand SMILES. Use true/"on" to
    force, false/"off" to disable, or omit for auto (runs when SMILES ligands
    are present and ADMET deps are installed). Dict options: mode, ligands,
    model_variant, max_new_tokens, include_scoring, task_ids.
    Folding can take minutes depending on inputs and hardware; use async_mode for
    long runs and poll refua_job sparingly (for example, every 10-30 seconds) or
    follow the recommended_poll_seconds returned by refua_job.
    """

    def run() -> dict[str, Any]:
        action_value = str(action or "fold").lower()
        if action_value not in {"fold", "affinity"}:
            raise ValueError("action must be 'fold' or 'affinity'.")

        boltz_opts = _parse_boltz_options(boltz)
        boltzgen_opts = _parse_boltzgen_options(boltzgen)
        admet_mode, admet_opts = _parse_admet_options(admet)

        ligand_specs: list[dict[str, Any]] = []
        ligand_index = 1
        for item in entities:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("type", "")).lower() == "ligand":
                ligand_id = f"L{ligand_index}"
                ligand_index += 1
                smiles = item.get("smiles")
                if smiles is not None:
                    ligand_specs.append({"ligand_id": ligand_id, "smiles": str(smiles)})

        entity_types = [str(item.get("type", "")).lower() for item in entities]
        has_boltz_entities = any(
            kind in {"protein", "dna", "rna", "ligand"} for kind in entity_types
        )
        has_boltzgen_entities = any(kind in {"binder", "file"} for kind in entity_types)
        wants_affinity = affinity not in (None, False)

        run_boltz_local = (
            bool(run_boltz)
            if run_boltz is not None
            else bool(has_boltz_entities or constraints or wants_affinity)
        )
        run_boltzgen_local = (
            bool(run_boltzgen)
            if run_boltzgen is not None
            else bool(has_boltzgen_entities)
        )

        if action_value == "affinity":
            run_boltz_local = True
            run_boltzgen_local = False

        if constraints and not run_boltz_local:
            raise ValueError("constraints require run_boltz=true.")
        if wants_affinity and not run_boltz_local and action_value == "fold":
            raise ValueError("affinity requests require run_boltz=true.")

        boltz_model = None
        if run_boltz_local or action_value == "affinity":
            boltz_model = _build_boltz2_from_options(boltz_opts)

        has_ccd = any(
            str(item.get("type", "")).lower() == "ligand"
            and item.get("ccd") is not None
            for item in entities
        )
        boltz_mol_dir = None
        if has_ccd:
            boltz_mol_dir = _resolve_boltz_mol_dir(boltz_model, boltz_opts)
            if boltz_mol_dir is None or not boltz_mol_dir.exists():
                raise FileNotFoundError(
                    "CCD ligands require Boltz2 molecule assets. "
                    "Set boltz.cache_dir or enable run_boltz with auto_download."
                )

        complex_spec, ligand_alias_map, _, _ = _build_complex_from_spec(
            name=name,
            base_dir=base_dir,
            entities=entities,
            boltz_mol_dir=boltz_mol_dir,
        )

        _apply_constraints(complex_spec, constraints, ligand_alias_map)

        admet_output: dict[str, Any] | None = None
        if admet_mode != "off":
            has_smiles_ligands = bool(ligand_specs)
            wants_admet = admet_mode == "on" or (
                admet_mode == "auto" and has_smiles_ligands
            )
            if wants_admet:
                if not _ADMET_AVAILABLE:
                    if admet_mode == "on":
                        raise ValueError(
                            "ADMET requested but refua[admet] is not installed."
                        )
                    admet_output = {
                        "status": "unavailable",
                        "reason": "Install refua[admet] to enable ADMET predictions.",
                    }
                else:
                    requested = _normalize_admet_ligands(admet_opts.get("ligands"))
                    targets = _select_admet_ligands(
                        ligand_specs,
                        requested,
                        ligand_alias_map,
                    )
                    if not targets:
                        if admet_mode == "on":
                            raise ValueError(
                                "ADMET requested but no SMILES ligands are available."
                            )
                    else:
                        normalized_tasks = _normalize_admet_task_ids(
                            admet_opts.get("task_ids")
                        )
                        model_variant = str(admet_opts.get("model_variant", "9b-chat"))
                        max_new_tokens = int(admet_opts.get("max_new_tokens", 8))
                        include_scoring = bool(admet_opts.get("include_scoring", True))
                        results = []
                        for target in targets:
                            profile = _admet_analyze(
                                smiles=target["smiles"],
                                model_variant=model_variant,
                                max_new_tokens=max_new_tokens,
                                include_scoring=include_scoring,
                                task_ids=normalized_tasks,
                            )
                            profile["ligand_id"] = target["ligand_id"]
                            results.append(profile)
                        admet_output = {"status": "success", "results": results}

        affinity_requested, affinity_binder = _resolve_affinity_request(
            affinity, ligand_alias_map
        )

        if action_value == "affinity":
            affinity_result = complex_spec.affinity(
                binder=affinity_binder,
                boltz=boltz_model,
            )
            output: dict[str, Any] = {
                "name": name,
                "binder": affinity_binder,
                "affinity": {
                    "ic50": affinity_result.ic50,
                    "binding_probability": affinity_result.binding_probability,
                    "ic50_1": affinity_result.ic50_1,
                    "binding_probability_1": affinity_result.binding_probability_1,
                    "ic50_2": affinity_result.ic50_2,
                    "binding_probability_2": affinity_result.binding_probability_2,
                },
            }
            if ligand_alias_map:
                output["ligand_id_map"] = ligand_alias_map
            if admet_output is not None:
                output["admet"] = admet_output
            return output

        if affinity_requested:
            complex_spec.request_affinity(affinity_binder)

        boltzgen_model = None
        if run_boltzgen_local:
            boltzgen_model = _build_boltzgen_from_options(boltzgen_opts)

        result = complex_spec.fold(
            boltz=boltz_model,
            boltzgen=boltzgen_model,
            run_boltz=run_boltz_local,
            run_boltzgen=run_boltzgen_local,
        )

        output = {
            "name": name,
            "backend": result.backend,
            "chain_ids": result.chain_ids,
            "binder_sequences": result.binder_sequences,
        }
        if ligand_alias_map:
            output["ligand_id_map"] = ligand_alias_map
        if admet_output is not None:
            output["admet"] = admet_output

        if result.affinity is not None:
            output["affinity"] = {
                "ic50": result.affinity.ic50,
                "binding_probability": result.affinity.binding_probability,
                "ic50_1": result.affinity.ic50_1,
                "binding_probability_1": result.affinity.binding_probability_1,
                "ic50_2": result.affinity.ic50_2,
                "binding_probability_2": result.affinity.binding_probability_2,
            }

        if result.structure is None:
            if structure_output_path or return_mmcif or return_bcif_base64:
                raise ValueError(
                    "Structure output requested but no structure was produced."
                )
        else:
            output_kind = _resolve_output_format(
                structure_output_path,
                structure_output_format,
            )
            if output_kind is None and structure_output_path is not None:
                output_kind = "cif"

            mmcif_text = None
            bcif_bytes = None
            if output_kind == "cif" or return_mmcif:
                mmcif_text = result.to_mmcif()
            if output_kind == "bcif" or return_bcif_base64:
                bcif_bytes = result.to_bcif()

            output_written = None
            if structure_output_path and output_kind:
                output_written = _write_structure(
                    output_path=structure_output_path,
                    output_format=output_kind,
                    mmcif_text=mmcif_text,
                    bcif_bytes=bcif_bytes,
                )

            structure_info: dict[str, Any] = {
                "confidence_score": result.structure.confidence_score,
                "output_path": output_written,
                "output_format": output_kind,
            }
            if return_mmcif and mmcif_text is not None:
                structure_info["mmcif"] = mmcif_text
            if return_bcif_base64 and bcif_bytes is not None:
                structure_info["bcif_base64"] = base64.b64encode(bcif_bytes).decode(
                    "ascii"
                )
            output["structure"] = structure_info

        features = result.features
        if features is None:
            if feature_output_path:
                raise ValueError(
                    "Feature output requested but no features were produced."
                )
        else:
            features = dict(features)  # Convert Mapping to dict for type compatibility
            feature_format = None
            output_written = None
            if feature_output_path:
                feature_format = _resolve_feature_output_format(
                    feature_output_path, feature_output_format
                )
                output_written = _save_features(
                    output_path=feature_output_path,
                    output_format=feature_format,
                    features=features,
                )
            output["features"] = {
                "feature_keys": sorted(features.keys()),
                "feature_shapes": _summarize_features(features),
                "output_path": output_written,
                "output_format": feature_format,
            }

        return output

    if async_mode:
        job_id = _submit_job("refua_complex", run)
        return {"job_id": job_id, "status": "queued"}

    return run()


@mcp.tool()
def refua_job(job_id: str, *, include_result: bool = False) -> dict[str, Any]:
    """Check status for a background refua job.

    Responses may include recommended_poll_seconds plus queue/estimate metadata for
    queued or running jobs.
    """
    return _job_snapshot(job_id, include_result)


if _ADMET_AVAILABLE:

    @lru_cache(maxsize=4)
    def _get_admet_predictor(
        model_variant: str,
        task_ids: tuple[str, ...] | None,
    ) -> AdmetPredictor:
        from refua.admet import AdmetPredictor  # type: ignore[reportMissingImports]

        return AdmetPredictor(model_variant=model_variant, task_ids=task_ids)

    def _admet_analyze(
        *,
        smiles: str,
        model_variant: str,
        max_new_tokens: int,
        include_scoring: bool,
        task_ids: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        from refua.admet import AdmetScorer, admet_profile  # type: ignore[reportMissingImports]

        if task_ids is None:
            return admet_profile(
                smiles,
                model_variant=model_variant,
                max_new_tokens=max_new_tokens,
                include_scoring=include_scoring,
            )

        predictor = _get_admet_predictor(model_variant, task_ids)
        predictions, raw_outputs = predictor.predict(
            smiles,
            max_new_tokens=max_new_tokens,
        )
        result: dict[str, Any] = {
            "smiles": smiles,
            "predictions": predictions,
            "raw_outputs": raw_outputs,
            "missing_tasks": list(predictor.missing_task_ids),
        }
        if include_scoring:
            scorer = AdmetScorer()
            result.update(scorer.analyze_profile(predictions))
        return result

    @mcp.tool()
    def refua_admet_profile(
        smiles: str,
        *,
        model_variant: str = "9b-chat",
        max_new_tokens: int = 8,
        include_scoring: bool = True,
        task_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run model-based ADMET predictions for a SMILES string.

        Requires refua[admet] (transformers + huggingface_hub). Optionally pass
        task_ids to restrict the endpoints that are evaluated.
        """
        if not smiles:
            raise ValueError("smiles is required.")
        normalized_tasks = _normalize_admet_task_ids(task_ids)
        return _admet_analyze(
            smiles=str(smiles),
            model_variant=str(model_variant),
            max_new_tokens=int(max_new_tokens),
            include_scoring=bool(include_scoring),
            task_ids=normalized_tasks,
        )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
