from __future__ import annotations

import base64
import importlib.util
import json
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
from typing import TYPE_CHECKING, Annotated, Any, Callable, Iterable, Literal, Mapping

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from mcp.types import TaskExecutionMode
from mcp.types import TasksCallCapability
from mcp.types import Tool as McpTool
from mcp.types import ToolExecution
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from refua import Boltz2, BoltzGen, Complex, SmallMolecule
    from refua.admet import AdmetPredictor  # type: ignore[reportMissingImports]
else:
    Boltz2 = BoltzGen = Complex = SmallMolecule = Any
    AdmetPredictor = Any

SERVER_INSTRUCTIONS = """
Use the typed Refua tools directly instead of speculative probing.

Recommended sequence:
1) Read recipe resources (`refua://recipes/index` and `refua://recipes/{recipe_name}`)
   for canonical call shapes.
2) Call `refua_validate_spec` to normalize/validate before expensive work.
3) Execute with the focused tool:
   - `refua_fold` for structure/design folds
   - `refua_affinity` for affinity-only predictions
   - `refua_antibody_design` for antibody-heavy workflows
4) For long runs, set `async_mode=true` and poll `refua_job` using
   `recommended_poll_seconds` or `wait_for_terminal_seconds`.
"""

mcp = FastMCP("refua-mcp", instructions=SERVER_INSTRUCTIONS)

DEFAULT_BOLTZ_CACHE = str(Path("~/.boltz").expanduser())
JOB_HISTORY_LIMIT = 100
JOB_MAX_WORKERS = 1
POLL_MIN_SECONDS = 30
POLL_MAX_SECONDS = 120
POLL_QUEUE_STEP_SECONDS = 15
POLL_FRACTION = 0.35
LONG_POLL_MAX_WAIT_SECONDS = 900.0
LONG_POLL_MIN_SLEEP_SECONDS = 5.0
LONG_POLL_MAX_SLEEP_SECONDS = 120.0
ADMET_DEPENDENCIES = ("transformers", "huggingface_hub")


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _admet_available() -> bool:
    return all(_module_available(dep) for dep in ADMET_DEPENDENCIES)


_ADMET_AVAILABLE = _admet_available()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChainEntitySpec(StrictModel):
    id: str | None = None
    ids: list[str] | tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _validate_id_fields(self) -> "ChainEntitySpec":
        if self.id is not None and self.ids is not None:
            raise ValueError("Use either id or ids, not both.")
        if self.ids is not None and len(self.ids) == 0:
            raise ValueError("ids cannot be empty.")
        return self


class ModificationSpec(StrictModel):
    position: int
    ccd: str


class ProteinEntity(ChainEntitySpec):
    type: Literal["protein"]
    sequence: str
    modifications: list[ModificationSpec | tuple[int, str]] = Field(
        default_factory=list
    )
    msa_a3m: str | None = None
    msa_taxonomy: str | None = None
    msa_max_seqs: int | None = None
    binding_types: Any | None = None
    secondary_structure: Any | None = None
    cyclic: bool = False


class DNAEntity(ChainEntitySpec):
    type: Literal["dna"]
    sequence: str
    modifications: list[ModificationSpec | tuple[int, str]] = Field(
        default_factory=list
    )
    cyclic: bool = False


class RNAEntity(ChainEntitySpec):
    type: Literal["rna"]
    sequence: str
    modifications: list[ModificationSpec | tuple[int, str]] = Field(
        default_factory=list
    )
    cyclic: bool = False


class BinderEntity(ChainEntitySpec):
    type: Literal["binder"]
    spec: str | None = None
    sequence: str | None = None
    length: int | None = None
    binding_types: Any | None = None
    secondary_structure: Any | None = None
    cyclic: bool = False
    template_values: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_binder_input(self) -> "BinderEntity":
        if self.length is not None and self.length < 1:
            raise ValueError("binder length must be >= 1.")
        if self.spec is None and self.sequence is None and self.length is None:
            raise ValueError(
                "binder requires at least one of spec, sequence, or length."
            )
        return self


class PeptideEntity(ChainEntitySpec):
    type: Literal["peptide"]
    spec: str | None = None
    sequence: str | None = None
    length: int | None = None
    segment_lengths: tuple[int, int, int] | None = None
    disulfide: bool = False
    binding_types: Any | None = None
    secondary_structure: Any | None = None
    cyclic: bool | None = None
    template_values: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_peptide_input(self) -> "PeptideEntity":
        if self.length is not None and self.length < 1:
            raise ValueError("peptide length must be >= 1.")
        if self.segment_lengths is not None and any(
            item < 1 for item in self.segment_lengths
        ):
            raise ValueError("segment_lengths values must be >= 1.")
        return self


class AntibodyEntity(StrictModel):
    type: Literal["antibody"]
    ids: list[str] | tuple[str, str] | None = None
    heavy_id: str | None = None
    light_id: str | None = None
    heavy_cdr_lengths: tuple[int, int, int] | None = None
    light_cdr_lengths: tuple[int, int, int] | None = None
    heavy_binding_types: Any | None = None
    light_binding_types: Any | None = None
    heavy_secondary_structure: Any | None = None
    light_secondary_structure: Any | None = None
    heavy_cyclic: bool | None = None
    light_cyclic: bool | None = None
    heavy_spec: str | None = None
    heavy_sequence: str | None = None
    light_spec: str | None = None
    light_sequence: str | None = None
    heavy_template_values: dict[str, Any] | None = None
    light_template_values: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_antibody_input(self) -> "AntibodyEntity":
        if self.ids is not None:
            if len(self.ids) != 2:
                raise ValueError("antibody ids must include exactly two values.")
            if self.heavy_id is not None or self.light_id is not None:
                raise ValueError("use either ids=[heavy,light] or heavy_id/light_id.")
        for field_name in ("heavy_cdr_lengths", "light_cdr_lengths"):
            value = getattr(self, field_name)
            if value is not None and any(item < 1 for item in value):
                raise ValueError(f"{field_name} values must be >= 1.")
        return self


class LigandEntity(ChainEntitySpec):
    type: Literal["ligand"]
    smiles: str | None = None
    ccd: str | None = None

    @model_validator(mode="after")
    def _validate_ligand_input(self) -> "LigandEntity":
        if (self.smiles is None) == (self.ccd is None):
            raise ValueError("ligand requires exactly one of smiles or ccd.")
        if self.ids is not None and len(self.ids) != 1:
            raise ValueError("ligand ids must contain exactly one value.")
        return self


class FileEntity(StrictModel):
    type: Literal["file"]
    path: str
    include: Any | None = None
    exclude: Any | None = None
    include_proximity: Any | None = None
    binding_types: Any | None = None
    structure_groups: Any | None = None
    design: Any | None = None
    not_design: Any | None = None
    secondary_structure: Any | None = None
    design_insertions: Any | None = None
    fuse: Any | None = None
    msa: Any | None = None
    use_assembly: Any | None = None
    reset_res_index: Any | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


EntitySpec = Annotated[
    ProteinEntity
    | DNAEntity
    | RNAEntity
    | BinderEntity
    | PeptideEntity
    | AntibodyEntity
    | LigandEntity
    | FileEntity,
    Field(discriminator="type"),
]

ContextEntitySpec = Annotated[
    ProteinEntity
    | DNAEntity
    | RNAEntity
    | BinderEntity
    | PeptideEntity
    | LigandEntity
    | FileEntity,
    Field(discriminator="type"),
]


class BondConstraint(StrictModel):
    type: Literal["bond"]
    atom1: tuple[str, int, str]
    atom2: tuple[str, int, str]


class PocketConstraint(StrictModel):
    type: Literal["pocket"]
    binder: str
    contacts: list[tuple[str, int]]
    max_distance: float = 6.0
    force: bool = False

    @model_validator(mode="after")
    def _validate_contacts(self) -> "PocketConstraint":
        if not self.contacts:
            raise ValueError("pocket constraints require at least one contact.")
        return self


class ContactConstraint(StrictModel):
    type: Literal["contact"]
    token1: tuple[str, int]
    token2: tuple[str, int]
    max_distance: float = 6.0
    force: bool = False


ConstraintSpec = Annotated[
    BondConstraint | PocketConstraint | ContactConstraint,
    Field(discriminator="type"),
]


class AffinityOptions(StrictModel):
    binder: str | None = None


class BoltzOptions(StrictModel):
    cache_dir: str | None = None
    device: str | None = None
    auto_download: bool = True
    use_kernels: bool = True
    affinity_mw_correction: bool = True
    predict_args: dict[str, Any] | None = None
    affinity_predict_args: dict[str, Any] | None = None


class BoltzGenOptions(StrictModel):
    mol_dir: str | None = None
    auto_download: bool = True
    cache_dir: str | None = None
    token: str | None = None
    force_download: bool = False


class AdmetOptions(StrictModel):
    mode: Literal["auto", "on", "off"] | None = None
    enabled: bool | None = None
    ligands: str | list[str] | None = None
    model_variant: str | None = None
    max_new_tokens: int | None = None
    include_scoring: bool | None = None
    task_ids: list[str] | None = None


AdmetArg = bool | Literal["auto", "on", "off"] | AdmetOptions | None
AffinityArg = bool | AffinityOptions | None


class AffinityResult(StrictModel):
    ic50: float | None = None
    binding_probability: float | None = None
    ic50_1: float | None = None
    binding_probability_1: float | None = None
    ic50_2: float | None = None
    binding_probability_2: float | None = None


class StructureResult(StrictModel):
    confidence_score: float
    output_path: str | None = None
    output_format: Literal["cif", "bcif"] | None = None
    mmcif: str | None = None
    bcif_base64: str | None = None


class FeatureResult(StrictModel):
    feature_keys: list[str]
    feature_shapes: dict[str, list[int]]
    output_path: str | None = None
    output_format: Literal["torch", "npz"] | None = None


class FoldResult(StrictModel):
    name: str
    backend: str
    chain_ids: Any
    binder_sequences: Any
    ligand_id_map: dict[str, str] | None = None
    admet: dict[str, Any] | None = None
    affinity: AffinityResult | None = None
    structure: StructureResult | None = None
    features: FeatureResult | None = None


class AffinityResultResponse(StrictModel):
    name: str
    binder: str | None = None
    affinity: AffinityResult
    ligand_id_map: dict[str, str] | None = None
    admet: dict[str, Any] | None = None


class QueuedJobResponse(StrictModel):
    job_id: str
    status: Literal["queued"] = "queued"


class ValidationPlan(StrictModel):
    action: Literal["fold", "affinity"]
    run_boltz: bool
    run_boltzgen: bool
    entity_type_counts: dict[str, int]
    ligand_id_map: dict[str, str]
    smiles_ligand_ids: list[str]


class ValidateSpecResult(StrictModel):
    valid: Literal[True] = True
    normalized_input: dict[str, Any]
    execution_plan: ValidationPlan
    warnings: list[str] = Field(default_factory=list)


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
_TASK_SUPPORT_BY_TOOL: dict[str, TaskExecutionMode] = {
    "refua_fold": "optional",
    "refua_affinity": "optional",
    "refua_antibody_design": "optional",
    "refua_admet_profile": "optional",
}


def _clamp_seconds(value: float, minimum: int, maximum: int) -> int:
    return int(max(minimum, min(maximum, round(value))))


def _recommend_poll_seconds(estimate_seconds: float | None, queue_position: int) -> int:
    if estimate_seconds is None:
        return min(
            POLL_MAX_SECONDS,
            POLL_MIN_SECONDS + queue_position * POLL_QUEUE_STEP_SECONDS,
        )
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


def _task_support_mode(tool_name: str) -> TaskExecutionMode:
    if tool_name == "refua_admet_profile" and not _ADMET_AVAILABLE:
        return "forbidden"
    return _TASK_SUPPORT_BY_TOOL.get(tool_name, "forbidden")


def _normalize_task_tool_result(result: Any) -> CallToolResult:
    if isinstance(result, BaseModel):
        result = result.model_dump(mode="json")
    if isinstance(result, CallToolResult):
        return result
    if isinstance(result, dict):
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(result, indent=2),
                )
            ],
            structuredContent=result,
            isError=False,
        )
    if isinstance(result, tuple) and len(result) == 2:
        unstructured, structured = result
        if not isinstance(structured, dict):
            raise ValueError("Task tool tuple results require dict structured output.")
        return CallToolResult(
            content=list(unstructured),
            structuredContent=structured,
            isError=False,
        )
    if isinstance(result, str):
        return CallToolResult(
            content=[TextContent(type="text", text=result)],
            isError=False,
        )
    if isinstance(result, Iterable):
        return CallToolResult(content=list(result), isError=False)
    raise ValueError(f"Unexpected task tool return type: {type(result).__name__}")


def _coerce_tool_result_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    if isinstance(result, dict):
        return result
    raise ValueError("Task-augmented tool runners must return dict-like output.")


def _build_task_runner(
    tool_name: str, arguments: Mapping[str, Any]
) -> Callable[[], dict[str, Any]] | None:
    kwargs = dict(arguments)
    if tool_name in {"refua_fold", "refua_affinity", "refua_antibody_design"}:
        # task-augmented execution already runs in background; avoid nested async jobs.
        kwargs["async_mode"] = False
        if tool_name == "refua_fold":
            return lambda: _coerce_tool_result_dict(refua_fold(**kwargs))
        if tool_name == "refua_affinity":
            return lambda: _coerce_tool_result_dict(refua_affinity(**kwargs))
        return lambda: _coerce_tool_result_dict(refua_antibody_design(**kwargs))
    if tool_name == "refua_admet_profile" and _ADMET_AVAILABLE:
        return lambda: refua_admet_profile(**kwargs)
    return None


def _long_poll_sleep_seconds(
    snapshot: Mapping[str, Any], remaining_seconds: float
) -> float:
    suggested = float(snapshot.get("recommended_poll_seconds", POLL_MIN_SECONDS))
    bounded = max(
        LONG_POLL_MIN_SLEEP_SECONDS, min(LONG_POLL_MAX_SLEEP_SECONDS, suggested)
    )
    return min(remaining_seconds, bounded)


def _poll_job_until_terminal(
    job_id: str,
    *,
    include_result: bool,
    wait_for_terminal_seconds: float,
) -> dict[str, Any]:
    capped_wait = max(
        0.0, min(float(wait_for_terminal_seconds), LONG_POLL_MAX_WAIT_SECONDS)
    )
    deadline = time.time() + capped_wait
    snapshot = _job_snapshot(job_id, include_result)
    while snapshot["status"] in {"queued", "running"}:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(_long_poll_sleep_seconds(snapshot, remaining))
        snapshot = _job_snapshot(job_id, include_result)
    return snapshot


async def _call_tool_with_task_support(
    name: str,
    arguments: dict[str, Any],
) -> Any:
    context = mcp.get_context()
    request_context = context.request_context
    experimental = request_context.experimental
    task_mode = _task_support_mode(name)

    if experimental is not None:
        experimental.validate_task_mode(task_mode)

    # Non task-augmented calls behave exactly like standard FastMCP tool execution.
    if experimental is None or not experimental.is_task:
        return await mcp._tool_manager.call_tool(
            name,
            arguments,
            context=context,
            convert_result=False,
        )

    runner = _build_task_runner(name, arguments)
    if runner is None:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Task-augmented execution is not implemented for tool '{name}'.",
                )
            ],
            isError=True,
        )

    async def work(task_context: Any) -> CallToolResult:
        await task_context.update_status("queued")
        job_id = _submit_job(name, runner)

        while True:
            snapshot = _job_snapshot(job_id, include_result=True)
            status = str(snapshot["status"])

            if status == "success":
                return _normalize_task_tool_result(snapshot.get("result"))

            if status == "error":
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=str(snapshot.get("error", "Task failed.")),
                        )
                    ],
                    isError=True,
                )

            if status == "queued":
                queue_position = snapshot.get("queue_position")
                if queue_position is not None:
                    await task_context.update_status(f"queued ({queue_position} ahead)")
            elif status == "running":
                eta = snapshot.get("estimated_remaining_seconds")
                if isinstance(eta, (float, int)):
                    await task_context.update_status(
                        f"running (~{max(0, int(round(float(eta))))}s remaining)"
                    )
                else:
                    await task_context.update_status("running")

            await anyio.sleep(
                _long_poll_sleep_seconds(snapshot, LONG_POLL_MAX_SLEEP_SECONDS),
            )

    return await experimental.run_task(
        work,
        model_immediate_response=f"{name} started in background task execution.",
    )


async def _list_tools_with_task_support() -> list[McpTool]:
    tools = mcp._tool_manager.list_tools()
    return [
        McpTool(
            name=info.name,
            title=info.title,
            description=info.description,
            inputSchema=info.parameters,
            outputSchema=info.output_schema,
            annotations=info.annotations,
            icons=info.icons,
            _meta=info.meta,
            execution=ToolExecution(taskSupport=_task_support_mode(info.name)),
        )
        for info in tools
    ]


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


def _coerce_triplet(
    value: Any,
    *,
    field: str,
) -> tuple[int, int, int]:
    values: list[Any]
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise ValueError(
            f"{field} must be a 3-item list/tuple or comma-separated string."
        )

    if len(values) != 3:
        raise ValueError(f"{field} must contain exactly 3 values.")
    try:
        first = int(values[0])
        second = int(values[1])
        third = int(values[2])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} values must be integers.") from exc
    normalized = (first, second, third)
    if any(item < 1 for item in normalized):
        raise ValueError(f"{field} values must be >= 1.")
    return normalized


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


def _make_binder(
    *,
    spec: Any = None,
    length: int | None = None,
    ids: Any = None,
    binding_types: Any = None,
    secondary_structure: Any = None,
    cyclic: bool = False,
    template_values: Mapping[str, Any] | None = None,
) -> Any:
    from refua import Binder

    binder_kwargs: dict[str, Any] = {
        "spec": spec,
        "length": length,
        "ids": ids,
        "binding_types": binding_types,
        "secondary_structure": secondary_structure,
        "cyclic": cyclic,
    }
    if template_values is not None:
        binder_kwargs["template_values"] = template_values

    try:
        return Binder(**binder_kwargs)
    except TypeError as exc:
        if template_values is not None and "template_values" in str(exc):
            raise ValueError("Binder template_values requires refua>=0.5.0.") from exc
        raise


def _get_binder_designs() -> Any:
    import refua as refua_pkg

    binder_designs = getattr(refua_pkg, "BinderDesigns", None)
    if binder_designs is None:
        raise ValueError(
            "peptide and antibody entities require refua>=0.5.0 (BinderDesigns)."
        )
    return binder_designs


def _build_complex_from_spec(
    *,
    name: str,
    base_dir: str | None,
    entities: list[dict[str, Any]],
    boltz_mol_dir: Path | None,
) -> tuple[Complex, dict[str, str], bool, bool]:
    from refua import Complex, DNA, Protein, RNA

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
                _make_binder(
                    spec=spec,
                    length=length,
                    template_values=entity.get("template_values"),
                    ids=ids,
                    binding_types=entity.get("binding_types"),
                    secondary_structure=entity.get("secondary_structure"),
                    cyclic=bool(entity.get("cyclic", False)),
                )
            )
            has_boltzgen = True
            continue

        if entity_type == "peptide":
            ids = _resolve_entity_ids(entity)
            common_kwargs: dict[str, Any] = {
                "binding_types": entity.get("binding_types"),
                "secondary_structure": entity.get("secondary_structure"),
            }
            if ids is not None:
                common_kwargs["ids"] = ids

            spec = entity.get("spec")
            if spec is None:
                spec = entity.get("sequence")
            if spec is not None:
                length = entity.get("length")
                if length is not None:
                    length = int(length)
                cyclic = (
                    bool(entity.get("cyclic")) if "cyclic" in entity else bool(False)
                )
                peptide_binder = _make_binder(
                    spec=spec,
                    length=length,
                    template_values=entity.get("template_values"),
                    cyclic=cyclic,
                    **common_kwargs,
                )
            elif "segment_lengths" in entity or bool(entity.get("disulfide")):
                disulfide_kwargs = dict(common_kwargs)
                if "cyclic" in entity:
                    disulfide_kwargs["cyclic"] = bool(entity.get("cyclic"))
                if "segment_lengths" in entity:
                    disulfide_kwargs["segment_lengths"] = _coerce_triplet(
                        entity["segment_lengths"],
                        field="segment_lengths",
                    )
                peptide_binder = _get_binder_designs().disulfide_peptide(
                    **disulfide_kwargs
                )
            else:
                peptide_kwargs = dict(common_kwargs)
                if "cyclic" in entity:
                    peptide_kwargs["cyclic"] = bool(entity.get("cyclic"))
                peptide_kwargs["length"] = int(entity.get("length", 12))
                peptide_binder = _get_binder_designs().peptide(**peptide_kwargs)

            complex_spec.add(peptide_binder)
            has_boltzgen = True
            continue

        if entity_type == "antibody":
            ids = _resolve_entity_ids(entity)
            antibody_kwargs: dict[str, Any] = {}

            if ids is not None:
                if isinstance(ids, str):
                    raise ValueError(
                        "Antibody entity ids must include exactly two ids (heavy, light)."
                    )
                if len(ids) != 2:
                    raise ValueError(
                        "Antibody entity ids must include exactly two ids (heavy, light)."
                    )
                antibody_kwargs["heavy_id"] = ids[0]
                antibody_kwargs["light_id"] = ids[1]
            else:
                heavy_id = entity.get("heavy_id")
                light_id = entity.get("light_id")
                if heavy_id is not None:
                    antibody_kwargs["heavy_id"] = str(heavy_id)
                if light_id is not None:
                    antibody_kwargs["light_id"] = str(light_id)

            if "heavy_cdr_lengths" in entity:
                antibody_kwargs["heavy_cdr_lengths"] = _coerce_triplet(
                    entity["heavy_cdr_lengths"],
                    field="heavy_cdr_lengths",
                )
            if "light_cdr_lengths" in entity:
                antibody_kwargs["light_cdr_lengths"] = _coerce_triplet(
                    entity["light_cdr_lengths"],
                    field="light_cdr_lengths",
                )
            if "heavy_binding_types" in entity:
                antibody_kwargs["heavy_binding_types"] = entity.get(
                    "heavy_binding_types"
                )
            if "light_binding_types" in entity:
                antibody_kwargs["light_binding_types"] = entity.get(
                    "light_binding_types"
                )
            if "heavy_secondary_structure" in entity:
                antibody_kwargs["heavy_secondary_structure"] = entity.get(
                    "heavy_secondary_structure"
                )
            if "light_secondary_structure" in entity:
                antibody_kwargs["light_secondary_structure"] = entity.get(
                    "light_secondary_structure"
                )
            if "heavy_cyclic" in entity:
                antibody_kwargs["heavy_cyclic"] = bool(entity.get("heavy_cyclic"))
            if "light_cyclic" in entity:
                antibody_kwargs["light_cyclic"] = bool(entity.get("light_cyclic"))

            antibody_pair = _get_binder_designs().antibody(**antibody_kwargs)

            heavy_spec = entity.get("heavy_spec", entity.get("heavy_sequence"))
            light_spec = entity.get("light_spec", entity.get("light_sequence"))

            if heavy_spec is not None:
                heavy_binder = _make_binder(
                    spec=heavy_spec,
                    template_values=entity.get("heavy_template_values"),
                    ids=antibody_pair.heavy.ids,
                    binding_types=antibody_pair.heavy.binding_types,
                    secondary_structure=antibody_pair.heavy.secondary_structure,
                    cyclic=antibody_pair.heavy.cyclic,
                )
            else:
                heavy_binder = antibody_pair.heavy

            if light_spec is not None:
                light_binder = _make_binder(
                    spec=light_spec,
                    template_values=entity.get("light_template_values"),
                    ids=antibody_pair.light.ids,
                    binding_types=antibody_pair.light.binding_types,
                    secondary_structure=antibody_pair.light.secondary_structure,
                    cyclic=antibody_pair.light.cyclic,
                )
            else:
                light_binder = antibody_pair.light

            complex_spec.add(heavy_binder, light_binder)
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


def _model_dict(value: BaseModel | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    return dict(value)


def _entities_to_payload(entities: Iterable[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for entity in entities:
        if isinstance(entity, BaseModel):
            payload.append(entity.model_dump(exclude_none=True))
        elif isinstance(entity, Mapping):
            payload.append(dict(entity))
        else:
            raise ValueError("Each entity must be a dict or typed entity model.")
    return payload


def _constraints_to_payload(
    constraints: Iterable[Any] | None,
) -> list[dict[str, Any]] | None:
    if constraints is None:
        return None
    payload: list[dict[str, Any]] = []
    for constraint in constraints:
        if isinstance(constraint, BaseModel):
            payload.append(constraint.model_dump(exclude_none=True))
        elif isinstance(constraint, Mapping):
            payload.append(dict(constraint))
        else:
            raise ValueError(
                "Each constraint must be a dict or typed constraint model."
            )
    return payload


def _normalize_affinity_arg(affinity: AffinityArg) -> bool | dict[str, Any] | None:
    if isinstance(affinity, BaseModel):
        return affinity.model_dump(exclude_none=True)
    return affinity


def _normalize_admet_arg(admet: AdmetArg) -> bool | str | dict[str, Any] | None:
    if isinstance(admet, BaseModel):
        return admet.model_dump(exclude_none=True)
    return admet


def _compact_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _analyze_entities(
    entities: list[dict[str, Any]],
) -> tuple[list[str], dict[str, int], list[dict[str, Any]], dict[str, str]]:
    entity_types: list[str] = []
    entity_type_counts: dict[str, int] = {}
    ligand_specs: list[dict[str, Any]] = []
    ligand_alias_map: dict[str, str] = {}

    ligand_index = 1
    for item in entities:
        kind = str(item.get("type", "")).lower()
        entity_types.append(kind)
        entity_type_counts[kind] = entity_type_counts.get(kind, 0) + 1

        if kind != "ligand":
            continue

        ligand_id = f"L{ligand_index}"
        smiles = item.get("smiles")
        if smiles is not None:
            ligand_specs.append({"ligand_id": ligand_id, "smiles": str(smiles)})

        alias_value = item.get("id", item.get("ids"))
        if alias_value is not None:
            if isinstance(alias_value, (list, tuple)):
                if len(alias_value) != 1:
                    raise ValueError("Ligand id must be a single string.")
                alias = str(alias_value[0])
            else:
                alias = str(alias_value)
            if alias.startswith("L") and alias[1:].isdigit() and alias != ligand_id:
                raise ValueError(
                    "Ligand id aliases cannot shadow unified ids. "
                    "Omit the alias or use a non-L name."
                )
            if alias in ligand_alias_map:
                raise ValueError(f"Duplicate ligand alias: {alias}")
            ligand_alias_map[alias] = ligand_id

        ligand_index += 1

    return entity_types, entity_type_counts, ligand_specs, ligand_alias_map


def _resolve_execution_modes(
    *,
    action: str,
    entity_types: list[str],
    constraints: list[dict[str, Any]] | None,
    affinity: bool | dict[str, Any] | None,
    run_boltz: bool | None,
    run_boltzgen: bool | None,
) -> tuple[Literal["fold", "affinity"], bool, bool]:
    raw_action = str(action or "fold").lower()
    if raw_action not in {"fold", "affinity"}:
        raise ValueError("action must be 'fold' or 'affinity'.")
    action_value: Literal["fold", "affinity"] = (
        "affinity" if raw_action == "affinity" else "fold"
    )

    has_boltz_entities = any(
        kind in {"protein", "dna", "rna", "ligand"} for kind in entity_types
    )
    has_boltzgen_entities = any(
        kind in {"binder", "peptide", "antibody", "file"} for kind in entity_types
    )
    wants_affinity = affinity not in (None, False)

    run_boltz_local = (
        bool(run_boltz)
        if run_boltz is not None
        else bool(has_boltz_entities or constraints or wants_affinity)
    )
    run_boltzgen_local = (
        bool(run_boltzgen) if run_boltzgen is not None else bool(has_boltzgen_entities)
    )

    if action_value == "affinity":
        run_boltz_local = True
        run_boltzgen_local = False

    if constraints and not run_boltz_local:
        raise ValueError("constraints require run_boltz=true.")
    if wants_affinity and not run_boltz_local and action_value == "fold":
        raise ValueError("affinity requests require run_boltz=true.")

    return action_value, run_boltz_local, run_boltzgen_local


def _affinity_to_dict(value: Any) -> dict[str, float | None]:
    return {
        "ic50": value.ic50,
        "binding_probability": value.binding_probability,
        "ic50_1": value.ic50_1,
        "binding_probability_1": value.binding_probability_1,
        "ic50_2": value.ic50_2,
        "binding_probability_2": value.binding_probability_2,
    }


def _run_complex_operation(
    *,
    entities: list[dict[str, Any]],
    name: str,
    base_dir: str | None,
    constraints: list[dict[str, Any]] | None,
    affinity: bool | dict[str, Any] | None,
    action: Literal["fold", "affinity"],
    run_boltz: bool | None,
    run_boltzgen: bool | None,
    boltz: dict[str, Any] | None,
    boltzgen: dict[str, Any] | None,
    admet: bool | str | dict[str, Any] | None,
    structure_output_path: str | None,
    structure_output_format: str | None,
    return_mmcif: bool,
    return_bcif_base64: bool,
    feature_output_path: str | None,
    feature_output_format: str | None,
) -> dict[str, Any]:
    boltz_opts = _parse_boltz_options(boltz)
    boltzgen_opts = _parse_boltzgen_options(boltzgen)
    admet_mode, admet_opts = _parse_admet_options(admet)

    entity_types, _, ligand_specs, _ = _analyze_entities(entities)
    action_value, run_boltz_local, run_boltzgen_local = _resolve_execution_modes(
        action=action,
        entity_types=entity_types,
        constraints=constraints,
        affinity=affinity,
        run_boltz=run_boltz,
        run_boltzgen=run_boltzgen,
    )

    boltz_model = None
    if run_boltz_local or action_value == "affinity":
        boltz_model = _build_boltz2_from_options(boltz_opts)

    has_ccd = any(
        str(item.get("type", "")).lower() == "ligand" and item.get("ccd") is not None
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
            "affinity": _affinity_to_dict(affinity_result),
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
        output["affinity"] = _affinity_to_dict(result.affinity)

    if result.structure is None:
        if structure_output_path or return_mmcif or return_bcif_base64:
            raise ValueError(
                "Structure output requested but no structure was produced."
            )
    else:
        output_kind = _resolve_output_format(
            structure_output_path, structure_output_format
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
            structure_info["bcif_base64"] = base64.b64encode(bcif_bytes).decode("ascii")
        output["structure"] = structure_info

    features = result.features
    if features is None:
        if feature_output_path:
            raise ValueError("Feature output requested but no features were produced.")
    else:
        features = dict(features)
        feature_format = None
        output_written = None
        if feature_output_path:
            feature_format = _resolve_feature_output_format(
                feature_output_path,
                feature_output_format,
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


def _queue_job(tool_name: str, runner: Callable[[], BaseModel]) -> QueuedJobResponse:
    job_id = _submit_job(tool_name, lambda: runner().model_dump(mode="json"))
    return QueuedJobResponse(job_id=job_id)


@mcp.tool()
def refua_fold(
    entities: list[EntitySpec],
    *,
    name: str = "complex",
    base_dir: str | None = None,
    constraints: list[ConstraintSpec] | None = None,
    affinity: AffinityArg = None,
    run_boltz: bool | None = None,
    run_boltzgen: bool | None = None,
    boltz: BoltzOptions | None = None,
    boltzgen: BoltzGenOptions | None = None,
    admet: AdmetArg = None,
    structure_output_path: str | None = None,
    structure_output_format: Literal["cif", "bcif"] | None = None,
    return_mmcif: bool = False,
    return_bcif_base64: bool = False,
    feature_output_path: str | None = None,
    feature_output_format: Literal["torch", "npz"] | None = None,
    async_mode: bool = False,
) -> FoldResult | QueuedJobResponse:
    """Run Refua fold/design workflows with strict typed inputs."""

    entities_payload = _entities_to_payload(entities)
    constraints_payload = _constraints_to_payload(constraints)
    affinity_payload = _normalize_affinity_arg(affinity)
    boltz_payload = _model_dict(boltz)
    boltzgen_payload = _model_dict(boltzgen)
    admet_payload = _normalize_admet_arg(admet)

    def run() -> FoldResult:
        output = _run_complex_operation(
            entities=entities_payload,
            name=name,
            base_dir=base_dir,
            constraints=constraints_payload,
            affinity=affinity_payload,
            action="fold",
            run_boltz=run_boltz,
            run_boltzgen=run_boltzgen,
            boltz=boltz_payload,
            boltzgen=boltzgen_payload,
            admet=admet_payload,
            structure_output_path=structure_output_path,
            structure_output_format=structure_output_format,
            return_mmcif=return_mmcif,
            return_bcif_base64=return_bcif_base64,
            feature_output_path=feature_output_path,
            feature_output_format=feature_output_format,
        )
        return FoldResult.model_validate(output)

    if async_mode:
        return _queue_job("refua_fold", run)
    return run()


@mcp.tool()
def refua_affinity(
    entities: list[EntitySpec],
    *,
    name: str = "complex",
    base_dir: str | None = None,
    binder: str | None = None,
    boltz: BoltzOptions | None = None,
    admet: AdmetArg = None,
    async_mode: bool = False,
) -> AffinityResultResponse | QueuedJobResponse:
    """Run affinity-only predictions with strict typed inputs."""

    entities_payload = _entities_to_payload(entities)
    boltz_payload = _model_dict(boltz)
    admet_payload = _normalize_admet_arg(admet)
    affinity_payload: bool | dict[str, Any] = (
        {"binder": binder} if binder is not None else True
    )

    def run() -> AffinityResultResponse:
        output = _run_complex_operation(
            entities=entities_payload,
            name=name,
            base_dir=base_dir,
            constraints=None,
            affinity=affinity_payload,
            action="affinity",
            run_boltz=True,
            run_boltzgen=False,
            boltz=boltz_payload,
            boltzgen=None,
            admet=admet_payload,
            structure_output_path=None,
            structure_output_format=None,
            return_mmcif=False,
            return_bcif_base64=False,
            feature_output_path=None,
            feature_output_format=None,
        )
        return AffinityResultResponse.model_validate(output)

    if async_mode:
        return _queue_job("refua_affinity", run)
    return run()


@mcp.tool()
def refua_antibody_design(
    antibody: AntibodyEntity,
    *,
    context_entities: list[ContextEntitySpec] | None = None,
    name: str = "antibody_design",
    base_dir: str | None = None,
    constraints: list[ConstraintSpec] | None = None,
    affinity: AffinityArg = None,
    run_boltz: bool | None = None,
    run_boltzgen: bool | None = None,
    boltz: BoltzOptions | None = None,
    boltzgen: BoltzGenOptions | None = None,
    admet: AdmetArg = None,
    structure_output_path: str | None = None,
    structure_output_format: Literal["cif", "bcif"] | None = None,
    return_mmcif: bool = False,
    return_bcif_base64: bool = False,
    feature_output_path: str | None = None,
    feature_output_format: Literal["torch", "npz"] | None = None,
    async_mode: bool = False,
) -> FoldResult | QueuedJobResponse:
    """Design/fold with an explicit antibody entrypoint plus optional context entities."""

    merged_entities: list[EntitySpec] = [*(context_entities or []), antibody]
    entities_payload = _entities_to_payload(merged_entities)
    constraints_payload = _constraints_to_payload(constraints)
    affinity_payload = _normalize_affinity_arg(affinity)
    boltz_payload = _model_dict(boltz)
    boltzgen_payload = _model_dict(boltzgen)
    admet_payload = _normalize_admet_arg(admet)

    def run() -> FoldResult:
        output = _run_complex_operation(
            entities=entities_payload,
            name=name,
            base_dir=base_dir,
            constraints=constraints_payload,
            affinity=affinity_payload,
            action="fold",
            run_boltz=run_boltz,
            run_boltzgen=run_boltzgen,
            boltz=boltz_payload,
            boltzgen=boltzgen_payload,
            admet=admet_payload,
            structure_output_path=structure_output_path,
            structure_output_format=structure_output_format,
            return_mmcif=return_mmcif,
            return_bcif_base64=return_bcif_base64,
            feature_output_path=feature_output_path,
            feature_output_format=feature_output_format,
        )
        return FoldResult.model_validate(output)

    if async_mode:
        return _queue_job("refua_antibody_design", run)
    return run()


@mcp.tool()
def refua_validate_spec(
    entities: list[EntitySpec],
    *,
    action: Literal["fold", "affinity"] = "fold",
    name: str = "complex",
    base_dir: str | None = None,
    constraints: list[ConstraintSpec] | None = None,
    affinity: AffinityArg = None,
    run_boltz: bool | None = None,
    run_boltzgen: bool | None = None,
    boltz: BoltzOptions | None = None,
    boltzgen: BoltzGenOptions | None = None,
    admet: AdmetArg = None,
    structure_output_path: str | None = None,
    structure_output_format: Literal["cif", "bcif"] | None = None,
    feature_output_path: str | None = None,
    feature_output_format: Literal["torch", "npz"] | None = None,
    deep_validate: bool = False,
) -> ValidateSpecResult:
    """Validate and normalize a request without running fold/affinity inference."""

    entities_payload = _entities_to_payload(entities)
    constraints_payload = _constraints_to_payload(constraints)
    affinity_payload = _normalize_affinity_arg(affinity)
    boltz_payload = _model_dict(boltz)
    boltzgen_payload = _model_dict(boltzgen)
    admet_payload = _normalize_admet_arg(admet)

    boltz_opts = _parse_boltz_options(boltz_payload)
    _parse_boltzgen_options(boltzgen_payload)
    admet_mode, _ = _parse_admet_options(admet_payload)
    entity_types, entity_type_counts, ligand_specs, ligand_alias_map = (
        _analyze_entities(entities_payload)
    )
    action_value, run_boltz_local, run_boltzgen_local = _resolve_execution_modes(
        action=action,
        entity_types=entity_types,
        constraints=constraints_payload,
        affinity=affinity_payload,
        run_boltz=run_boltz,
        run_boltzgen=run_boltzgen,
    )

    if structure_output_path or structure_output_format:
        _resolve_output_format(structure_output_path, structure_output_format)
    if feature_output_path:
        _resolve_feature_output_format(feature_output_path, feature_output_format)

    warnings: list[str] = []
    if feature_output_format and not feature_output_path:
        warnings.append(
            "feature_output_format is ignored unless feature_output_path is provided."
        )
    if admet_mode == "auto" and not ligand_specs:
        warnings.append(
            "admet='auto' will be skipped because no SMILES ligands are present."
        )
    if admet_mode == "on" and not ligand_specs:
        raise ValueError("ADMET requested but no SMILES ligands are available.")

    if deep_validate:
        has_ccd = any(
            str(item.get("type", "")).lower() == "ligand"
            and item.get("ccd") is not None
            for item in entities_payload
        )
        should_deep_validate = True
        boltz_mol_dir = None
        if has_ccd:
            candidate_mol_dir = _resolve_boltz_mol_dir(None, boltz_opts)
            if candidate_mol_dir is None or not candidate_mol_dir.exists():
                should_deep_validate = False
                warnings.append(
                    "Skipped deep CCD ligand validation because Boltz molecule assets "
                    "are not available locally."
                )
            else:
                boltz_mol_dir = candidate_mol_dir

        if should_deep_validate:
            complex_spec, deep_alias_map, _, _ = _build_complex_from_spec(
                name=name,
                base_dir=base_dir,
                entities=entities_payload,
                boltz_mol_dir=boltz_mol_dir,
            )
            ligand_alias_map = deep_alias_map
            _apply_constraints(complex_spec, constraints_payload, ligand_alias_map)
    else:
        warnings.append(
            "Deep entity construction checks were skipped. Set deep_validate=true "
            "to validate against local Refua assets."
        )

    normalized_input = _compact_dict(
        {
            "action": action_value,
            "name": name,
            "base_dir": base_dir,
            "entities": entities_payload,
            "constraints": constraints_payload,
            "affinity": affinity_payload,
            "run_boltz": run_boltz,
            "run_boltzgen": run_boltzgen,
            "boltz": boltz_payload,
            "boltzgen": boltzgen_payload,
            "admet": admet_payload,
            "structure_output_path": structure_output_path,
            "structure_output_format": structure_output_format,
            "feature_output_path": feature_output_path,
            "feature_output_format": feature_output_format,
            "deep_validate": deep_validate,
        }
    )

    return ValidateSpecResult(
        normalized_input=normalized_input,
        execution_plan=ValidationPlan(
            action=action_value,
            run_boltz=run_boltz_local,
            run_boltzgen=run_boltzgen_local,
            entity_type_counts=entity_type_counts,
            ligand_id_map=ligand_alias_map,
            smiles_ligand_ids=[entry["ligand_id"] for entry in ligand_specs],
        ),
        warnings=warnings,
    )


_RECIPE_LIBRARY: dict[str, dict[str, Any]] = {
    "fold_protein_ligand": {
        "tool": "refua_fold",
        "args": {
            "name": "protein_ligand_fold",
            "entities": [
                {
                    "type": "protein",
                    "id": "A",
                    "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
                },
                {"type": "ligand", "id": "lig", "smiles": "CCO"},
            ],
            "constraints": [
                {"type": "pocket", "binder": "lig", "contacts": [["A", 5], ["A", 8]]}
            ],
            "affinity": {"binder": "lig"},
        },
    },
    "affinity_only": {
        "tool": "refua_affinity",
        "args": {
            "name": "protein_ligand_affinity",
            "entities": [
                {
                    "type": "protein",
                    "id": "A",
                    "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
                },
                {"type": "ligand", "id": "lig", "smiles": "CCO"},
            ],
            "binder": "lig",
        },
    },
    "antibody_design": {
        "tool": "refua_antibody_design",
        "args": {
            "name": "antibody_design",
            "antibody": {
                "type": "antibody",
                "ids": ["H", "L"],
                "heavy_cdr_lengths": [12, 10, 14],
                "light_cdr_lengths": [10, 9, 9],
            },
            "context_entities": [
                {
                    "type": "protein",
                    "id": "A",
                    "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
                }
            ],
        },
    },
}


@mcp.resource(
    "refua://recipes/index",
    name="refua_recipe_index",
    description="Index of canonical Refua MCP recipes.",
)
def refua_recipe_index() -> str:
    return json.dumps(
        {
            "template_uri": "refua://recipes/{recipe_name}",
            "recipe_names": sorted(_RECIPE_LIBRARY.keys()),
            "note": "Fetch refua://recipes/{recipe_name} for concrete tool args.",
        },
        indent=2,
    )


@mcp.resource(
    "refua://recipes/{recipe_name}",
    name="refua_recipe_template",
    description="Canonical Refua MCP recipe by name.",
)
def refua_recipe_template(recipe_name: str) -> str:
    key = str(recipe_name).strip().lower().replace("-", "_")
    recipe = _RECIPE_LIBRARY.get(key)
    if recipe is None:
        raise ValueError(
            f"Unknown recipe '{recipe_name}'. Available: {sorted(_RECIPE_LIBRARY)}"
        )
    return json.dumps(recipe, indent=2)


@mcp.tool()
def refua_job(
    job_id: str,
    *,
    include_result: bool = False,
    wait_for_terminal_seconds: float | None = None,
) -> dict[str, Any]:
    """Check status for a background refua job.

    Responses may include recommended_poll_seconds plus queue/estimate metadata for
    queued or running jobs.

    wait_for_terminal_seconds optionally blocks until the job reaches a terminal state
    (success/error) or the timeout is reached. Use this to reduce client-side polling.
    """
    if wait_for_terminal_seconds is None:
        return _job_snapshot(job_id, include_result)
    wait_seconds = float(wait_for_terminal_seconds)
    if wait_seconds <= 0:
        return _job_snapshot(job_id, include_result)
    return _poll_job_until_terminal(
        job_id,
        include_result=include_result,
        wait_for_terminal_seconds=wait_seconds,
    )


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
        Supports experimental task-augmented execution via MCP tasks.
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


def _configure_experimental_task_support() -> None:
    # FastMCP does not yet expose taskSupport metadata directly. We register the
    # low-level handlers to advertise execution.taskSupport and support
    # task-augmented calls for long-running tools.
    lowlevel = mcp._mcp_server
    lowlevel.experimental.enable_tasks()
    original_update_capabilities = lowlevel.experimental.update_capabilities

    def update_capabilities_with_tool_task_call(capabilities: Any) -> None:
        original_update_capabilities(capabilities)
        if (
            capabilities.tasks is not None
            and capabilities.tasks.requests is not None
            and capabilities.tasks.requests.tools is not None
        ):
            capabilities.tasks.requests.tools.call = TasksCallCapability()

    lowlevel.experimental.update_capabilities = update_capabilities_with_tool_task_call
    lowlevel.list_tools()(_list_tools_with_task_support)
    lowlevel.call_tool(validate_input=False)(_call_tool_with_task_support)


_configure_experimental_task_support()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
