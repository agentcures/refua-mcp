from __future__ import annotations

import base64
import importlib.util
import json
import logging
import os
import re
import statistics
import threading
import time
import uuid
from contextlib import contextmanager
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Callable, Iterable, Literal, Mapping

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import (
    LATEST_PROTOCOL_VERSION,
    CallToolResult,
    Completion,
    PromptReference,
    ResourceTemplateReference,
    TextContent,
)
from mcp.types import TaskExecutionMode
from mcp.types import TasksCallCapability
from mcp.types import Tool as McpTool
from mcp.types import ToolExecution
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

if TYPE_CHECKING:
    from refua import Boltz2, BoltzGen, Complex, SmallMolecule
    from refua.admet import AdmetPredictor  # type: ignore[reportMissingImports]
else:
    Boltz2 = BoltzGen = Complex = SmallMolecule = Any
    AdmetPredictor = Any

LOGGER = logging.getLogger(__name__)
MCP_SPEC_REVISION = "2025-11-25"
if str(LATEST_PROTOCOL_VERSION) != MCP_SPEC_REVISION:
    LOGGER.warning(
        "MCP SDK latest protocol version (%s) differs from server target revision (%s).",
        str(LATEST_PROTOCOL_VERSION),
        MCP_SPEC_REVISION,
    )

SERVER_INSTRUCTIONS = """
Use the typed Refua tools directly instead of speculative probing.

Recommended sequence:
1) Read capability/resource guidance (`refua://capabilities`,
   `refua://recipes/index`, and `refua://recipes/{recipe_name}`).
2) For data-driven workflows, use `refua_data_list` to discover datasets and
   `refua_data_materialize` / `refua_data_query` for local dataset access.
   For preclinical workflows, use `refua_preclinical_templates` to bootstrap
   specs, then `refua_preclinical_plan` / `refua_preclinical_schedule` /
   `refua_preclinical_bioanalysis` / `refua_preclinical_workup`.
   For CMC operations, use `refua_preclinical_cmc_templates`,
   `refua_preclinical_cmc_plan`, `refua_preclinical_batch_record`,
   `refua_preclinical_stability_plan`, `refua_preclinical_stability_assess`,
   and `refua_preclinical_release_assess`.
3) For sequence-only analysis, use `refua_protein_properties`.
   Use `properties` for explicit property names, or `groups` for grouped summaries.
4) Call `refua_validate_spec` to normalize/validate before expensive work.
   Do not execute schema probes, sanity folds, or smoke-test designs.
5) Execute with the focused tool:
   - `refua_fold` for structure/design folds
   - `refua_affinity` for affinity-only predictions
   - `refua_antibody_design` for antibody-heavy workflows
   - `refua_clinical_simulator` for clinical trial simulation (optional extra)
6) For long runs, set `async_mode=true` and poll `refua_job` using
   `recommended_poll_seconds` or `wait_for_terminal_seconds`.
"""

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
DEFAULT_TASK_TIMEOUT_SECONDS = 7200.0
DEFAULT_QUEUE_TIMEOUT_SECONDS = 1800.0


def _parse_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


def _parse_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0.")
    return parsed


def _parse_env_csv(name: str) -> tuple[str, ...]:
    value = os.environ.get(name, "")
    if not value.strip():
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _runtime_mount_path() -> str:
    mount_path = os.environ.get("REFUA_MCP_MOUNT_PATH", "/").strip() or "/"
    if not mount_path.startswith("/"):
        mount_path = f"/{mount_path}"
    return mount_path


def _default_allowed_hosts(host: str, port: int) -> tuple[str, ...]:
    defaults = {
        host,
        f"{host}:{port}",
    }
    if host in {"127.0.0.1", "localhost"}:
        defaults.update(
            {
                "localhost",
                f"localhost:{port}",
                "127.0.0.1",
                f"127.0.0.1:{port}",
            }
        )
    return tuple(sorted(defaults))


def _default_allowed_origins(host: str, port: int) -> tuple[str, ...]:
    defaults = {
        f"http://{host}:{port}",
    }
    if host in {"127.0.0.1", "localhost"}:
        defaults.update(
            {
                f"http://localhost:{port}",
                f"http://127.0.0.1:{port}",
            }
        )
    return tuple(sorted(defaults))


class _StaticTokenVerifier:
    def __init__(self, tokens: tuple[str, ...]):
        self._tokens = frozenset(tokens)

    async def verify_token(self, token: str) -> Any:
        if token in self._tokens:
            from mcp.server.auth.provider import AccessToken

            return AccessToken(
                token=token,
                client_id="refua-mcp-static-client",
                scopes=[],
                expires_at=None,
                resource=None,
            )
        return None


@dataclass(frozen=True, slots=True)
class RuntimeServerConfig:
    transport: Literal["stdio", "sse", "streamable-http"]
    host: str
    port: int
    mount_path: str
    task_timeout_seconds: float
    queue_timeout_seconds: float
    enable_dns_rebinding_protection: bool
    allowed_hosts: tuple[str, ...] = field(default_factory=tuple)
    allowed_origins: tuple[str, ...] = field(default_factory=tuple)
    token_count: int = 0
    transport_security: TransportSecuritySettings | None = None
    token_verifier: Any | None = None


def _build_runtime_server_config() -> RuntimeServerConfig:
    transport_raw = os.environ.get("REFUA_MCP_TRANSPORT", "stdio").strip().lower()
    if transport_raw not in {"stdio", "sse", "streamable-http"}:
        raise ValueError(
            "REFUA_MCP_TRANSPORT must be one of: stdio, sse, streamable-http."
        )
    transport: Literal["stdio", "sse", "streamable-http"] = transport_raw  # type: ignore[assignment]

    host = os.environ.get("REFUA_MCP_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("REFUA_MCP_PORT", "8000"))
    if port <= 0:
        raise ValueError("REFUA_MCP_PORT must be > 0.")
    mount_path = _runtime_mount_path()

    task_timeout_seconds = _parse_env_float(
        "REFUA_MCP_TASK_TIMEOUT_SECONDS",
        DEFAULT_TASK_TIMEOUT_SECONDS,
    )
    queue_timeout_seconds = _parse_env_float(
        "REFUA_MCP_QUEUE_TIMEOUT_SECONDS",
        DEFAULT_QUEUE_TIMEOUT_SECONDS,
    )

    enable_dns_rebinding_protection = _parse_env_bool(
        "REFUA_MCP_ENABLE_DNS_REBINDING_PROTECTION",
        transport in {"sse", "streamable-http"},
    )
    allowed_hosts = _parse_env_csv("REFUA_MCP_ALLOWED_HOSTS")
    allowed_origins = _parse_env_csv("REFUA_MCP_ALLOWED_ORIGINS")
    if transport in {"sse", "streamable-http"} and enable_dns_rebinding_protection:
        if not allowed_hosts:
            allowed_hosts = _default_allowed_hosts(host, port)
        if not allowed_origins:
            allowed_origins = _default_allowed_origins(host, port)

    transport_security = None
    if transport in {"sse", "streamable-http"}:
        transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=enable_dns_rebinding_protection,
            allowed_hosts=list(allowed_hosts),
            allowed_origins=list(allowed_origins),
        )

    auth_tokens = _parse_env_csv("REFUA_MCP_AUTH_TOKENS")
    token_verifier = _StaticTokenVerifier(auth_tokens) if auth_tokens else None

    return RuntimeServerConfig(
        transport=transport,
        host=host,
        port=port,
        mount_path=mount_path,
        task_timeout_seconds=task_timeout_seconds,
        queue_timeout_seconds=queue_timeout_seconds,
        enable_dns_rebinding_protection=enable_dns_rebinding_protection,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
        token_count=len(auth_tokens),
        transport_security=transport_security,
        token_verifier=token_verifier,
    )


_RUNTIME_CONFIG = _build_runtime_server_config()
mcp = FastMCP(
    "refua-mcp",
    instructions=SERVER_INSTRUCTIONS,
    host=_RUNTIME_CONFIG.host,
    port=_RUNTIME_CONFIG.port,
    mount_path=_RUNTIME_CONFIG.mount_path,
    token_verifier=_RUNTIME_CONFIG.token_verifier,
    transport_security=_RUNTIME_CONFIG.transport_security,
)

try:  # noqa: SIM105
    _MCP_SDK_VERSION = package_version("mcp")
except PackageNotFoundError:
    _MCP_SDK_VERSION = None

try:  # noqa: SIM105
    _REFUA_VERSION = package_version("refua")
except PackageNotFoundError:
    _REFUA_VERSION = None

try:  # noqa: SIM105
    _REFUA_CLINICAL_VERSION = package_version("refua-clinical")
except PackageNotFoundError:
    _REFUA_CLINICAL_VERSION = None

try:  # noqa: SIM105
    _REFUA_DATA_VERSION = package_version("refua-data")
except PackageNotFoundError:
    _REFUA_DATA_VERSION = None

try:  # noqa: SIM105
    _REFUA_PRECLINICAL_VERSION = package_version("refua-preclinical")
except PackageNotFoundError:
    _REFUA_PRECLINICAL_VERSION = None

try:  # Optional observability dependency.
    from opentelemetry import metrics as otel_metrics  # type: ignore[reportMissingImports]
    from opentelemetry import trace as otel_trace  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover - environment dependent optional import.
    otel_metrics = None
    otel_trace = None

_OTEL_AVAILABLE = otel_trace is not None and otel_metrics is not None
_OTEL_TRACER = (
    otel_trace.get_tracer("refua-mcp", _MCP_SDK_VERSION or "unknown")
    if _OTEL_AVAILABLE
    else None
)
_OTEL_METER = (
    otel_metrics.get_meter("refua-mcp", _MCP_SDK_VERSION or "unknown")
    if _OTEL_AVAILABLE
    else None
)
_JOB_SUBMITTED_COUNTER = (
    _OTEL_METER.create_counter(
        "refua_mcp.jobs.submitted",
        unit="1",
        description="Count of background jobs submitted.",
    )
    if _OTEL_METER is not None
    else None
)
_JOB_COMPLETED_COUNTER = (
    _OTEL_METER.create_counter(
        "refua_mcp.jobs.completed",
        unit="1",
        description="Count of successfully completed background jobs.",
    )
    if _OTEL_METER is not None
    else None
)
_JOB_FAILED_COUNTER = (
    _OTEL_METER.create_counter(
        "refua_mcp.jobs.failed",
        unit="1",
        description="Count of failed background jobs.",
    )
    if _OTEL_METER is not None
    else None
)
_JOB_CANCELLED_COUNTER = (
    _OTEL_METER.create_counter(
        "refua_mcp.jobs.cancelled",
        unit="1",
        description="Count of cancelled background jobs.",
    )
    if _OTEL_METER is not None
    else None
)
_JOB_RUNTIME_HISTOGRAM = (
    _OTEL_METER.create_histogram(
        "refua_mcp.jobs.runtime_seconds",
        unit="s",
        description="Background job runtime in seconds.",
    )
    if _OTEL_METER is not None
    else None
)
_JOB_QUEUE_DEPTH_HISTOGRAM = (
    _OTEL_METER.create_histogram(
        "refua_mcp.jobs.queue_depth",
        unit="1",
        description="Queue depth observed when jobs are submitted or polled.",
    )
    if _OTEL_METER is not None
    else None
)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _admet_available() -> bool:
    return all(_module_available(dep) for dep in ADMET_DEPENDENCIES)


def _clinical_available() -> bool:
    if not _module_available("refua_clinical"):
        return False
    try:
        from refua_clinical import ClinicalStudy  # type: ignore[import-not-found]
    except Exception:
        return False
    return hasattr(ClinicalStudy, "simulate")


def _data_available() -> bool:
    if not _module_available("refua_data"):
        return False
    try:
        from refua_data import DatasetManager  # type: ignore[import-not-found]
    except Exception:
        return False
    return hasattr(DatasetManager, "materialize")


def _preclinical_available() -> bool:
    if not _module_available("refua_preclinical"):
        return False
    try:
        import refua_preclinical as preclinical  # type: ignore[import-not-found]
    except Exception:
        return False
    required_exports = (
        "default_study_spec",
        "study_spec_from_mapping",
        "build_study_plan",
        "build_in_vivo_schedule",
        "run_bioanalytical_pipeline",
        "build_workup",
        "default_templates",
        "latest_preclinical_references",
        "default_cmc_templates",
        "build_formulation_process_plan",
        "generate_batch_record",
        "build_stability_study_plan",
        "assess_stability_results",
        "evaluate_release_criteria",
        "latest_cmc_references",
    )
    return all(hasattr(preclinical, name) for name in required_exports)


def _get_clinical_study_cls() -> Any:
    from refua_clinical import ClinicalStudy  # type: ignore[import-not-found]

    return ClinicalStudy


def _get_preclinical_module() -> Any:
    import refua_preclinical  # type: ignore[import-not-found]

    return refua_preclinical


_ADMET_AVAILABLE = _admet_available()
_CLINICAL_AVAILABLE = _clinical_available()
_DATA_AVAILABLE = _data_available()
_PRECLINICAL_AVAILABLE = _preclinical_available()
_PROBE_RUN_NAME_RE = re.compile(
    r"(sanity|probe|schema|smoke|dry[_\-\s]?run)",
    re.IGNORECASE,
)


def _normalize_cache_root(cache_root: str | None) -> str | None:
    if cache_root is None:
        return None
    text = str(cache_root).strip()
    if not text:
        return None
    return str(Path(text).expanduser().resolve())


@lru_cache(maxsize=8)
def _get_refua_data_manager(cache_root: str | None) -> Any:
    from refua_data import DataCache, DatasetManager  # type: ignore[import-not-found]

    if cache_root is None:
        return DatasetManager()
    return DatasetManager(cache=DataCache(Path(cache_root)))


def _normalize_string_column_list(value: Any, *, field_name: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be an array of strings when provided.")
    columns: list[str] = []
    seen: set[str] = set()
    for raw in value:
        text = str(raw).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        columns.append(text)
    if not columns:
        raise ValueError(f"{field_name} must contain at least one non-empty string.")
    return columns


def _normalize_data_query_filters(filters: Mapping[str, Any] | None) -> dict[str, Any]:
    if filters is None:
        return {}
    if not isinstance(filters, Mapping):
        raise ValueError("filters must be an object when provided.")
    return {str(key): value for key, value in filters.items()}


def _apply_data_query_filters(frame: Any, filters: Mapping[str, Any]) -> Any:
    if not filters:
        return frame

    filtered = frame
    for column, condition in filters.items():
        if column not in filtered.columns:
            raise ValueError(f"Unknown filter column '{column}'.")
        series = filtered[column]

        if isinstance(condition, Mapping):
            for op, raw_value in condition.items():
                op_name = str(op).strip().lower()
                if op_name == "eq":
                    filtered = filtered[series == raw_value]
                elif op_name == "ne":
                    filtered = filtered[series != raw_value]
                elif op_name == "gt":
                    filtered = filtered[series > raw_value]
                elif op_name in {"gte", "ge"}:
                    filtered = filtered[series >= raw_value]
                elif op_name == "lt":
                    filtered = filtered[series < raw_value]
                elif op_name in {"lte", "le"}:
                    filtered = filtered[series <= raw_value]
                elif op_name == "in":
                    if not isinstance(raw_value, (list, tuple, set)):
                        raise ValueError(f"filters.{column}.in must be an array value.")
                    filtered = filtered[series.isin(list(raw_value))]
                elif op_name == "contains":
                    pattern = str(raw_value)
                    filtered = filtered[
                        series.astype(str).str.contains(
                            pattern,
                            case=False,
                            na=False,
                            regex=False,
                        )
                    ]
                else:
                    raise ValueError(
                        f"Unsupported filter operation '{op_name}' for column '{column}'."
                    )
                series = filtered[column]
            continue

        if isinstance(condition, (list, tuple, set)):
            filtered = filtered[series.isin(list(condition))]
        else:
            filtered = filtered[series == condition]

    return filtered


@contextmanager
def _trace_span(name: str, **attributes: Any) -> Any:
    if _OTEL_TRACER is None:
        yield None
        return
    with _OTEL_TRACER.start_as_current_span(name) as span:
        for key, value in attributes.items():
            if value is None:
                continue
            try:
                span.set_attribute(key, value)
            except Exception:
                continue
        yield span


def _metric_add(
    counter: Any, value: int, *, attributes: Mapping[str, Any] | None = None
) -> None:
    if counter is None:
        return
    try:
        counter.add(value, attributes=dict(attributes or {}))
    except Exception:
        return


def _metric_record(
    histogram: Any, value: float, *, attributes: Mapping[str, Any] | None = None
) -> None:
    if histogram is None:
        return
    try:
        histogram.record(value, attributes=dict(attributes or {}))
    except Exception:
        return


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorContract(StrictModel):
    code: str
    message: str
    hint: str | None = None
    retryable: bool = False
    details: dict[str, Any] | None = None


def _error_contract(
    *,
    code: str,
    message: str,
    hint: str | None = None,
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = ErrorContract(
        code=code,
        message=message,
        hint=hint,
        retryable=retryable,
        details=dict(details) if details is not None else None,
    )
    return payload.model_dump(exclude_none=True)


def _error_contract_from_exception(exc: Exception) -> dict[str, Any]:
    message = str(exc).strip() or type(exc).__name__
    code = "internal_error"
    hint: str | None = "Retry or inspect server logs for details."
    retryable = False
    exception_name = type(exc).__name__

    if isinstance(exc, ValueError):
        code = "invalid_input"
        hint = "Check tool arguments against the published schema."
    elif exception_name in {"ValidationError"}:
        code = "invalid_input"
        hint = "Check tool arguments against the published schema."
    elif exception_name in {"ToolError"}:
        if any(
            token in message.lower()
            for token in (" required", "invalid", "unknown", "must be", "cannot")
        ):
            code = "invalid_input"
            hint = "Check tool arguments against the published schema."
        else:
            code = "tool_execution_error"
            hint = "Inspect tool arguments and server logs."
    elif isinstance(exc, FileNotFoundError):
        code = "asset_not_found"
        hint = "Confirm that required files/assets are present and readable."
    elif isinstance(exc, ModuleNotFoundError):
        code = "dependency_missing"
        hint = "Install the missing dependency and restart the MCP server."
    elif isinstance(exc, PermissionError):
        code = "permission_denied"
        hint = "Grant the process access to required files/directories."
    elif isinstance(exc, TimeoutError):
        code = "timeout"
        hint = "Increase timeout values or reduce workload size."
        retryable = True
    elif isinstance(exc, KeyError):
        code = "not_found"
        hint = "Verify referenced ids/names exist."
    elif isinstance(exc, RuntimeError):
        code = "runtime_error"
        hint = "Inspect runtime configuration and required model assets."
        retryable = True

    lower_message = message.lower()
    if "refua[admet]" in lower_message:
        code = "dependency_missing"
        hint = "Install refua[admet] to enable ADMET predictions."
    elif "task-augmented execution is not implemented" in lower_message:
        code = "task_mode_unsupported"
        hint = "Invoke the tool without task augmentation or choose a supported tool."
    elif "unknown recipe" in lower_message:
        code = "unknown_recipe"
        hint = "Read refua://recipes/index to list valid recipe names."

    return _error_contract(
        code=code,
        message=message,
        hint=hint,
        retryable=retryable,
        details={"exception_type": exception_name},
    )


def _error_call_tool_result(error: Mapping[str, Any]) -> CallToolResult:
    payload = {"error": dict(error)}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, indent=2))],
        structuredContent=payload,
        isError=True,
    )


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
StructureOutputFormatArg = Literal["cif", "mmcif", "bcif"] | None
FeatureOutputFormatArg = Literal["torch", "npz", "json"] | None

_ANTIBODY_ENTITY_ADAPTER = TypeAdapter(AntibodyEntity)
_ENTITY_LIST_ADAPTER = TypeAdapter(list[EntitySpec])
_CONTEXT_ENTITY_LIST_ADAPTER = TypeAdapter(list[ContextEntitySpec])
_CONSTRAINT_LIST_ADAPTER = TypeAdapter(list[ConstraintSpec])


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
    warnings: list[str] = Field(default_factory=list)


class AffinityResultResponse(StrictModel):
    name: str
    binder: str | None = None
    affinity: AffinityResult
    ligand_id_map: dict[str, str] | None = None
    admet: dict[str, Any] | None = None


class ProteinPropertiesResult(StrictModel):
    sequence: str
    normalized_sequence: str
    values: dict[str, Any]
    selected_properties: list[str] | None = None
    selected_groups: list[str] | None = None
    available_properties: list[str] | None = None
    available_property_groups: list[str] | None = None


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
    error: dict[str, Any] | None = None
    queue_timeout_seconds: float | None = None
    cancel_requested: bool = False


_JOB_LOCK = threading.Lock()
_JOB_STORE: OrderedDict[str, JobRecord] = OrderedDict()
_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=JOB_MAX_WORKERS)
_TASK_JOB_MAP: dict[str, str] = {}
_TASK_SUPPORT_BY_TOOL: dict[str, TaskExecutionMode] = {
    "refua_validate_spec": "optional",
    "refua_fold": "optional",
    "refua_affinity": "optional",
    "refua_antibody_design": "optional",
    "refua_protein_properties": "optional",
    "refua_job": "optional",
    "refua_admet_profile": "optional",
}
if _CLINICAL_AVAILABLE:
    _TASK_SUPPORT_BY_TOOL["refua_clinical_simulator"] = "optional"
if _DATA_AVAILABLE:
    _TASK_SUPPORT_BY_TOOL["refua_data_list"] = "optional"
    _TASK_SUPPORT_BY_TOOL["refua_data_fetch"] = "optional"
    _TASK_SUPPORT_BY_TOOL["refua_data_materialize"] = "optional"
    _TASK_SUPPORT_BY_TOOL["refua_data_query"] = "optional"
if _PRECLINICAL_AVAILABLE:
    _TASK_SUPPORT_BY_TOOL["refua_preclinical_templates"] = "optional"
    _TASK_SUPPORT_BY_TOOL["refua_preclinical_plan"] = "optional"
    _TASK_SUPPORT_BY_TOOL["refua_preclinical_schedule"] = "optional"
    _TASK_SUPPORT_BY_TOOL["refua_preclinical_bioanalysis"] = "optional"
    _TASK_SUPPORT_BY_TOOL["refua_preclinical_workup"] = "optional"
    _TASK_SUPPORT_BY_TOOL["refua_preclinical_cmc_templates"] = "optional"
    _TASK_SUPPORT_BY_TOOL["refua_preclinical_cmc_plan"] = "optional"
    _TASK_SUPPORT_BY_TOOL["refua_preclinical_batch_record"] = "optional"
    _TASK_SUPPORT_BY_TOOL["refua_preclinical_stability_plan"] = "optional"
    _TASK_SUPPORT_BY_TOOL["refua_preclinical_stability_assess"] = "optional"
    _TASK_SUPPORT_BY_TOOL["refua_preclinical_release_assess"] = "optional"


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


def _register_task_job(task_id: str, job_id: str) -> None:
    with _JOB_LOCK:
        _TASK_JOB_MAP[str(task_id)] = str(job_id)


def _unregister_task_job(task_id: str, job_id: str | None = None) -> None:
    key = str(task_id)
    with _JOB_LOCK:
        current = _TASK_JOB_MAP.get(key)
        if current is None:
            return
        if job_id is not None and current != str(job_id):
            return
        _TASK_JOB_MAP.pop(key, None)


def _lookup_task_job(task_id: str) -> str | None:
    with _JOB_LOCK:
        return _TASK_JOB_MAP.get(str(task_id))


def _queue_timeout_seconds(job: JobRecord) -> float:
    if job.queue_timeout_seconds is not None:
        return max(0.0, float(job.queue_timeout_seconds))
    return max(0.0, float(_RUNTIME_CONFIG.queue_timeout_seconds))


def _cancel_job(
    job_id: str,
    *,
    reason: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot_only = False
    with _JOB_LOCK:
        job = _JOB_STORE.get(job_id)
        if job is None:
            raise ValueError(f"Unknown job id: {job_id}")
        if job.status in {"success", "error", "cancelled"}:
            snapshot_only = True
        else:
            cancellation_error = dict(
                reason
                or _error_contract(
                    code="job_cancelled",
                    message="Job was cancelled.",
                    hint="Resubmit the job if you still need the result.",
                    retryable=True,
                )
            )
            job.cancel_requested = True
            job.status = "cancelled"
            job.error = cancellation_error
            if job.finished_at is None:
                job.finished_at = time.time()

            _metric_add(
                _JOB_CANCELLED_COUNTER,
                1,
                attributes={"tool": job.tool},
            )
    if snapshot_only:
        return _job_snapshot(job_id, include_result=False)

    return _job_snapshot(job_id, include_result=False)


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
    if tool_name == "refua_validate_spec":
        return lambda: _coerce_tool_result_dict(refua_validate_spec(**kwargs))
    if tool_name in {"refua_fold", "refua_affinity", "refua_antibody_design"}:
        # task-augmented execution already runs in background; avoid nested async jobs.
        kwargs["async_mode"] = False
        if tool_name == "refua_fold":
            return lambda: _coerce_tool_result_dict(refua_fold(**kwargs))
        if tool_name == "refua_affinity":
            return lambda: _coerce_tool_result_dict(refua_affinity(**kwargs))
        return lambda: _coerce_tool_result_dict(refua_antibody_design(**kwargs))
    if tool_name == "refua_protein_properties":
        return lambda: _coerce_tool_result_dict(refua_protein_properties(**kwargs))
    if tool_name == "refua_job":
        return lambda: _coerce_tool_result_dict(refua_job(**kwargs))
    if tool_name == "refua_admet_profile" and _ADMET_AVAILABLE:
        return lambda: refua_admet_profile(**kwargs)
    if tool_name == "refua_clinical_simulator" and _CLINICAL_AVAILABLE:
        return lambda: _coerce_tool_result_dict(refua_clinical_simulator(**kwargs))
    if tool_name == "refua_data_list" and _DATA_AVAILABLE:
        return lambda: _coerce_tool_result_dict(refua_data_list(**kwargs))
    if tool_name == "refua_data_fetch" and _DATA_AVAILABLE:
        return lambda: _coerce_tool_result_dict(refua_data_fetch(**kwargs))
    if tool_name == "refua_data_materialize" and _DATA_AVAILABLE:
        return lambda: _coerce_tool_result_dict(refua_data_materialize(**kwargs))
    if tool_name == "refua_data_query" and _DATA_AVAILABLE:
        return lambda: _coerce_tool_result_dict(refua_data_query(**kwargs))
    if tool_name == "refua_preclinical_templates" and _PRECLINICAL_AVAILABLE:
        return lambda: _coerce_tool_result_dict(refua_preclinical_templates(**kwargs))
    if tool_name == "refua_preclinical_plan" and _PRECLINICAL_AVAILABLE:
        return lambda: _coerce_tool_result_dict(refua_preclinical_plan(**kwargs))
    if tool_name == "refua_preclinical_schedule" and _PRECLINICAL_AVAILABLE:
        return lambda: _coerce_tool_result_dict(refua_preclinical_schedule(**kwargs))
    if tool_name == "refua_preclinical_bioanalysis" and _PRECLINICAL_AVAILABLE:
        return lambda: _coerce_tool_result_dict(refua_preclinical_bioanalysis(**kwargs))
    if tool_name == "refua_preclinical_workup" and _PRECLINICAL_AVAILABLE:
        return lambda: _coerce_tool_result_dict(refua_preclinical_workup(**kwargs))
    if tool_name == "refua_preclinical_cmc_templates" and _PRECLINICAL_AVAILABLE:
        return lambda: _coerce_tool_result_dict(
            refua_preclinical_cmc_templates(**kwargs)
        )
    if tool_name == "refua_preclinical_cmc_plan" and _PRECLINICAL_AVAILABLE:
        return lambda: _coerce_tool_result_dict(refua_preclinical_cmc_plan(**kwargs))
    if tool_name == "refua_preclinical_batch_record" and _PRECLINICAL_AVAILABLE:
        return lambda: _coerce_tool_result_dict(
            refua_preclinical_batch_record(**kwargs)
        )
    if tool_name == "refua_preclinical_stability_plan" and _PRECLINICAL_AVAILABLE:
        return lambda: _coerce_tool_result_dict(
            refua_preclinical_stability_plan(**kwargs)
        )
    if tool_name == "refua_preclinical_stability_assess" and _PRECLINICAL_AVAILABLE:
        return lambda: _coerce_tool_result_dict(
            refua_preclinical_stability_assess(**kwargs)
        )
    if tool_name == "refua_preclinical_release_assess" and _PRECLINICAL_AVAILABLE:
        return lambda: _coerce_tool_result_dict(
            refua_preclinical_release_assess(**kwargs)
        )
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
        try:
            result = await mcp._tool_manager.call_tool(
                name,
                arguments,
                context=context,
                convert_result=True,
            )
            return _normalize_task_tool_result(result)
        except Exception as exc:
            return _error_call_tool_result(_error_contract_from_exception(exc))

    runner = _build_task_runner(name, arguments)
    if runner is None:
        return _error_call_tool_result(
            _error_contract(
                code="task_mode_unsupported",
                message=f"Task-augmented execution is not implemented for tool '{name}'.",
                hint="Invoke this tool without task augmentation.",
                retryable=False,
            )
        )

    async def work(task_context: Any) -> CallToolResult:
        task_id = str(getattr(task_context, "task_id", ""))
        with _trace_span("refua.task.run", tool=name):
            await task_context.update_status("queued")
            job_id = _submit_job(name, runner)
            if task_id:
                _register_task_job(task_id, job_id)

            started = time.time()
            timeout_seconds = _RUNTIME_CONFIG.task_timeout_seconds

            try:
                while True:
                    snapshot = _job_snapshot(job_id, include_result=True)
                    status = str(snapshot["status"])

                    if status == "success":
                        return _normalize_task_tool_result(snapshot.get("result"))

                    if status in {"error", "cancelled"}:
                        error_payload = snapshot.get("error")
                        if not isinstance(error_payload, Mapping):
                            error_payload = _error_contract(
                                code="task_failed",
                                message=str(error_payload or "Task failed."),
                                hint="Inspect server logs for additional diagnostics.",
                            )
                        return _error_call_tool_result(error_payload)

                    if (
                        timeout_seconds > 0
                        and (time.time() - started) > timeout_seconds
                    ):
                        timeout_error = _error_contract(
                            code="task_timeout",
                            message=(
                                f"Task exceeded timeout of {int(timeout_seconds)} seconds "
                                f"for tool '{name}'."
                            ),
                            hint="Increase REFUA_MCP_TASK_TIMEOUT_SECONDS or reduce workload.",
                            retryable=True,
                        )
                        _cancel_job(job_id, reason=timeout_error)
                        return _error_call_tool_result(timeout_error)

                    if status == "queued":
                        queue_position = snapshot.get("queue_position")
                        if queue_position is not None:
                            await task_context.update_status(
                                f"queued ({queue_position} ahead)"
                            )
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
            finally:
                if task_id:
                    _unregister_task_job(task_id, job_id)

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


def _normalize_string_list_arg(
    value: list[str] | tuple[str, ...] | str | None,
    *,
    field_name: str,
) -> list[str] | None:
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} cannot be an empty string.")
        parsed: Any = text
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{field_name} must be valid JSON when provided as a list string."
                ) from exc
        if isinstance(parsed, str):
            items = [part.strip() for part in parsed.split(",") if part.strip()]
            if not items:
                raise ValueError(f"{field_name} cannot be empty.")
            return items
        value = parsed

    if isinstance(value, (list, tuple)):
        normalized = [str(item).strip() for item in value if str(item).strip()]
        if not normalized:
            raise ValueError(f"{field_name} cannot be empty.")
        return normalized

    raise ValueError(f"{field_name} must be a string or list of strings.")


def _resolve_refua_protein_property_api() -> (
    tuple[Any, Callable[[], Any], Callable[[], Any]]
):
    try:
        import refua as refua_pkg
    except Exception as exc:
        raise ValueError(
            "Protein properties API is unavailable. Install/upgrade refua first."
        ) from exc

    protein_properties_cls = getattr(refua_pkg, "ProteinProperties", None)
    available_properties_fn = getattr(refua_pkg, "available_protein_properties", None)
    available_groups_fn = getattr(refua_pkg, "available_protein_property_groups", None)

    if protein_properties_cls is None or available_properties_fn is None:
        try:
            from refua.protein import (  # noqa: PLC0415
                ProteinProperties,
                available_protein_properties,
            )
        except Exception as exc:
            raise ValueError(
                "Protein properties API is unavailable in this refua build. "
                "Upgrade to a version that includes ProteinProperties."
            ) from exc
        protein_properties_cls = ProteinProperties
        available_properties_fn = available_protein_properties

    if available_groups_fn is None:

        def _empty_groups() -> tuple[str, ...]:
            return ()

        available_groups_fn = _empty_groups

    return (
        protein_properties_cls,
        available_properties_fn,
        available_groups_fn,
    )


def _protein_property_catalog() -> (
    tuple[list[str], list[str], dict[str, dict[str, Any]]]
):
    fallback_names = [
        "length",
        "molecular_weight",
        "isoelectric_point",
        "gravy",
        "net_charge_ph_7_4",
    ]
    fallback_groups = ["basic", "charge", "biophysical"]
    fallback_specs = {
        "length": {"description": "Sequence length in residues.", "groups": ["basic"]},
        "molecular_weight": {
            "description": "Estimated molecular weight in Daltons.",
            "groups": ["basic"],
        },
        "isoelectric_point": {
            "description": "Estimated isoelectric point (pI).",
            "groups": ["basic", "charge"],
        },
        "gravy": {
            "description": "Grand average of hydropathy.",
            "groups": ["basic", "biophysical"],
        },
        "net_charge_ph_7_4": {
            "description": "Estimated net charge at pH 7.4.",
            "groups": ["charge"],
        },
    }

    try:
        (
            _protein_properties_cls,
            available_properties_fn,
            available_groups_fn,
        ) = _resolve_refua_protein_property_api()
        names = sorted(str(name) for name in available_properties_fn())
        groups = sorted(str(group) for group in available_groups_fn())
    except Exception:
        return fallback_names, fallback_groups, fallback_specs

    specs_payload: dict[str, dict[str, Any]] = {}
    try:
        from refua.protein import protein_property_specs  # noqa: PLC0415

        for name, spec in protein_property_specs().items():
            specs_payload[str(name)] = {
                "description": str(getattr(spec, "description", "")),
                "groups": sorted(str(group) for group in getattr(spec, "groups", ())),
            }
    except Exception:
        for name in names:
            specs_payload[name] = {"description": "", "groups": []}

    for name in names:
        specs_payload.setdefault(name, {"description": "", "groups": []})
    return names, groups, specs_payload


def _completion_values(
    candidates: list[str],
    *,
    partial: str,
    limit: int = 100,
) -> Completion:
    needle = partial.strip().lower()
    if needle:
        values = [item for item in candidates if item.lower().startswith(needle)]
    else:
        values = list(candidates)
    total = len(values)
    sliced = values[:limit]
    return Completion(values=sliced, total=total, hasMore=total > len(sliced))


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


def _parse_json_string_arg(value: Any, *, field_name: str) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} cannot be an empty string.")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{field_name} must be valid JSON when provided as a string."
        ) from exc


def _parse_json_strings_in_list(value: Any, *, field_name: str) -> Any:
    if not isinstance(value, (list, tuple)):
        return value
    normalized: list[Any] = []
    for index, item in enumerate(value):
        normalized.append(
            _parse_json_string_arg(item, field_name=f"{field_name}[{index}]")
        )
    return normalized


def _normalize_entities_arg(entities: list[EntitySpec | str] | str) -> list[EntitySpec]:
    parsed = _parse_json_string_arg(entities, field_name="entities")
    parsed = _parse_json_strings_in_list(parsed, field_name="entities")
    return _ENTITY_LIST_ADAPTER.validate_python(parsed)


def _normalize_constraints_arg(
    constraints: list[ConstraintSpec | str] | str | None,
) -> list[ConstraintSpec] | None:
    if constraints is None:
        return None
    parsed = _parse_json_string_arg(constraints, field_name="constraints")
    parsed = _parse_json_strings_in_list(parsed, field_name="constraints")
    return _CONSTRAINT_LIST_ADAPTER.validate_python(parsed)


def _normalize_antibody_arg(antibody: AntibodyEntity | str) -> AntibodyEntity:
    return _ANTIBODY_ENTITY_ADAPTER.validate_python(
        _parse_json_string_arg(antibody, field_name="antibody")
    )


def _normalize_context_entities_arg(
    context_entities: list[ContextEntitySpec | str] | str | None,
) -> list[ContextEntitySpec] | None:
    if context_entities is None:
        return None
    parsed = _parse_json_string_arg(context_entities, field_name="context_entities")
    parsed = _parse_json_strings_in_list(parsed, field_name="context_entities")
    return _CONTEXT_ENTITY_LIST_ADAPTER.validate_python(parsed)


def _enforce_non_exploratory_execution(
    name: str,
    *,
    allow_exploratory_run: bool,
) -> None:
    if allow_exploratory_run:
        return
    if _PROBE_RUN_NAME_RE.search(str(name)):
        raise ValueError(
            "Exploratory/sanity execution names are blocked by default. "
            "Use refua_validate_spec for schema checks, or set "
            "allow_exploratory_run=true to override."
        )


def _normalize_output_requests(
    *,
    structure_output_path: str | None,
    structure_output_format: str | None,
    feature_output_path: str | None,
    feature_output_format: str | None,
) -> tuple[str | None, str | None, str | None, str | None, list[str]]:
    warnings: list[str] = []

    normalized_structure_format = (
        _resolve_output_format(None, structure_output_format)
        if structure_output_format is not None
        else None
    )
    if (
        structure_output_format is not None
        and str(structure_output_format).strip().lower() == "mmcif"
    ):
        warnings.append("structure_output_format='mmcif' was normalized to 'cif'.")

    normalized_feature_path = feature_output_path
    normalized_feature_format = (
        str(feature_output_format).strip().lower()
        if feature_output_format is not None
        else None
    )

    if feature_output_path is not None:
        feature_suffix = Path(feature_output_path).suffix.lower()

        if normalized_feature_format == "json":
            normalized_feature_format = "npz"
            if feature_suffix == ".json":
                normalized_feature_path = str(
                    Path(feature_output_path).with_suffix(".npz")
                )
                warnings.append(
                    "feature_output_format='json' is not a file format; normalized to "
                    f"'npz' and feature_output_path to '{normalized_feature_path}'."
                )
            else:
                warnings.append(
                    "feature_output_format='json' is not a file format; normalized to "
                    "'npz'."
                )
        elif normalized_feature_format is None and feature_suffix == ".json":
            normalized_feature_format = "npz"
            normalized_feature_path = str(Path(feature_output_path).with_suffix(".npz"))
            warnings.append(
                "feature_output_path ending in '.json' is not supported for feature files; "
                f"normalized to '{normalized_feature_path}' with format 'npz'."
            )

        if normalized_feature_format is not None:
            normalized_feature_format = _resolve_feature_output_format(
                normalized_feature_path or feature_output_path,
                normalized_feature_format,
            )

    return (
        structure_output_path,
        normalized_structure_format,
        normalized_feature_path,
        normalized_feature_format,
        warnings,
    )


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
        if normalized not in {"cif", "mmcif", "bcif"}:
            raise ValueError("output_format must be 'cif', 'mmcif', or 'bcif'.")
        if normalized == "mmcif":
            return "cif"
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
        if normalized == "json":
            raise ValueError(
                "output_format='json' is only valid for inline MCP response summaries. "
                "For file output, use 'torch' or 'npz'."
            )
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
        if job.status in {"success", "error", "cancelled"}:
            _JOB_STORE.pop(job_id, None)


def _run_job(job_id: str, runner: Callable[[], dict[str, Any]]) -> None:
    attributes: dict[str, Any] = {"job_id": job_id}
    with _trace_span("refua.job.run", **attributes):
        with _JOB_LOCK:
            job = _JOB_STORE.get(job_id)
            if job is None:
                return
            attributes["tool"] = job.tool
            if job.cancel_requested or job.status == "cancelled":
                if job.finished_at is None:
                    job.finished_at = time.time()
                return

            now = time.time()
            queue_timeout_seconds = _queue_timeout_seconds(job)
            queued_for_seconds = max(0.0, now - job.created_at)
            if queue_timeout_seconds > 0 and queued_for_seconds > queue_timeout_seconds:
                job.status = "error"
                job.error = _error_contract(
                    code="queue_timeout",
                    message=(
                        f"Job exceeded queue timeout of {int(queue_timeout_seconds)} "
                        f"seconds before execution."
                    ),
                    hint=(
                        "Increase REFUA_MCP_QUEUE_TIMEOUT_SECONDS or reduce concurrent "
                        "job volume."
                    ),
                    retryable=True,
                    details={"queued_for_seconds": queued_for_seconds},
                )
                job.finished_at = now
                _metric_add(_JOB_FAILED_COUNTER, 1, attributes={"tool": job.tool})
                return

            job.status = "running"
            job.started_at = now

        started_at = time.time()
        try:
            with _trace_span("refua.job.runner", **attributes):
                result = runner()
        except Exception as exc:
            with _JOB_LOCK:
                job = _JOB_STORE.get(job_id)
                if job is None:
                    return
                if job.status == "cancelled" or job.cancel_requested:
                    if job.finished_at is None:
                        job.finished_at = time.time()
                    return
                job.status = "error"
                job.error = _error_contract_from_exception(exc)
                job.finished_at = time.time()
            _metric_add(
                _JOB_FAILED_COUNTER,
                1,
                attributes={"tool": attributes.get("tool", "unknown")},
            )
            _metric_record(
                _JOB_RUNTIME_HISTOGRAM,
                max(0.0, time.time() - started_at),
                attributes={
                    "tool": attributes.get("tool", "unknown"),
                    "status": "error",
                },
            )
            return

        with _JOB_LOCK:
            job = _JOB_STORE.get(job_id)
            if job is None:
                return
            if job.status == "cancelled" or job.cancel_requested:
                if job.finished_at is None:
                    job.finished_at = time.time()
                return
            job.status = "success"
            job.result = result
            job.finished_at = time.time()

        _metric_add(
            _JOB_COMPLETED_COUNTER,
            1,
            attributes={"tool": attributes.get("tool", "unknown")},
        )
        _metric_record(
            _JOB_RUNTIME_HISTOGRAM,
            max(0.0, time.time() - started_at),
            attributes={"tool": attributes.get("tool", "unknown"), "status": "success"},
        )


def _submit_job(
    tool: str,
    runner: Callable[[], dict[str, Any]],
    *,
    queue_timeout_seconds: float | None = None,
) -> str:
    job_id = uuid.uuid4().hex
    record = JobRecord(
        job_id=job_id,
        tool=tool,
        status="queued",
        created_at=time.time(),
        queue_timeout_seconds=queue_timeout_seconds,
    )
    with _JOB_LOCK:
        _JOB_STORE[job_id] = record
        queue_depth = _queue_depth_locked()
        _prune_jobs_locked()
    _metric_add(_JOB_SUBMITTED_COUNTER, 1, attributes={"tool": tool})
    _metric_record(
        _JOB_QUEUE_DEPTH_HISTOGRAM, float(queue_depth), attributes={"tool": tool}
    )
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
            "cancel_requested": job.cancel_requested,
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
            _metric_record(
                _JOB_QUEUE_DEPTH_HISTOGRAM,
                float(queue_depth),
                attributes={"tool": job.tool},
            )
        if job.status in {"error", "cancelled"} and job.error:
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
    output_warnings: list[str] | None = None,
) -> dict[str, Any]:
    with _trace_span(
        "refua.run_complex_operation",
        action=action,
        entity_count=len(entities),
        request_name=name,
    ):
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

        with _trace_span("refua.build_complex_from_spec", entity_count=len(entities)):
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
                        with _trace_span(
                            "refua.admet.batch", ligand_count=len(targets)
                        ):
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
            with _trace_span("refua.affinity.run"):
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

        with _trace_span(
            "refua.fold.run",
            run_boltz=run_boltz_local,
            run_boltzgen=run_boltzgen_local,
        ):
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
        if output_warnings:
            output["warnings"] = list(output_warnings)
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


def _queue_job(
    tool_name: str,
    runner: Callable[[], BaseModel],
    *,
    queue_timeout_seconds: float | None = None,
) -> QueuedJobResponse:
    job_id = _submit_job(
        tool_name,
        lambda: runner().model_dump(mode="json"),
        queue_timeout_seconds=queue_timeout_seconds,
    )
    return QueuedJobResponse(job_id=job_id)


@mcp.tool()
def refua_fold(
    entities: list[EntitySpec | str] | str,
    *,
    name: str = "complex",
    base_dir: str | None = None,
    constraints: list[ConstraintSpec | str] | str | None = None,
    affinity: AffinityArg = None,
    run_boltz: bool | None = None,
    run_boltzgen: bool | None = None,
    boltz: BoltzOptions | None = None,
    boltzgen: BoltzGenOptions | None = None,
    admet: AdmetArg = None,
    structure_output_path: str | None = None,
    structure_output_format: StructureOutputFormatArg = None,
    return_mmcif: bool = False,
    return_bcif_base64: bool = False,
    feature_output_path: str | None = None,
    feature_output_format: FeatureOutputFormatArg = None,
    allow_exploratory_run: bool = False,
    async_mode: bool = False,
    queue_timeout_seconds: float | None = None,
) -> FoldResult | QueuedJobResponse:
    """Run Refua fold/design workflows with strict typed inputs."""

    _enforce_non_exploratory_execution(
        name,
        allow_exploratory_run=allow_exploratory_run,
    )
    normalized_entities = _normalize_entities_arg(entities)
    normalized_constraints = _normalize_constraints_arg(constraints)
    (
        normalized_structure_output_path,
        normalized_structure_output_format,
        normalized_feature_output_path,
        normalized_feature_output_format,
        output_warnings,
    ) = _normalize_output_requests(
        structure_output_path=structure_output_path,
        structure_output_format=structure_output_format,
        feature_output_path=feature_output_path,
        feature_output_format=feature_output_format,
    )
    entities_payload = _entities_to_payload(normalized_entities)
    constraints_payload = _constraints_to_payload(normalized_constraints)
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
            structure_output_path=normalized_structure_output_path,
            structure_output_format=normalized_structure_output_format,
            return_mmcif=return_mmcif,
            return_bcif_base64=return_bcif_base64,
            feature_output_path=normalized_feature_output_path,
            feature_output_format=normalized_feature_output_format,
            output_warnings=output_warnings,
        )
        return FoldResult.model_validate(output)

    if async_mode:
        return _queue_job(
            "refua_fold",
            run,
            queue_timeout_seconds=queue_timeout_seconds,
        )
    return run()


@mcp.tool()
def refua_affinity(
    entities: list[EntitySpec | str] | str,
    *,
    name: str = "complex",
    base_dir: str | None = None,
    binder: str | None = None,
    boltz: BoltzOptions | None = None,
    admet: AdmetArg = None,
    async_mode: bool = False,
    queue_timeout_seconds: float | None = None,
) -> AffinityResultResponse | QueuedJobResponse:
    """Run affinity-only predictions with strict typed inputs."""

    normalized_entities = _normalize_entities_arg(entities)
    entities_payload = _entities_to_payload(normalized_entities)
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
        return _queue_job(
            "refua_affinity",
            run,
            queue_timeout_seconds=queue_timeout_seconds,
        )
    return run()


@mcp.tool()
def refua_antibody_design(
    antibody: AntibodyEntity | str,
    *,
    context_entities: list[ContextEntitySpec | str] | str | None = None,
    name: str = "antibody_design",
    base_dir: str | None = None,
    constraints: list[ConstraintSpec | str] | str | None = None,
    affinity: AffinityArg = None,
    run_boltz: bool | None = None,
    run_boltzgen: bool | None = None,
    boltz: BoltzOptions | None = None,
    boltzgen: BoltzGenOptions | None = None,
    admet: AdmetArg = None,
    structure_output_path: str | None = None,
    structure_output_format: StructureOutputFormatArg = None,
    return_mmcif: bool = False,
    return_bcif_base64: bool = False,
    feature_output_path: str | None = None,
    feature_output_format: FeatureOutputFormatArg = None,
    allow_exploratory_run: bool = False,
    async_mode: bool = False,
    queue_timeout_seconds: float | None = None,
) -> FoldResult | QueuedJobResponse:
    """Design/fold with an explicit antibody entrypoint plus optional context entities."""

    _enforce_non_exploratory_execution(
        name,
        allow_exploratory_run=allow_exploratory_run,
    )
    antibody_entity = _normalize_antibody_arg(antibody)
    context_entity_list = _normalize_context_entities_arg(context_entities)
    normalized_constraints = _normalize_constraints_arg(constraints)
    (
        normalized_structure_output_path,
        normalized_structure_output_format,
        normalized_feature_output_path,
        normalized_feature_output_format,
        output_warnings,
    ) = _normalize_output_requests(
        structure_output_path=structure_output_path,
        structure_output_format=structure_output_format,
        feature_output_path=feature_output_path,
        feature_output_format=feature_output_format,
    )
    merged_entities: list[EntitySpec] = [*(context_entity_list or []), antibody_entity]
    entities_payload = _entities_to_payload(merged_entities)
    constraints_payload = _constraints_to_payload(normalized_constraints)
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
            structure_output_path=normalized_structure_output_path,
            structure_output_format=normalized_structure_output_format,
            return_mmcif=return_mmcif,
            return_bcif_base64=return_bcif_base64,
            feature_output_path=normalized_feature_output_path,
            feature_output_format=normalized_feature_output_format,
            output_warnings=output_warnings,
        )
        return FoldResult.model_validate(output)

    if async_mode:
        return _queue_job(
            "refua_antibody_design",
            run,
            queue_timeout_seconds=queue_timeout_seconds,
        )
    return run()


@mcp.tool()
def refua_validate_spec(
    entities: list[EntitySpec | str] | str,
    *,
    action: Literal["fold", "affinity"] = "fold",
    name: str = "complex",
    base_dir: str | None = None,
    constraints: list[ConstraintSpec | str] | str | None = None,
    affinity: AffinityArg = None,
    run_boltz: bool | None = None,
    run_boltzgen: bool | None = None,
    boltz: BoltzOptions | None = None,
    boltzgen: BoltzGenOptions | None = None,
    admet: AdmetArg = None,
    structure_output_path: str | None = None,
    structure_output_format: StructureOutputFormatArg = None,
    feature_output_path: str | None = None,
    feature_output_format: FeatureOutputFormatArg = None,
    deep_validate: bool = False,
) -> ValidateSpecResult:
    """Validate and normalize a request without running fold/affinity inference."""

    with _trace_span("refua.validate_spec.parse"):
        normalized_entities = _normalize_entities_arg(entities)
        normalized_constraints = _normalize_constraints_arg(constraints)
        (
            normalized_structure_output_path,
            normalized_structure_output_format,
            normalized_feature_output_path,
            normalized_feature_output_format,
            output_warnings,
        ) = _normalize_output_requests(
            structure_output_path=structure_output_path,
            structure_output_format=structure_output_format,
            feature_output_path=feature_output_path,
            feature_output_format=feature_output_format,
        )
        entities_payload = _entities_to_payload(normalized_entities)
        constraints_payload = _constraints_to_payload(normalized_constraints)
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

    if normalized_structure_output_path or normalized_structure_output_format:
        _resolve_output_format(
            normalized_structure_output_path,
            normalized_structure_output_format,
        )
    if normalized_feature_output_path:
        _resolve_feature_output_format(
            normalized_feature_output_path,
            normalized_feature_output_format,
        )

    warnings: list[str] = list(output_warnings)
    if normalized_feature_output_format and not normalized_feature_output_path:
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
        with _trace_span(
            "refua.validate_spec.deep_validate",
            entity_count=len(entities_payload),
        ):
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
            "structure_output_path": normalized_structure_output_path,
            "structure_output_format": normalized_structure_output_format,
            "feature_output_path": normalized_feature_output_path,
            "feature_output_format": normalized_feature_output_format,
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


@mcp.tool()
def refua_protein_properties(
    sequence: str,
    *,
    properties: list[str] | tuple[str, ...] | str | None = None,
    groups: list[str] | tuple[str, ...] | str | None = None,
    lazy: bool = True,
    sanitize: bool = True,
    include_catalog: bool = False,
) -> ProteinPropertiesResult:
    """Compute protein properties from sequence with Refua's ProteinProperties API."""
    sequence_value = str(sequence or "").strip()
    if not sequence_value:
        raise ValueError("sequence is required.")

    selected_properties = _normalize_string_list_arg(
        properties,
        field_name="properties",
    )
    selected_groups = _normalize_string_list_arg(
        groups,
        field_name="groups",
    )
    if selected_properties is not None and selected_groups is not None:
        raise ValueError("Provide either properties or groups, not both.")

    (
        protein_properties_cls,
        available_properties_fn,
        available_groups_fn,
    ) = _resolve_refua_protein_property_api()

    builder = protein_properties_cls.from_sequence(
        sequence_value,
        lazy=bool(lazy),
        sanitize=bool(sanitize),
    )

    if selected_properties is not None:
        values: dict[str, Any] = {
            prop_name: builder.get(prop_name) for prop_name in selected_properties
        }
    elif selected_groups is not None:
        values = dict(builder.to_dict(groups=selected_groups))
    else:
        values = dict(builder.to_dict())

    available_properties: list[str] | None = None
    available_property_groups: list[str] | None = None
    if include_catalog:
        available_properties = sorted(str(name) for name in available_properties_fn())
        available_property_groups = sorted(
            str(group) for group in available_groups_fn()
        )

    normalized_sequence = str(getattr(builder, "sequence", sequence_value))
    return ProteinPropertiesResult(
        sequence=sequence_value,
        normalized_sequence=normalized_sequence,
        values=values,
        selected_properties=selected_properties,
        selected_groups=selected_groups,
        available_properties=available_properties,
        available_property_groups=available_property_groups,
    )


def _normalize_workup_options(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("workup_options must be an object when provided.")
    return dict(value)


def _normalize_preclinical_study(
    study: Mapping[str, Any] | None,
) -> Any:
    preclinical = _get_preclinical_module()
    if study is None:
        return preclinical.default_study_spec()
    if not isinstance(study, Mapping):
        raise ValueError("study must be an object when provided.")
    return preclinical.study_spec_from_mapping(dict(study))


def _normalize_preclinical_rows(
    rows: list[dict[str, Any]] | None,
    *,
    field_name: str,
    use_template_rows_when_missing: bool,
) -> list[dict[str, Any]] | None:
    preclinical = _get_preclinical_module()
    raw_rows: Any = rows
    if raw_rows is None and use_template_rows_when_missing:
        templates = preclinical.default_templates()
        raw_rows = templates.get("bioanalysis_rows")

    if raw_rows is None:
        return None
    if not isinstance(raw_rows, list):
        raise ValueError(f"{field_name} must be an array of objects when provided.")

    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{field_name}[{idx}] must be an object.")
        normalized.append(dict(row))
    return normalized


def _normalize_preclinical_cmc_config(
    cmc_config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if cmc_config is None:
        return None
    if not isinstance(cmc_config, Mapping):
        raise ValueError("cmc_config must be an object when provided.")
    preclinical = _get_preclinical_module()
    return preclinical.cmc_spec_from_mapping(dict(cmc_config))


def _normalize_preclinical_batch_results(
    batch_results: Mapping[str, Any] | list[dict[str, Any]] | None,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    if batch_results is None:
        return None
    if isinstance(batch_results, Mapping):
        return dict(batch_results)
    if isinstance(batch_results, list):
        normalized: list[dict[str, Any]] = []
        for idx, row in enumerate(batch_results):
            if not isinstance(row, Mapping):
                raise ValueError(f"batch_results[{idx}] must be an object.")
            normalized.append(dict(row))
        return normalized
    raise ValueError(
        "batch_results must be an object or array of objects when provided."
    )


if _PRECLINICAL_AVAILABLE:

    @mcp.tool()
    def refua_preclinical_templates(
        *,
        include_references: bool = True,
    ) -> dict[str, Any]:
        """Return default study/sample templates for refua-preclinical workflows."""
        preclinical = _get_preclinical_module()
        payload: dict[str, Any] = {
            "templates": preclinical.default_templates(),
        }
        if include_references:
            payload["references"] = preclinical.latest_preclinical_references()
            if hasattr(preclinical, "latest_cmc_references"):
                payload["cmc_references"] = preclinical.latest_cmc_references()
        if _REFUA_PRECLINICAL_VERSION is not None:
            payload["refua_preclinical_version"] = _REFUA_PRECLINICAL_VERSION
        return payload

    @mcp.tool()
    def refua_preclinical_cmc_templates(
        *,
        include_references: bool = True,
    ) -> dict[str, Any]:
        """Return default CMC templates for formulation/process/stability/release workflows."""
        preclinical = _get_preclinical_module()
        payload: dict[str, Any] = {
            "templates": preclinical.default_cmc_templates(),
        }
        if include_references and hasattr(preclinical, "latest_cmc_references"):
            payload["references"] = preclinical.latest_cmc_references()
        if _REFUA_PRECLINICAL_VERSION is not None:
            payload["refua_preclinical_version"] = _REFUA_PRECLINICAL_VERSION
        return payload

    @mcp.tool()
    def refua_preclinical_plan(
        study: dict[str, Any] | None = None,
        *,
        seed: int = 7,
        include_markdown: bool = False,
        include_references: bool = False,
    ) -> dict[str, Any]:
        """Build GLP/tox/pharmacology study plan outputs from a study spec."""
        preclinical = _get_preclinical_module()
        spec = _normalize_preclinical_study(study)
        plan = preclinical.build_study_plan(spec, seed=int(seed))
        payload: dict[str, Any] = {
            "study_id": str(getattr(spec, "study_id", plan.get("study_id", ""))),
            "plan": plan,
        }
        if include_markdown:
            payload["plan_markdown"] = preclinical.render_plan_markdown(plan)
        if include_references:
            payload["references"] = preclinical.latest_preclinical_references()
        return payload

    @mcp.tool()
    def refua_preclinical_schedule(
        study: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate in vivo dosing/sampling/observation schedule from a study spec."""
        preclinical = _get_preclinical_module()
        spec = _normalize_preclinical_study(study)
        schedule = preclinical.build_in_vivo_schedule(spec)
        return {
            "study_id": str(getattr(spec, "study_id", schedule.get("study_id", ""))),
            "schedule": schedule,
        }

    @mcp.tool()
    def refua_preclinical_cmc_plan(
        cmc_config: dict[str, Any] | None = None,
        *,
        include_references: bool = True,
    ) -> dict[str, Any]:
        """Build CMC formulation/process lifecycle outputs from an optional CMC config."""
        preclinical = _get_preclinical_module()
        normalized_cmc = _normalize_preclinical_cmc_config(cmc_config)
        cmc_plan = preclinical.build_formulation_process_plan(normalized_cmc)
        payload: dict[str, Any] = {"cmc_plan": cmc_plan}
        if include_references and hasattr(preclinical, "latest_cmc_references"):
            payload["references"] = preclinical.latest_cmc_references()
        return payload

    @mcp.tool()
    def refua_preclinical_batch_record(
        cmc_config: dict[str, Any] | None = None,
        *,
        batch_id: str = "BATCH-001",
        operator: str = "TBD",
        site: str = "TBD",
        manufacture_date: str | None = None,
    ) -> dict[str, Any]:
        """Generate an electronic batch record from an optional CMC config."""
        preclinical = _get_preclinical_module()
        normalized_cmc = _normalize_preclinical_cmc_config(cmc_config)
        batch_record = preclinical.generate_batch_record(
            normalized_cmc,
            batch_id=str(batch_id),
            operator=str(operator),
            site=str(site),
            manufacture_date=manufacture_date,
        )
        return {"batch_record": batch_record}

    @mcp.tool()
    def refua_preclinical_stability_plan(
        cmc_config: dict[str, Any] | None = None,
        *,
        batch_ids: list[str] | None = None,
        include_references: bool = True,
    ) -> dict[str, Any]:
        """Build stability schedule/sample plan from an optional CMC config."""
        preclinical = _get_preclinical_module()
        normalized_cmc = _normalize_preclinical_cmc_config(cmc_config)
        normalized_batch_ids = None
        if isinstance(batch_ids, list):
            normalized_batch_ids = [
                str(item).strip() for item in batch_ids if str(item).strip()
            ]
        stability_plan = preclinical.build_stability_study_plan(
            normalized_cmc,
            batch_ids=normalized_batch_ids,
        )
        payload: dict[str, Any] = {"stability_plan": stability_plan}
        if include_references and hasattr(preclinical, "latest_cmc_references"):
            payload["references"] = preclinical.latest_cmc_references()
        return payload

    @mcp.tool()
    def refua_preclinical_stability_assess(
        rows: list[dict[str, Any]] | None = None,
        *,
        cmc_config: dict[str, Any] | None = None,
        include_references: bool = False,
        use_template_rows_when_missing: bool = True,
    ) -> dict[str, Any]:
        """Assess stability rows against release criteria and trend summaries."""
        preclinical = _get_preclinical_module()
        normalized_rows = rows
        if normalized_rows is None and use_template_rows_when_missing:
            templates = preclinical.default_cmc_templates()
            raw_rows = templates.get("stability_results_rows")
            if isinstance(raw_rows, list):
                normalized_rows = [
                    dict(item) for item in raw_rows if isinstance(item, Mapping)
                ]
        if not isinstance(normalized_rows, list):
            raise ValueError(
                "rows are required when use_template_rows_when_missing=false."
            )
        for idx, row in enumerate(normalized_rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"rows[{idx}] must be an object.")
        normalized_cmc = _normalize_preclinical_cmc_config(cmc_config)
        criteria = preclinical.build_formulation_process_plan(normalized_cmc)["cmc"][
            "release_criteria"
        ]
        assessment = preclinical.assess_stability_results(
            [dict(item) for item in normalized_rows],
            release_criteria=criteria,
        )
        payload: dict[str, Any] = {"stability_assessment": assessment}
        if include_references and hasattr(preclinical, "latest_cmc_references"):
            payload["references"] = preclinical.latest_cmc_references()
        return payload

    @mcp.tool()
    def refua_preclinical_release_assess(
        batch_results: dict[str, Any] | list[dict[str, Any]],
        *,
        cmc_config: dict[str, Any] | None = None,
        stability_results: list[dict[str, Any]] | None = None,
        include_references: bool = False,
    ) -> dict[str, Any]:
        """Evaluate release decision from batch results and optional stability evidence."""
        preclinical = _get_preclinical_module()
        normalized_cmc = _normalize_preclinical_cmc_config(cmc_config)
        normalized_batch_results = _normalize_preclinical_batch_results(batch_results)
        if normalized_batch_results is None:
            raise ValueError("batch_results must be provided.")
        cmc_plan = preclinical.build_formulation_process_plan(normalized_cmc)
        criteria = cmc_plan["cmc"]["release_criteria"]
        cqa = cmc_plan["cmc"]["critical_quality_attributes"]

        stability_assessment = None
        if stability_results is not None:
            for idx, row in enumerate(stability_results):
                if not isinstance(row, Mapping):
                    raise ValueError(f"stability_results[{idx}] must be an object.")
            stability_assessment = preclinical.assess_stability_results(
                [dict(item) for item in stability_results],
                release_criteria=criteria,
            )

        release_assessment = preclinical.evaluate_release_criteria(
            batch_results=normalized_batch_results,
            release_criteria=criteria,
            stability_assessment=stability_assessment,
            critical_quality_attributes=cqa,
        )
        payload: dict[str, Any] = {
            "release_assessment": release_assessment,
            "stability_assessment": stability_assessment,
        }
        if include_references and hasattr(preclinical, "latest_cmc_references"):
            payload["references"] = preclinical.latest_cmc_references()
        return payload

    @mcp.tool()
    def refua_preclinical_bioanalysis(
        study: dict[str, Any] | None = None,
        *,
        rows: list[dict[str, Any]] | None = None,
        lloq_ng_ml: float = 1.0,
        use_template_rows_when_missing: bool = True,
    ) -> dict[str, Any]:
        """Run bioanalytical ETL/QC summary + simple NCA outputs."""
        if lloq_ng_ml <= 0:
            raise ValueError("lloq_ng_ml must be > 0.")
        preclinical = _get_preclinical_module()
        spec = _normalize_preclinical_study(study)
        normalized_rows = _normalize_preclinical_rows(
            rows,
            field_name="rows",
            use_template_rows_when_missing=bool(use_template_rows_when_missing),
        )
        if normalized_rows is None:
            raise ValueError(
                "rows are required when use_template_rows_when_missing=false."
            )
        bioanalysis = preclinical.run_bioanalytical_pipeline(
            spec,
            normalized_rows,
            lloq_ng_ml=float(lloq_ng_ml),
        )
        return {
            "study_id": str(getattr(spec, "study_id", bioanalysis.get("study_id", ""))),
            "bioanalysis": bioanalysis,
        }

    @mcp.tool()
    def refua_preclinical_workup(
        study: dict[str, Any] | None = None,
        *,
        samples: list[dict[str, Any]] | None = None,
        seed: int = 7,
        lloq_ng_ml: float = 1.0,
        cmc_config: dict[str, Any] | None = None,
        stability_results: list[dict[str, Any]] | None = None,
        batch_results: dict[str, Any] | list[dict[str, Any]] | None = None,
        batch_id: str = "BATCH-001",
        include_markdown: bool = False,
        include_references: bool = False,
        use_template_rows_when_missing: bool = False,
    ) -> dict[str, Any]:
        """Build an integrated preclinical package with plan/schedule and optional bioanalysis."""
        if lloq_ng_ml <= 0:
            raise ValueError("lloq_ng_ml must be > 0.")
        preclinical = _get_preclinical_module()
        spec = _normalize_preclinical_study(study)
        normalized_samples = _normalize_preclinical_rows(
            samples,
            field_name="samples",
            use_template_rows_when_missing=bool(use_template_rows_when_missing),
        )
        normalized_cmc = _normalize_preclinical_cmc_config(cmc_config)
        normalized_stability_results = _normalize_preclinical_rows(
            stability_results,
            field_name="stability_results",
            use_template_rows_when_missing=False,
        )
        normalized_batch_results = _normalize_preclinical_batch_results(batch_results)
        workup = preclinical.build_workup(
            spec,
            samples=normalized_samples,
            seed=int(seed),
            lloq_ng_ml=float(lloq_ng_ml),
            cmc_config=normalized_cmc,
            stability_results=normalized_stability_results,
            batch_results=normalized_batch_results,
            batch_id=str(batch_id),
        )
        payload: dict[str, Any] = {
            "study_id": str(getattr(spec, "study_id", workup.get("study_id", ""))),
            "workup": workup,
        }
        if include_markdown:
            plan_payload = workup.get("plan")
            if isinstance(plan_payload, Mapping):
                payload["plan_markdown"] = preclinical.render_plan_markdown(
                    dict(plan_payload)
                )
        if include_references:
            payload["references"] = preclinical.latest_preclinical_references()
            if hasattr(preclinical, "latest_cmc_references"):
                payload["cmc_references"] = preclinical.latest_cmc_references()
        return payload


if _CLINICAL_AVAILABLE:

    @mcp.tool()
    def refua_clinical_simulator(
        config: dict[str, Any] | None = None,
        *,
        trial_id: str | None = None,
        indication: str | None = None,
        phase: str | None = None,
        objective: str | None = None,
        seed: int | None = None,
        replicates: int | None = None,
        include_replicates: bool = False,
        include_workup: bool = False,
        workup_options: dict[str, Any] | None = None,
        admet_profile: dict[str, Any] | None = None,
        refua_payload: dict[str, Any] | None = None,
        apply_refua_payload: bool = True,
        refua_ligand_id: str | None = None,
        refua_max_candidate_arms: int = 4,
    ) -> dict[str, Any]:
        """Run refua-clinical simulation with optional Refua payload integration."""

        study_cls = _get_clinical_study_cls()
        if config is None:
            study = study_cls.default()
        else:
            if not isinstance(config, Mapping):
                raise ValueError("config must be an object when provided.")
            study = study_cls.from_config(dict(config))
        study.trial(
            trial_id=trial_id,
            indication=indication,
            phase=phase,
            objective=objective,
            seed=seed,
            replicates=replicates,
        )

        if admet_profile is not None:
            if not isinstance(admet_profile, Mapping):
                raise ValueError("admet_profile must be an object when provided.")
            study.admet_profile(dict(admet_profile), apply=True)

        if refua_payload is not None:
            if not isinstance(refua_payload, Mapping):
                raise ValueError("refua_payload must be an object when provided.")
            study.refua_payload(
                dict(refua_payload),
                apply=bool(apply_refua_payload),
                ligand_id=refua_ligand_id,
                max_candidate_arms=max(1, int(refua_max_candidate_arms)),
            )

        run = study.simulate(replicates=replicates, seed=seed)
        run_payload = run.to_dict()
        if not include_replicates:
            run_payload.pop("replicates", None)

        response: dict[str, Any] = {
            "run_id": run.run_id,
            "summary": dict(run.summary),
            "run": run_payload,
        }

        if include_workup:
            options = _normalize_workup_options(workup_options)
            workup = run.workup(**options)
            workup_payload: dict[str, Any] = {
                "protocol": workup.protocol.to_dict(),
                "optimization": workup.optimization.to_dict(),
                "voi": workup.voi.to_dict(),
                "advice": workup.advice.to_dict(),
            }
            if workup.transportability is not None:
                workup_payload["transportability"] = workup.transportability
            response["workup"] = workup_payload

        return response


if _DATA_AVAILABLE:

    @mcp.tool()
    def refua_data_list(
        *,
        tag: str | None = None,
        limit: int = 200,
        include_usage_notes: bool = True,
        include_urls: bool = False,
        cache_root: str | None = None,
    ) -> dict[str, Any]:
        """List datasets from refua-data catalog (optionally filtered by tag)."""
        resolved_cache_root = _normalize_cache_root(cache_root)
        manager = _get_refua_data_manager(resolved_cache_root)
        tag_value = str(tag).strip() if tag is not None else None
        if tag_value == "":
            tag_value = None
        if limit < 1:
            raise ValueError("limit must be >= 1.")

        datasets = manager.list_datasets(tag=tag_value)
        selected = datasets[: int(limit)]
        items: list[dict[str, Any]] = []
        for dataset in selected:
            snapshot = dict(dataset.metadata_snapshot())
            if not include_usage_notes:
                snapshot.pop("usage_notes", None)
            if not include_urls:
                snapshot.pop("urls", None)
            items.append(snapshot)

        return {
            "tag": tag_value,
            "count": len(items),
            "total_available": len(datasets),
            "datasets": items,
            "cache_root": resolved_cache_root,
        }

    @mcp.tool()
    def refua_data_fetch(
        dataset_id: str,
        *,
        force: bool = False,
        refresh: bool = False,
        timeout_seconds: float = 120.0,
        cache_root: str | None = None,
        include_metadata: bool = False,
    ) -> dict[str, Any]:
        """Fetch a refua-data dataset into local cache."""
        dataset_key = str(dataset_id or "").strip()
        if not dataset_key:
            raise ValueError("dataset_id is required.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0.")

        resolved_cache_root = _normalize_cache_root(cache_root)
        manager = _get_refua_data_manager(resolved_cache_root)
        result = manager.fetch(
            dataset_key,
            force=bool(force),
            refresh=bool(refresh),
            timeout_seconds=float(timeout_seconds),
        )

        payload: dict[str, Any] = {
            "dataset_id": result.dataset_id,
            "version": result.version,
            "raw_path": str(result.raw_path),
            "metadata_path": str(result.metadata_path),
            "source_url": result.source_url,
            "cache_hit": bool(result.cache_hit),
            "refreshed": bool(result.refreshed),
            "bytes_downloaded": int(result.bytes_downloaded),
            "sha256": result.sha256,
            "cache_root": resolved_cache_root,
        }
        if include_metadata:
            meta = manager.cache.read_json(result.metadata_path)
            payload["metadata"] = meta if isinstance(meta, dict) else {}
        return payload

    @mcp.tool()
    def refua_data_materialize(
        dataset_id: str,
        *,
        force: bool = False,
        refresh: bool = False,
        chunksize: int = 100_000,
        timeout_seconds: float = 120.0,
        cache_root: str | None = None,
        include_manifest: bool = False,
    ) -> dict[str, Any]:
        """Fetch + materialize a refua-data dataset into chunked parquet."""
        dataset_key = str(dataset_id or "").strip()
        if not dataset_key:
            raise ValueError("dataset_id is required.")
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0.")

        resolved_cache_root = _normalize_cache_root(cache_root)
        manager = _get_refua_data_manager(resolved_cache_root)
        result = manager.materialize(
            dataset_key,
            force=bool(force),
            refresh=bool(refresh),
            chunksize=int(chunksize),
            timeout_seconds=float(timeout_seconds),
        )

        payload: dict[str, Any] = {
            "dataset_id": result.dataset_id,
            "version": result.version,
            "parquet_dir": str(result.parquet_dir),
            "manifest_path": str(result.manifest_path),
            "parts": [str(path) for path in result.parts],
            "part_count": len(result.parts),
            "row_count": int(result.row_count),
            "cache_hit": bool(result.cache_hit),
            "source_sha256": result.source_sha256,
            "cache_root": resolved_cache_root,
        }
        if include_manifest:
            manifest = manager.cache.read_json(result.manifest_path)
            payload["manifest"] = manifest if isinstance(manifest, dict) else {}
        return payload

    @mcp.tool()
    def refua_data_query(
        dataset_id: str,
        *,
        columns: list[str] | tuple[str, ...] | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        cache_root: str | None = None,
        materialize_if_missing: bool = True,
        force_materialize: bool = False,
        refresh: bool = False,
        chunksize: int = 100_000,
        timeout_seconds: float = 120.0,
    ) -> dict[str, Any]:
        """Query materialized parquet rows from a refua-data dataset."""
        if limit < 1:
            raise ValueError("limit must be >= 1.")
        if limit > 5000:
            raise ValueError("limit must be <= 5000.")
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0.")

        dataset_key = str(dataset_id or "").strip()
        if not dataset_key:
            raise ValueError("dataset_id is required.")

        query_columns = _normalize_string_column_list(columns, field_name="columns")
        query_filters = _normalize_data_query_filters(filters)
        resolved_cache_root = _normalize_cache_root(cache_root)
        manager = _get_refua_data_manager(resolved_cache_root)

        manifest: dict[str, Any] = {}
        manifest_path_text: str | None = None
        if materialize_if_missing:
            materialized = manager.materialize(
                dataset_key,
                force=bool(force_materialize),
                refresh=bool(refresh),
                chunksize=int(chunksize),
                timeout_seconds=float(timeout_seconds),
            )
            parts = list(materialized.parts)
            manifest_path_text = str(materialized.manifest_path)
            manifest_raw = manager.cache.read_json(materialized.manifest_path)
            if isinstance(manifest_raw, dict):
                manifest = dict(manifest_raw)
            dataset_meta = manager.catalog.get(dataset_key).metadata_snapshot()
        else:
            dataset = manager.catalog.get(dataset_key)
            dataset_meta = dataset.metadata_snapshot()
            parquet_dir = manager.cache.parquet_dir(dataset)
            manifest_path = manager.cache.parquet_manifest(dataset)
            manifest_raw = manager.cache.read_json(manifest_path)
            if not isinstance(manifest_raw, dict):
                raise ValueError(
                    f"Dataset '{dataset_key}' has no parquet manifest. Set materialize_if_missing=true."
                )
            manifest_path_text = str(manifest_path)
            manifest = dict(manifest_raw)
            parts_raw = manifest.get("parts")
            if not isinstance(parts_raw, list) or not parts_raw:
                raise ValueError(
                    f"Dataset '{dataset_key}' parquet manifest has no parts."
                )
            parts = [parquet_dir.joinpath(str(name)) for name in parts_raw]
            if not all(path.exists() for path in parts):
                raise ValueError(
                    f"Dataset '{dataset_key}' parquet parts are missing. Re-materialize with force_materialize=true."
                )

        import pandas as pd

        rows: list[dict[str, Any]] = []
        scanned_rows = 0
        scanned_parts = 0
        for part in parts:
            frame = pd.read_parquet(part, columns=query_columns)
            scanned_parts += 1
            scanned_rows += int(len(frame))

            filtered = _apply_data_query_filters(frame, query_filters)
            if filtered.empty:
                continue
            if len(rows) >= limit:
                break
            remaining = int(limit) - len(rows)
            batch = filtered.head(remaining)
            rows.extend(batch.to_dict(orient="records"))
            if len(rows) >= limit:
                break

        row_count_estimate = manifest.get("row_count")
        row_count: int | None = None
        if isinstance(row_count_estimate, (int, float, str)):
            try:
                row_count = int(row_count_estimate)
            except (TypeError, ValueError):
                row_count = None
        return {
            "dataset_id": dataset_key,
            "columns": query_columns,
            "filters": query_filters,
            "limit": int(limit),
            "returned_rows": len(rows),
            "scanned_rows": scanned_rows,
            "scanned_parts": scanned_parts,
            "row_count_estimate": row_count,
            "rows": rows,
            "dataset": dataset_meta,
            "cache_root": resolved_cache_root,
            "manifest_path": manifest_path_text,
        }


def _capabilities_payload() -> dict[str, Any]:
    try:
        _resolve_refua_protein_property_api()
        protein_property_api = True
    except Exception:
        protein_property_api = False
    property_names, property_groups, _ = _protein_property_catalog()

    return {
        "mcp_spec_revision": MCP_SPEC_REVISION,
        "mcp_latest_protocol_version": str(LATEST_PROTOCOL_VERSION),
        "mcp_sdk_version": _MCP_SDK_VERSION,
        "refua_version": _REFUA_VERSION,
        "refua_clinical_version": _REFUA_CLINICAL_VERSION,
        "refua_data_version": _REFUA_DATA_VERSION,
        "refua_preclinical_version": _REFUA_PRECLINICAL_VERSION,
        "runtime": {
            "transport": _RUNTIME_CONFIG.transport,
            "host": _RUNTIME_CONFIG.host,
            "port": _RUNTIME_CONFIG.port,
            "mount_path": _RUNTIME_CONFIG.mount_path,
            "task_timeout_seconds": _RUNTIME_CONFIG.task_timeout_seconds,
            "queue_timeout_seconds": _RUNTIME_CONFIG.queue_timeout_seconds,
            "dns_rebinding_protection": _RUNTIME_CONFIG.enable_dns_rebinding_protection,
            "allowed_hosts": list(_RUNTIME_CONFIG.allowed_hosts),
            "allowed_origins": list(_RUNTIME_CONFIG.allowed_origins),
            "auth_enabled": bool(_RUNTIME_CONFIG.token_count),
            "auth_token_count": _RUNTIME_CONFIG.token_count,
        },
        "features": {
            "admet_available": _ADMET_AVAILABLE,
            "clinical_simulator_available": _CLINICAL_AVAILABLE,
            "data_available": _DATA_AVAILABLE,
            "preclinical_available": _PRECLINICAL_AVAILABLE,
            "protein_properties_api_available": protein_property_api,
            "otel_available": _OTEL_AVAILABLE,
            "experimental_tasks_enabled": True,
        },
        "task_support_by_tool": {
            tool_name: _task_support_mode(tool_name)
            for tool_name in _TASK_SUPPORT_BY_TOOL
        },
        "protein_property_counts": {
            "properties": len(property_names),
            "groups": len(property_groups),
        },
    }


@mcp.resource(
    "refua://capabilities",
    name="refua_capabilities",
    description="Runtime capabilities and feature flags for this Refua MCP server.",
)
def refua_capabilities() -> str:
    return json.dumps(_capabilities_payload(), indent=2)


@mcp.resource(
    "refua://protein-properties/index",
    name="refua_protein_property_index",
    description="Index of protein property names/groups available through Refua.",
)
def refua_protein_property_index() -> str:
    names, groups, _ = _protein_property_catalog()
    return json.dumps(
        {
            "template_uris": [
                "refua://protein-properties/group/{group_name}",
                "refua://protein-properties/property/{property_name}",
            ],
            "property_names": names,
            "property_groups": groups,
            "count": {"properties": len(names), "groups": len(groups)},
        },
        indent=2,
    )


@mcp.resource(
    "refua://protein-properties/group/{group_name}",
    name="refua_protein_property_group",
    description="Protein property names for a specific property group.",
)
def refua_protein_property_group(group_name: str) -> str:
    names, groups, specs = _protein_property_catalog()
    normalized = str(group_name).strip().lower()
    if normalized not in {group.lower() for group in groups}:
        raise ValueError(
            f"Unknown protein property group '{group_name}'. Available: {groups}"
        )
    grouped = [
        name
        for name in names
        if normalized
        in {
            str(group).strip().lower()
            for group in specs.get(name, {}).get("groups", [])
        }
    ]
    return json.dumps(
        {
            "group_name": normalized,
            "properties": grouped,
            "count": len(grouped),
        },
        indent=2,
    )


@mcp.resource(
    "refua://protein-properties/property/{property_name}",
    name="refua_protein_property_detail",
    description="Property metadata for a single protein property name.",
)
def refua_protein_property_detail(property_name: str) -> str:
    names, _groups, specs = _protein_property_catalog()
    normalized = str(property_name).strip()
    if normalized not in names:
        raise ValueError(
            f"Unknown protein property '{property_name}'. Available count: {len(names)}"
        )
    payload = {
        "property_name": normalized,
        "description": specs.get(normalized, {}).get("description", ""),
        "groups": specs.get(normalized, {}).get("groups", []),
    }
    return json.dumps(payload, indent=2)


@mcp.completion()
async def refua_completion(
    ref: PromptReference | ResourceTemplateReference,
    argument: Any,
    context: Any | None,
) -> Completion | None:
    del context
    if isinstance(ref, PromptReference):
        return None
    ref_uri = str(ref.uri)
    arg_name = str(argument.name)
    partial = str(argument.value or "")

    if ref_uri == "refua://recipes/{recipe_name}" and arg_name == "recipe_name":
        return _completion_values(sorted(_RECIPE_LIBRARY), partial=partial)

    if (
        ref_uri == "refua://protein-properties/group/{group_name}"
        and arg_name == "group_name"
    ):
        _names, groups, _specs = _protein_property_catalog()
        return _completion_values(groups, partial=partial)

    if (
        ref_uri == "refua://protein-properties/property/{property_name}"
        and arg_name == "property_name"
    ):
        names, _groups, _specs = _protein_property_catalog()
        return _completion_values(names, partial=partial)

    return None


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
    "protein_properties": {
        "tool": "refua_protein_properties",
        "args": {
            "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
            "groups": ["basic"],
            "include_catalog": True,
        },
    },
}

if _DATA_AVAILABLE:
    _RECIPE_LIBRARY["data_materialize"] = {
        "tool": "refua_data_materialize",
        "args": {
            "dataset_id": "chembl_activity_ki_human",
            "chunksize": 100000,
            "refresh": False,
        },
    }
    _RECIPE_LIBRARY["data_query"] = {
        "tool": "refua_data_query",
        "args": {
            "dataset_id": "chembl_activity_ki_human",
            "columns": ["molecule_chembl_id", "standard_value", "standard_units"],
            "filters": {"standard_units": "nM"},
            "limit": 25,
        },
    }

if _CLINICAL_AVAILABLE:
    _RECIPE_LIBRARY["clinical_simulation"] = {
        "tool": "refua_clinical_simulator",
        "args": {
            "trial_id": "small_molecule_phase2",
            "indication": "Oncology",
            "phase": "Phase II",
            "replicates": 80,
            "include_workup": True,
            "include_replicates": False,
        },
    }

if _PRECLINICAL_AVAILABLE:
    _RECIPE_LIBRARY["preclinical_plan"] = {
        "tool": "refua_preclinical_plan",
        "args": {
            "seed": 7,
            "include_markdown": True,
            "include_references": True,
        },
    }
    _RECIPE_LIBRARY["preclinical_workup"] = {
        "tool": "refua_preclinical_workup",
        "args": {
            "seed": 7,
            "lloq_ng_ml": 1.0,
            "use_template_rows_when_missing": True,
            "include_references": True,
        },
    }
    _RECIPE_LIBRARY["preclinical_cmc_plan"] = {
        "tool": "refua_preclinical_cmc_plan",
        "args": {
            "include_references": True,
        },
    }
    _RECIPE_LIBRARY["preclinical_cmc_release"] = {
        "tool": "refua_preclinical_release_assess",
        "args": {
            "batch_results": {
                "assay_percent": 99.0,
                "content_uniformity_av": 9.8,
                "dissolution_q30_percent": 92.0,
                "total_impurities_percent": 0.7,
                "water_content_percent": 1.5,
                "appearance_score": 5.0,
            },
            "include_references": True,
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
    cancel: bool = False,
) -> dict[str, Any]:
    """Check status for a background refua job.

    Responses may include recommended_poll_seconds plus queue/estimate metadata for
    queued or running jobs.

    wait_for_terminal_seconds optionally blocks until the job reaches a terminal state
    (success/error/cancelled) or the timeout is reached. Use this to reduce
    client-side polling.
    """
    if cancel:
        _cancel_job(
            job_id,
            reason=_error_contract(
                code="job_cancelled",
                message="Job cancelled by client request.",
                hint="Resubmit the job if you still need the result.",
                retryable=True,
            ),
        )

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

    @lowlevel.experimental.cancel_task()
    async def _cancel_task_with_job_cleanup(req: Any) -> Any:
        from mcp.shared.experimental.tasks.helpers import cancel_task as mcp_cancel_task

        support = lowlevel.experimental.task_support
        if support is None:
            raise RuntimeError("Task support is not enabled.")
        result = await mcp_cancel_task(support.store, req.params.taskId)
        mapped_job_id = _lookup_task_job(req.params.taskId)
        if mapped_job_id is not None:
            _cancel_job(
                mapped_job_id,
                reason=_error_contract(
                    code="task_cancelled",
                    message=f"Task '{req.params.taskId}' was cancelled by the client.",
                    hint="Resubmit the task to run it again.",
                    retryable=True,
                ),
            )
            _unregister_task_job(req.params.taskId, mapped_job_id)
        return result


_configure_experimental_task_support()


def main() -> None:
    run_mount_path = (
        _RUNTIME_CONFIG.mount_path
        if _RUNTIME_CONFIG.transport in {"sse", "streamable-http"}
        else None
    )
    mcp.run(
        transport=_RUNTIME_CONFIG.transport,
        mount_path=run_mount_path,
    )


if __name__ == "__main__":
    main()
