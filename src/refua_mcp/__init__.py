"""Refua MCP server package."""

from importlib.metadata import PackageNotFoundError, version as _version
from pathlib import Path
import tomllib

__all__ = ["__version__"]


def _read_version_from_pyproject() -> str | None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject_path.exists():
        return None
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    version = project.get("version")
    if not version:
        return None
    return str(version)


_local_version = _read_version_from_pyproject()

try:
    _installed_version = _version("refua-mcp")
except PackageNotFoundError:
    __version__ = _local_version or "unknown"
else:
    __version__ = (
        _local_version
        if _local_version is not None and _installed_version != _local_version
        else _installed_version
    )
