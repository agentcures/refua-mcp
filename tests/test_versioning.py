import json
from pathlib import Path
import tomllib

from refua_mcp import __version__

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def test_version_matches_pyproject() -> None:
    assert __version__ == _pyproject_version()


def test_mcp_json_version_matches_pyproject() -> None:
    data = json.loads((ROOT / "mcp.json").read_text(encoding="utf-8"))
    assert data["version"] == _pyproject_version()


def test_mcp_json_uses_python3() -> None:
    data = json.loads((ROOT / "mcp.json").read_text(encoding="utf-8"))
    assert data["command"] == "python3"
