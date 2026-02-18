from __future__ import annotations

import json

import anyio
from mcp.types import CompletionArgument, ResourceTemplateReference
import pytest

import refua_mcp.server as server


def test_capabilities_resource_reports_core_runtime_flags() -> None:
    payload = json.loads(server.refua_capabilities())
    assert payload["mcp_spec_revision"] == server.MCP_SPEC_REVISION
    assert payload["mcp_latest_protocol_version"] == server.LATEST_PROTOCOL_VERSION
    assert "features" in payload
    assert "runtime" in payload
    assert payload["runtime"]["task_timeout_seconds"] >= 0
    assert payload["runtime"]["queue_timeout_seconds"] >= 0
    assert (
        payload["features"]["clinical_simulator_available"]
        == server._CLINICAL_AVAILABLE
    )
    assert payload["features"]["data_available"] == server._DATA_AVAILABLE


def test_protein_property_resources_are_exposed() -> None:
    index = json.loads(server.refua_protein_property_index())
    assert "property_names" in index
    assert "property_groups" in index
    assert index["count"]["properties"] == len(index["property_names"])


def test_recipe_index_conditionally_includes_clinical_recipe() -> None:
    payload = json.loads(server.refua_recipe_index())
    names = set(payload["recipe_names"])
    if server._CLINICAL_AVAILABLE:
        assert "clinical_simulation" in names
    else:
        assert "clinical_simulation" not in names
    if server._DATA_AVAILABLE:
        assert "data_materialize" in names
        assert "data_query" in names
    else:
        assert "data_materialize" not in names
        assert "data_query" not in names


def test_recipe_completion_suggests_known_recipes() -> None:
    async def _run() -> None:
        completion = await server.refua_completion(
            ResourceTemplateReference(
                type="ref/resource",
                uri="refua://recipes/{recipe_name}",
            ),
            CompletionArgument(name="recipe_name", value="pro"),
            None,
        )
        assert completion is not None
        assert "protein_properties" in completion.values

    anyio.run(_run)


def test_protein_property_completion_suggests_groups() -> None:
    async def _run() -> None:
        completion = await server.refua_completion(
            ResourceTemplateReference(
                type="ref/resource",
                uri="refua://protein-properties/group/{group_name}",
            ),
            CompletionArgument(name="group_name", value="ba"),
            None,
        )
        assert completion is not None
        assert any(value.startswith("ba") for value in completion.values)

    anyio.run(_run)


def test_runtime_config_parses_secure_streamable_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REFUA_MCP_TRANSPORT", "streamable-http")
    monkeypatch.setenv("REFUA_MCP_HOST", "127.0.0.1")
    monkeypatch.setenv("REFUA_MCP_PORT", "9100")
    monkeypatch.setenv("REFUA_MCP_AUTH_TOKENS", "token-a,token-b")
    monkeypatch.setenv("REFUA_MCP_ALLOWED_HOSTS", "127.0.0.1:9100")
    monkeypatch.setenv("REFUA_MCP_ALLOWED_ORIGINS", "http://127.0.0.1:9100")
    config = server._build_runtime_server_config()

    assert config.transport == "streamable-http"
    assert config.token_count == 2
    assert config.enable_dns_rebinding_protection is True
    assert config.allowed_hosts == ("127.0.0.1:9100",)
    assert config.allowed_origins == ("http://127.0.0.1:9100",)
