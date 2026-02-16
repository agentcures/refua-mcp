from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anyio
import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

import refua_mcp.server as server

ROOT = Path(__file__).resolve().parents[1]


def _server_python() -> str:
    override = os.environ.get("REFUA_MCP_TEST_SERVER_PYTHON")
    if override:
        return override
    return sys.executable


async def _with_session() -> tuple[ClientSession, object]:
    params = StdioServerParameters(
        command=_server_python(),
        args=["-m", "refua_mcp.server"],
        cwd=ROOT,
        env={
            "PYTHONPATH": str(ROOT / "src"),
            "REFUA_MCP_TRANSPORT": "stdio",
        },
    )
    client_cm = stdio_client(params)
    read_stream, write_stream = await client_cm.__aenter__()
    session_cm = ClientSession(read_stream, write_stream)
    session = await session_cm.__aenter__()

    class _Closer:
        async def aclose(self) -> None:
            await session_cm.__aexit__(None, None, None)
            await client_cm.__aexit__(None, None, None)

    return session, _Closer()


def test_sdk_acceptance_initialize_and_discovery() -> None:
    async def _run() -> None:
        session, closer = await _with_session()
        try:
            init = await session.initialize()
            assert init.protocolVersion == server.MCP_SPEC_REVISION

            tools = await session.list_tools()
            tool_names = {tool.name for tool in tools.tools}
            assert "refua_protein_properties" in tool_names
            assert "refua_fold" in tool_names

            templates = await session.list_resource_templates()
            template_uris = {
                template.uriTemplate for template in templates.resourceTemplates
            }
            assert "refua://recipes/{recipe_name}" in template_uris
            assert "refua://protein-properties/group/{group_name}" in template_uris
            assert (
                "refua://protein-properties/property/{property_name}" in template_uris
            )

            capabilities = await session.read_resource("refua://capabilities")
            text = capabilities.contents[0].text
            payload = json.loads(text)
            assert payload["mcp_spec_revision"] == server.MCP_SPEC_REVISION
        finally:
            await closer.aclose()

    anyio.run(_run)


def test_sdk_acceptance_completion_and_tool_call() -> None:
    async def _run() -> None:
        session, closer = await _with_session()
        try:
            await session.initialize()

            recipe_completion = await session.complete(
                types.ResourceTemplateReference(
                    type="ref/resource",
                    uri="refua://recipes/{recipe_name}",
                ),
                {"name": "recipe_name", "value": "pro"},
            )
            assert "protein_properties" in recipe_completion.completion.values

            result = await session.call_tool(
                "refua_protein_properties",
                {
                    "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
                    "groups": ["basic"],
                },
            )
            assert result.isError is False
            assert isinstance(result.structuredContent, dict)
            assert "values" in result.structuredContent
        finally:
            await closer.aclose()

    anyio.run(_run)


def test_sdk_acceptance_task_augmented_validate_spec() -> None:
    async def _run() -> None:
        session, closer = await _with_session()
        try:
            await session.initialize()
            task = await session.experimental.call_tool_as_task(
                "refua_validate_spec",
                {
                    "entities": [
                        {
                            "type": "protein",
                            "id": "A",
                            "sequence": "MKTAYIAK",
                        }
                    ]
                },
                ttl=120000,
            )
            task_id = task.task.taskId

            terminal = None
            for _ in range(200):
                status = await session.experimental.get_task(task_id)
                if status.status in {"completed", "failed", "cancelled"}:
                    terminal = status
                    break
                await anyio.sleep(0.01)

            assert terminal is not None
            assert terminal.status == "completed"

            final = await session.experimental.get_task_result(
                task_id,
                types.CallToolResult,
            )
            assert final.isError is False
            assert isinstance(final.structuredContent, dict)
            assert final.structuredContent.get("valid") is True
        finally:
            await closer.aclose()

    anyio.run(_run)
