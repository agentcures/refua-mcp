from __future__ import annotations

import mcp.types as types

import refua_mcp.server as server


def test_protocol_target_revision_matches_sdk_latest() -> None:
    assert server.MCP_SPEC_REVISION == "2025-11-25"
    assert str(types.LATEST_PROTOCOL_VERSION) == server.MCP_SPEC_REVISION


def test_tasks_handlers_are_registered_for_experimental_support() -> None:
    lowlevel = server.mcp._mcp_server
    assert lowlevel.experimental.task_support is not None

    required_handlers = {
        types.GetTaskRequest,
        types.GetTaskPayloadRequest,
        types.ListTasksRequest,
        types.CancelTaskRequest,
    }
    assert required_handlers.issubset(set(lowlevel.request_handlers))
