from __future__ import annotations

from mcp.types import CallToolResult

import refua_mcp.server as server


def test_recommend_poll_seconds_without_estimate_scales_by_queue() -> None:
    assert server._recommend_poll_seconds(None, queue_position=0) == 30
    assert server._recommend_poll_seconds(None, queue_position=1) == 45
    assert server._recommend_poll_seconds(None, queue_position=6) == 120


def test_recommend_poll_seconds_with_estimate_uses_floor_and_cap() -> None:
    assert server._recommend_poll_seconds(5, queue_position=0) == 30
    assert server._recommend_poll_seconds(60, queue_position=0) == 30
    assert server._recommend_poll_seconds(300, queue_position=0) == 105
    assert server._recommend_poll_seconds(1_200, queue_position=0) == 120


def test_task_support_mode_defaults() -> None:
    assert server._task_support_mode("refua_fold") == "optional"
    assert server._task_support_mode("refua_affinity") == "optional"
    assert server._task_support_mode("refua_antibody_design") == "optional"
    assert server._task_support_mode("unknown_tool") == "forbidden"


def test_normalize_task_tool_result_structured_dict() -> None:
    result = server._normalize_task_tool_result({"ok": True})
    assert isinstance(result, CallToolResult)
    assert result.isError is False
    assert result.structuredContent == {"ok": True}


def test_long_poll_sleep_seconds_bounds() -> None:
    assert server._long_poll_sleep_seconds({"recommended_poll_seconds": 1}, 20) == 5
    assert server._long_poll_sleep_seconds({"recommended_poll_seconds": 300}, 20) == 20
    assert server._long_poll_sleep_seconds({"recommended_poll_seconds": 30}, 20) == 20


def test_poll_job_until_terminal_waits_once_then_returns(monkeypatch) -> None:
    snapshots = iter(
        [
            {"status": "running", "recommended_poll_seconds": 40},
            {"status": "success", "result": {"ok": True}},
        ]
    )
    now = {"t": 100.0}
    sleep_calls: list[float] = []

    def fake_snapshot(job_id: str, include_result: bool) -> dict[str, object]:
        return dict(next(snapshots))

    def fake_time() -> float:
        return now["t"]

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        now["t"] += seconds

    monkeypatch.setattr(server, "_job_snapshot", fake_snapshot)
    monkeypatch.setattr(server.time, "time", fake_time)
    monkeypatch.setattr(server.time, "sleep", fake_sleep)

    result = server._poll_job_until_terminal(
        "job-1",
        include_result=True,
        wait_for_terminal_seconds=120,
    )
    assert result["status"] == "success"
    assert sleep_calls == [40]


def test_poll_job_until_terminal_respects_timeout(monkeypatch) -> None:
    now = {"t": 100.0}
    sleep_calls: list[float] = []
    call_count = {"n": 0}

    def fake_snapshot(job_id: str, include_result: bool) -> dict[str, object]:
        call_count["n"] += 1
        return {"status": "running", "recommended_poll_seconds": 30}

    def fake_time() -> float:
        return now["t"]

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        now["t"] += seconds

    monkeypatch.setattr(server, "_job_snapshot", fake_snapshot)
    monkeypatch.setattr(server.time, "time", fake_time)
    monkeypatch.setattr(server.time, "sleep", fake_sleep)

    result = server._poll_job_until_terminal(
        "job-2",
        include_result=False,
        wait_for_terminal_seconds=10,
    )
    assert result["status"] == "running"
    assert sleep_calls == [10]
    assert call_count["n"] == 2


def test_refua_job_wait_for_terminal_uses_long_poll(monkeypatch) -> None:
    expected = {"status": "success", "result": {"done": True}}

    def fake_long_poll(
        job_id: str,
        *,
        include_result: bool,
        wait_for_terminal_seconds: float,
    ) -> dict[str, object]:
        assert job_id == "abc123"
        assert include_result is True
        assert wait_for_terminal_seconds == 9
        return expected

    monkeypatch.setattr(server, "_poll_job_until_terminal", fake_long_poll)
    result = server.refua_job(
        "abc123",
        include_result=True,
        wait_for_terminal_seconds=9,
    )
    assert result == expected
