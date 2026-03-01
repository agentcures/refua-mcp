from __future__ import annotations

import threading
import time

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
    expected = "optional" if server._CLINICAL_AVAILABLE else "forbidden"
    assert server._task_support_mode("refua_clinical_simulator") == expected
    data_expected = "optional" if server._DATA_AVAILABLE else "forbidden"
    assert server._task_support_mode("refua_data_query") == data_expected
    preclinical_expected = "optional" if server._PRECLINICAL_AVAILABLE else "forbidden"
    assert server._task_support_mode("refua_preclinical_workup") == preclinical_expected
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


def _wait_for_terminal(job_id: str, *, timeout: float = 5.0) -> dict[str, object]:
    deadline = time.time() + timeout
    snapshot = server._job_snapshot(job_id, include_result=True)
    while snapshot["status"] in {"queued", "running"}:
        if time.time() >= deadline:
            raise TimeoutError(f"Job did not reach terminal state: {job_id}")
        time.sleep(0.01)
        snapshot = server._job_snapshot(job_id, include_result=True)
    return snapshot


def test_cancel_job_marks_queued_job_cancelled() -> None:
    gate = threading.Event()

    def blocking_runner() -> dict[str, bool]:
        gate.wait(timeout=2.0)
        return {"ok": True}

    first = server._submit_job("test_tool", blocking_runner, queue_timeout_seconds=30)
    second = server._submit_job(
        "test_tool",
        lambda: {"ok": True},
        queue_timeout_seconds=30,
    )

    snapshot = server._cancel_job(
        second,
        reason={
            "code": "job_cancelled",
            "message": "cancelled from test",
            "retryable": True,
        },
    )
    assert snapshot["status"] == "cancelled"
    assert snapshot["error"]["code"] == "job_cancelled"

    gate.set()
    _wait_for_terminal(first)


def test_queue_timeout_fails_stale_job() -> None:
    gate = threading.Event()

    def blocking_runner() -> dict[str, bool]:
        gate.wait(timeout=2.0)
        return {"ok": True}

    first = server._submit_job("test_tool", blocking_runner, queue_timeout_seconds=30)
    second = server._submit_job(
        "test_tool",
        lambda: {"ok": True},
        queue_timeout_seconds=0.01,
    )

    time.sleep(0.05)
    gate.set()

    first_snapshot = _wait_for_terminal(first)
    assert first_snapshot["status"] in {"success", "cancelled", "error"}

    second_snapshot = _wait_for_terminal(second)
    assert second_snapshot["status"] == "error"
    assert second_snapshot["error"]["code"] == "queue_timeout"
