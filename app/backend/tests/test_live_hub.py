import asyncio

import numpy as np
import pytest

from cardiosignal_api.live.analyzer import ShardAnalyzer
from cardiosignal_api.live.hub import MemoryHub
from cardiosignal_api.live.monitor import LIVE_FS
from test_live_monitor import synthetic_ecg


def feed(hub: MemoryHub, pid: str, samples: np.ndarray, now: float) -> None:
    asyncio.run(hub.ingest(pid, samples, now))


def fed_hub(patients: int, seconds: float = 8, **kwargs) -> MemoryHub:
    hub = MemoryHub(**kwargs)
    for i in range(patients):
        pid = f"p{i:03d}"
        assert hub.connect(pid)
        feed(hub, pid, synthetic_ecg(seconds, bpm=60 + i), now=seconds)
    return hub


def test_snapshot_reports_heart_rate_per_patient():
    snap = fed_hub(3).tick(now=8.0)
    by_id = {p["id"]: p for p in snap["patients"]}
    assert set(by_id) == {"p000", "p001", "p002"}
    assert by_id["p002"]["heart_rate_bpm"] == pytest.approx(62, rel=0.05)
    assert by_id["p002"]["rr_irregularity"] < 0.1 and by_id["p002"]["alerts"] == []
    assert snap["summary"] | {"analysis_ms": 0} == {
        "patients": 3, "ok": 3, "warming_up": 0, "no_signal": 0, "alerts_active": 0, "analysis_ms": 0,
        "devices_connected": 3, "workers": 1, "pipeline_lag_ms": 0.0}


def test_connected_patient_is_listed_before_first_sample():
    hub = MemoryHub()
    hub.connect("new")
    assert hub.tick(now=0.0)["patients"][0]["status"] == "no_signal"


def test_snapshot_sends_new_waveform_samples_once():
    hub = fed_hub(1, seconds=2)
    first = hub.tick(now=2.0)["waveforms"]["p000"]
    assert first["fs"] == LIVE_FS // 2 and len(first["samples"]) == LIVE_FS   # 2 s at half rate
    assert hub.tick(now=2.5)["waveforms"]["p000"]["samples"] == []


def test_waveforms_are_capped():
    snap = fed_hub(5, max_waveforms=2).tick(now=8.0)
    assert len(snap["patients"]) == 5 and set(snap["waveforms"]) == {"p000", "p001"}


def test_capacity_limit_and_duplicate_connections_are_refused():
    hub = MemoryHub(max_patients=1)
    assert hub.connect("a")
    assert not hub.connect("a")
    assert not hub.connect("b")


def test_disconnected_patient_shows_no_signal_then_is_dropped():
    hub = fed_hub(1)
    hub.disconnect("p000")
    assert hub.tick(now=14.0)["patients"][0]["status"] == "no_signal"
    assert hub.tick(now=8.0 + ShardAnalyzer.STALE_SECONDS + 1)["patients"] == []


def test_reconnect_after_disconnect_is_allowed():
    hub = fed_hub(1)
    hub.disconnect("p000")
    assert hub.connect("p000")


def test_ingest_for_unknown_patient_is_ignored():
    hub = MemoryHub()
    feed(hub, "ghost", np.zeros(10), now=0.0)
    assert hub.tick(now=0.0)["patients"] == []


def test_fast_heart_rate_raises_an_alert_and_moves_patient_first():
    hub = MemoryHub(max_waveforms=1)
    for pid in ("a-calm", "z-racing"):
        hub.connect(pid)
    for second in range(20):
        now = float(second + 8)
        feed(hub, "a-calm", synthetic_ecg(1, bpm=70, seed=second), now)
        feed(hub, "z-racing", synthetic_ecg(1, bpm=170, seed=second), now)
        snap = hub.tick(now)
    assert [p["id"] for p in snap["patients"]] == ["z-racing", "a-calm"]
    assert snap["patients"][0]["alerts"] == ["high_heart_rate"]
    assert list(snap["waveforms"]) == ["z-racing"]            # the only waveform slot goes to the alert
    assert snap["summary"]["alerts_active"] == 1
    [alert] = snap["alerts"]["active"]
    assert alert["patient_id"] == "z-racing" and alert["severity"] == "critical"
    assert snap["alerts"]["recent"][0]["id"] == alert["id"]


def test_publish_drops_oldest_message_for_slow_subscriber():
    hub = MemoryHub()
    queue = hub.subscribe()
    for i in range(5):
        hub.publish({"n": i})
    received = []
    while not queue.empty():
        received.append(queue.get_nowait()["n"])
    assert received[-1] == 4 and len(received) <= MemoryHub.SUBSCRIBER_QUEUE
    hub.unsubscribe(queue)
    hub.publish({"n": 5})
    assert queue.empty() or queue.get_nowait()["n"] != 5


def test_run_publishes_snapshots_periodically():
    async def scenario():
        hub = MemoryHub()
        hub.connect("p")
        await hub.ingest("p", synthetic_ecg(8), hub.clock())
        queue = hub.subscribe()
        task = asyncio.create_task(hub.run(interval=0.01))
        message = await asyncio.wait_for(queue.get(), timeout=2)
        task.cancel()
        return message
    message = asyncio.run(scenario())
    assert message["type"] == "snapshot" and message["summary"]["patients"] == 1
    assert message["patients"][0]["status"] == "ok"
