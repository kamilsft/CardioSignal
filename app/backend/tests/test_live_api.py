import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from cardiosignal_api.config import Settings
from cardiosignal_api.live import routes
from cardiosignal_api.live.monitor import LIVE_FS
from cardiosignal_api.main import create_app
from test_live_monitor import synthetic_ecg

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples.npz"
TOKEN = "test-token"
DEVICE_HEADERS = {"X-Device-Token": TOKEN}


class NoModel:
    """The live monitor never runs the translation model."""


def make_client(**overrides) -> TestClient:
    options = {"samples_path": SAMPLES, "live_ingest_token": TOKEN, "live_tick_seconds": 0.05,
               "rate_limit_per_minute": 10_000, **overrides}
    settings = replace(Settings(), **options)
    return TestClient(create_app(settings, translator=NoModel()))


@pytest.fixture
def client():
    with make_client() as c:
        yield c


def send_seconds(ws, seconds: float, bpm: float = 72) -> None:
    signal = synthetic_ecg(seconds, bpm=bpm)
    for start in range(0, len(signal), LIVE_FS // 4):
        ws.send_json({"samples": signal[start:start + LIVE_FS // 4].tolist()})


def wait_for_patients(client, predicate, timeout: float = 3.0) -> list[dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        patients = client.get("/api/live/patients").json()["data"]["patients"]
        if predicate(patients):
            return patients
        time.sleep(0.05)
    raise AssertionError(f"condition not met, last: {patients}")


def test_device_stream_shows_up_with_heart_rate(client):
    with client.websocket_connect("/api/live/ingest/bed-12", headers=DEVICE_HEADERS) as ws:
        send_seconds(ws, 8, bpm=72)
        patients = wait_for_patients(client, lambda ps: ps and ps[0]["status"] == "ok")
    assert patients[0]["id"] == "bed-12" and patients[0]["heart_rate_bpm"] == pytest.approx(72, rel=0.05)


def test_dashboard_stream_receives_snapshots(client):
    with client.websocket_connect("/api/live/ingest/bed-1", headers=DEVICE_HEADERS) as device:
        send_seconds(device, 2)
        with client.websocket_connect("/api/live/stream", headers={"Origin": "http://localhost:5173"}) as ws:
            message = ws.receive_json()
    assert message["type"] == "snapshot" and message["summary"]["patients"] == 1


@pytest.mark.parametrize("headers", [{}, {"X-Device-Token": "wrong"}])
def test_ingest_requires_valid_token(client, headers):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/api/live/ingest/bed-1", headers=headers) as ws:
            ws.receive_json()
    assert exc.value.code == 1008


def test_ingest_is_disabled_without_configured_token():
    with make_client(live_ingest_token="") as c, pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect("/api/live/ingest/bed-1", headers=DEVICE_HEADERS) as ws:
            ws.receive_json()
    assert exc.value.code == 1008 and "disabled" in exc.value.reason


def test_invalid_patient_id_is_rejected(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/api/live/ingest/bad%20id!", headers=DEVICE_HEADERS) as ws:
            ws.receive_json()
    assert exc.value.code == 1008


@pytest.mark.parametrize("message", [
    {"samples": "nope"},
    {"samples": [1.0, float("nan")]},
    {"samples": [0.0] * (LIVE_FS + 1)},
    {"other": [1.0]},
    {"samples": [True, 1.0]},
    {"samples": ["1.0"]},
    "[" * 20_000,
])
def test_malformed_device_messages_close_the_connection(client, message):
    with client.websocket_connect("/api/live/ingest/bed-1", headers=DEVICE_HEADERS) as ws:
        ws.send_text(message) if isinstance(message, str) else ws.send_json(message)
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.code == 1003


def test_device_sending_faster_than_real_time_is_disconnected(client):
    with client.websocket_connect("/api/live/ingest/bed-1", headers=DEVICE_HEADERS) as ws:
        for _ in range(15):     # 15 s of data at once: more than the 10 s reconnect backlog allowed
            ws.send_json({"samples": [0.0] * LIVE_FS})
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.code == 1008


def test_reconnecting_device_replaces_its_old_connection(client):
    with client.websocket_connect("/api/live/ingest/bed-1", headers=DEVICE_HEADERS) as old:
        with client.websocket_connect("/api/live/ingest/bed-1", headers=DEVICE_HEADERS) as new:
            with pytest.raises(WebSocketDisconnect) as exc:
                old.receive_json()
            assert exc.value.code == routes.REPLACED
            send_seconds(new, 8)
            patients = wait_for_patients(client, lambda ps: ps and ps[0]["status"] == "ok")
    assert [p["id"] for p in patients] == ["bed-1"]


def test_full_monitor_asks_device_to_retry_later():
    with make_client(live_max_patients=1) as c:
        with c.websocket_connect("/api/live/ingest/bed-1", headers=DEVICE_HEADERS):
            with pytest.raises(WebSocketDisconnect) as exc:
                with c.websocket_connect("/api/live/ingest/bed-2", headers=DEVICE_HEADERS) as ws:
                    ws.receive_json()
    assert exc.value.code == routes.TRY_AGAIN_LATER


def test_silent_device_is_disconnected_and_frees_its_slot(monkeypatch):
    monkeypatch.setattr(routes, "IDLE_TIMEOUT_SECONDS", 0.2)
    with make_client(live_max_patients=1) as c:
        with c.websocket_connect("/api/live/ingest/bed-1", headers=DEVICE_HEADERS) as ws:
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
        assert exc.value.code == routes.GOING_AWAY
        with c.websocket_connect("/api/live/ingest/bed-2", headers=DEVICE_HEADERS) as ws:
            send_seconds(ws, 1)


def test_viewers_per_ip_are_limited():
    with make_client(live_max_viewers_per_ip=1) as c:
        with c.websocket_connect("/api/live/stream") as first:
            first.receive_json()
            with pytest.raises(WebSocketDisconnect) as exc:
                with c.websocket_connect("/api/live/stream") as second:
                    second.receive_json()
    assert exc.value.code == routes.TRY_AGAIN_LATER


def test_dashboard_from_foreign_origin_is_refused(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/api/live/stream", headers={"Origin": "https://evil.example"}) as ws:
            ws.receive_json()
    assert exc.value.code == 1008
