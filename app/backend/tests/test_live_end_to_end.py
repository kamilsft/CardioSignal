"""The real simulator streaming over real WebSockets into a real uvicorn server."""
import asyncio
import socket
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
import uvicorn

from cardiosignal_api.config import Settings
from cardiosignal_api.live.simulator import FatalDeviceError, main, make_patients, parse_args, run, stream_patient
from cardiosignal_api.main import create_app
from cardiosignal_api.samples import SampleStore

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples.npz"
TOKEN = "e2e-token"


class NoModel:
    pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    settings = replace(Settings(), samples_path=SAMPLES, live_ingest_token=TOKEN, live_tick_seconds=0.1)
    app = create_app(settings, translator=NoModel())
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started and time.monotonic() < deadline:
        time.sleep(0.05)
    yield app, f"ws://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=5)


async def stream_for(seconds: float, coroutine) -> None:
    try:
        await asyncio.wait_for(coroutine, timeout=seconds)
    except TimeoutError:
        pass


def test_simulated_patients_appear_on_the_monitor(server):
    app, url = server
    patients = make_patients(SampleStore(SAMPLES), count=3, seed=0)
    asyncio.run(stream_for(4.0, run(patients, url, TOKEN, chunk_seconds=0.25)))
    latest = app.state.hub.latest
    by_id = {p["id"]: p for p in latest["patients"]}
    assert {"patient-0001", "patient-0002", "patient-0003"} <= set(by_id)
    assert all(by_id[p.patient_id]["status"] == "ok" for p in patients), by_id


def test_wrong_token_stops_the_device(server):
    _, url = server
    patient = make_patients(SampleStore(SAMPLES), count=1, seed=1)[0]
    with pytest.raises(FatalDeviceError, match="token"):
        asyncio.run(asyncio.wait_for(stream_patient(patient, url, "wrong", 0.25), timeout=5))


def test_unreachable_server_is_retried(caplog):   # Windows needs ~2 s to report a refused connection
    patient = make_patients(SampleStore(SAMPLES), count=1, seed=2)[0]
    asyncio.run(stream_for(5.0, stream_patient(patient, f"ws://127.0.0.1:{free_port()}", TOKEN, 0.25)))
    assert "retrying" in caplog.text


def test_main_requires_token(monkeypatch):
    monkeypatch.delenv("LIVE_INGEST_TOKEN", raising=False)
    assert main(["--patients", "1"]) == 2


@pytest.mark.parametrize("argv", [["--patients", "0"], ["--chunk-ms", "5"]])
def test_parse_args_rejects_bad_values(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)
