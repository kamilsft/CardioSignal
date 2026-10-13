import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cardiosignal_api.config import Settings
from cardiosignal_api.main import create_app
from cardiosignal_api.rate_limit import RateLimiter
from cardiosignal_api.translator import Translator, window_starts

BACKEND = Path(__file__).resolve().parent.parent
MODEL = BACKEND / "models" / "model.pt"
SAMPLES = BACKEND / "data" / "samples.npz"


def synthetic_ecg(seconds=6, fs=2000, bpm=72):
    x = np.zeros(seconds * fs)
    x[::int(fs * 60 / bpm)] = 1.0
    return np.convolve(x, np.hanning(40), mode="same") + 0.01 * np.random.default_rng(0).standard_normal(len(x))


@pytest.fixture(scope="module")
def translator():
    return Translator(str(MODEL), torch_threads=2)


def make_client(translator, **overrides):
    settings = replace(Settings(), samples_path=SAMPLES, **overrides)
    return TestClient(create_app(settings, translator=translator))


@pytest.fixture
def client(translator):
    with make_client(translator) as c:
        yield c


def test_health(client):
    body = client.get("/api/health").json()
    assert body["success"] and body["data"]["model_loaded"] and body["data"]["samples"] == 6


def test_samples_list_and_get(client):
    items = client.get("/api/samples").json()["data"]
    assert {i["label"] for i in items} == {"healthy", "cvd"}
    sample = client.get(f"/api/samples/{items[0]['id']}").json()["data"]
    assert sample["sample_rate"] == 2000 and len(sample["ecg"]) == len(sample["pcg"]) == 12000


def test_unknown_sample_is_404_envelope(client):
    r = client.get("/api/samples/nope")
    assert r.status_code == 404 and r.json() == {"success": False, "data": None, "error": "Sample 'nope' not found."}


def test_translate_ecg_to_pcg_json(client):
    r = client.post("/api/translate", json={"direction": "ecg_to_pcg", "sample_rate": 2000,
                                            "signal": synthetic_ecg().tolist()})
    data = r.json()["data"]
    assert r.status_code == 200 and data["sample_rate"] == 2000
    assert len(data["output_signal"]) == len(data["input_signal"]) == 12000
    assert max(abs(v) for v in data["output_signal"]) <= 1.0
    assert data["input_analysis"]["heart_rate_bpm"] == pytest.approx(72, abs=2)


def test_translate_resamples_and_flips_inverted_ecg(client):
    ecg_500hz = -synthetic_ecg(seconds=10, fs=500)
    data = client.post("/api/translate", json={"direction": "ecg_to_pcg", "sample_rate": 500,
                                               "signal": ecg_500hz.tolist()}).json()["data"]
    assert len(data["output_signal"]) == 20000           # 10 s at 2 kHz
    assert data["ecg_was_flipped"] and any("upside down" in w for w in data["warnings"])


def test_translate_pcg_to_ecg_upload_csv(client, translator):
    pcg = translator.translate(synthetic_ecg(), 2000, "ecg_to_pcg").output_signal
    csv = "\n".join(f"{v:.5f}" for v in pcg).encode()
    r = client.post("/api/translate/upload", data={"direction": "pcg_to_ecg", "sample_rate": "2000"},
                    files={"file": ("heart.csv", io.BytesIO(csv), "text/csv")})
    data = r.json()["data"]
    assert r.status_code == 200 and data["direction"] == "pcg_to_ecg"
    assert data["input_analysis"]["quality"] is not None


def test_invalid_input_returns_friendly_422(client):
    r = client.post("/api/translate", json={"direction": "ecg_to_pcg", "sample_rate": 2000, "signal": [0.1] * 100})
    assert r.status_code == 422 and not r.json()["success"] and "at least" in r.json()["error"]
    r = client.post("/api/translate", json={"direction": "sideways", "sample_rate": 2000, "signal": [1.0]})
    assert r.status_code == 422 and r.json()["error"].startswith("Invalid request: direction")


def test_upload_too_large(translator):
    with make_client(translator, max_upload_bytes=10) as c:
        r = c.post("/api/translate/upload", data={"direction": "ecg_to_pcg", "sample_rate": "100"},
                   files={"file": ("big.csv", io.BytesIO(b"1\n" * 100), "text/csv")})
    assert r.status_code == 422 and "too large" in r.json()["error"]


def test_rate_limit(translator):
    with make_client(translator, rate_limit_per_minute=1) as c:
        body = {"direction": "ecg_to_pcg", "sample_rate": 2000, "signal": [0.0] * 10}
        c.post("/api/translate", json=body)
        r = c.post("/api/translate", json=body)
    assert r.status_code == 429 and "Too many requests" in r.json()["error"]


def test_busy_server_returns_503(translator):
    with make_client(translator, max_concurrent_inferences=1) as c:
        slots = c.app.state.inference_slots
        c.portal.call(slots.acquire)   # occupy the only slot from the app's own event loop
        try:
            r = c.post("/api/translate", json={"direction": "ecg_to_pcg", "sample_rate": 2000,
                                               "signal": synthetic_ecg().tolist()})
        finally:
            slots.release()
    assert r.status_code == 503 and "busy" in r.json()["error"]


def test_unknown_route_uses_envelope(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.json()["success"] is False and r.json()["error"]


def test_input_signal_is_exactly_what_the_model_saw(client):
    data = client.post("/api/translate", json={"direction": "ecg_to_pcg", "sample_rate": 2000,
                                               "signal": synthetic_ecg(seconds=10).tolist()}).json()["data"]
    last_window = np.array(data["input_signal"][-12000:])
    assert np.max(np.abs(last_window)) == pytest.approx(1.0, abs=1e-3)   # scaled as one 6 s window


def test_rate_limiter_evicts_idle_clients_and_caps_memory():
    now = [0.0]
    limiter = RateLimiter(5, window_seconds=60, clock=lambda: now[0], max_clients=3)
    assert all(limiter.allow(f"ip{i}") for i in range(3))
    assert not limiter.allow("ip-new")          # full, and nobody is idle yet: refuse instead of growing
    now[0] = 61
    assert limiter.allow("ip-new")              # idle clients were swept
    assert limiter.tracked_clients() == 1


def test_rate_limiter_window():
    now = [0.0]
    limiter = RateLimiter(2, window_seconds=60, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    assert limiter.allow("b")
    now[0] = 61
    assert limiter.allow("a")


def test_window_starts_cover_tail():
    assert window_starts(5000) == [0]
    assert window_starts(30000) == [0, 12000, 18000]
    assert window_starts(24000) == [0, 12000]
