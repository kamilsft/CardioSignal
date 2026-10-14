from pathlib import Path

import numpy as np
import pytest

from cardiosignal_api.live.monitor import LIVE_FS, heart_rate
from cardiosignal_api.live.simulator import SimulatedPatient, build_loop, make_patients
from cardiosignal_api.samples import SampleStore

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples.npz"
SOURCE_FS = 2000


def synthetic_source(bpm: float = 72, seconds: float = 6) -> np.ndarray:
    x = np.zeros(int(seconds * SOURCE_FS))
    x[int(0.2 * SOURCE_FS)::int(SOURCE_FS * 60 / bpm)] = 1.0
    return np.convolve(x, np.hanning(80), mode="same")


def test_loop_is_resampled_scaled_and_starts_on_a_beat():
    loop = build_loop(synthetic_source(bpm=60), source_fs=SOURCE_FS, stretch=1.0)
    assert np.max(np.abs(loop)) == pytest.approx(1.0, abs=1e-6)
    assert np.argmax(loop[: LIVE_FS // 2]) < LIVE_FS // 20       # first beat within 50 ms of the start
    assert len(loop) == pytest.approx(5 * LIVE_FS, abs=3)        # whole beats only: 5 RR intervals of 1 s


def test_stretch_slows_the_heart_rate():
    loop = build_loop(synthetic_source(bpm=80), source_fs=SOURCE_FS, stretch=1.25)
    assert heart_rate(np.tile(loop, 3)) == pytest.approx(64, rel=0.05)


def test_patient_chunks_wrap_around_seamlessly():
    patient = SimulatedPatient("p", loop=np.arange(10, dtype=np.float32), noise=0.0, seed=0, start=8)
    assert patient.next_chunk(5).tolist() == [8, 9, 0, 1, 2]
    assert patient.next_chunk(3).tolist() == [3, 4, 5]


def test_replayed_patient_keeps_a_steady_heart_rate():
    patient = SimulatedPatient("p", loop=build_loop(synthetic_source(bpm=70), SOURCE_FS, 1.0), noise=0.01, seed=1)
    stream = np.concatenate([patient.next_chunk(LIVE_FS // 4) for _ in range(40)])   # 10 s in 250 ms chunks
    assert heart_rate(stream[-8 * LIVE_FS:]) == pytest.approx(70, rel=0.05)


def test_make_patients_from_real_recordings_is_deterministic():
    store = SampleStore(SAMPLES)
    a = make_patients(store, count=12, seed=7)
    b = make_patients(store, count=12, seed=7)
    assert [p.patient_id for p in a] == [f"patient-{i:04d}" for i in range(1, 13)]
    assert all(np.array_equal(x.next_chunk(100), y.next_chunk(100)) for x, y in zip(a, b))


def test_simulated_real_recordings_have_plausible_heart_rates():
    for patient in make_patients(SampleStore(SAMPLES), count=6, seed=0):
        bpm = heart_rate(patient.next_chunk(8 * LIVE_FS))
        assert bpm is not None and 35 <= bpm <= 160, (patient.patient_id, bpm)


def test_make_patients_requires_recordings(tmp_path):
    with pytest.raises(ValueError, match="No recordings"):
        make_patients(SampleStore(tmp_path / "missing.npz"), count=3, seed=0)
