import numpy as np
import pytest

from cardiosignal_api.live.monitor import LIVE_FS, PatientStream, heart_rate, rhythm, vitals_for


def synthetic_ecg(seconds: float, fs: int = LIVE_FS, bpm: float = 72, seed: int = 0) -> np.ndarray:
    x = np.zeros(int(seconds * fs))
    x[::int(fs * 60 / bpm)] = 1.0
    return np.convolve(x, np.hanning(10), mode="same") + 0.01 * np.random.default_rng(seed).standard_normal(len(x))


def test_ring_buffer_keeps_latest_samples_in_order():
    stream = PatientStream("p1", fs=10, window_seconds=1)        # capacity 10
    stream.push(np.arange(6), now=0.0)
    stream.push(np.arange(6, 15), now=1.0)
    assert stream.window().tolist() == list(range(5, 15))


def test_window_before_full_returns_only_received_samples():
    stream = PatientStream("p1", fs=10, window_seconds=1)
    stream.push(np.array([1.0, 2.0]), now=0.0)
    assert stream.window().tolist() == [1.0, 2.0]


def test_take_new_returns_each_sample_once():
    stream = PatientStream("p1", fs=10, window_seconds=1)
    stream.push(np.array([1.0, 2.0]), now=0.0)
    stream.push(np.array([3.0]), now=0.1)
    assert stream.take_new().tolist() == [1.0, 2.0, 3.0]
    assert stream.take_new().tolist() == []


def test_take_new_is_capped_to_buffer_capacity():
    stream = PatientStream("p1", fs=10, window_seconds=1)
    stream.push(np.arange(25), now=0.0)
    assert stream.take_new().tolist() == list(range(15, 25))


def test_take_new_in_pairs_holds_back_the_odd_sample():
    stream = PatientStream("p1", fs=10, window_seconds=1)
    stream.push(np.array([1.0, 2.0, 3.0]), now=0.0)
    assert stream.take_new(multiple=2).tolist() == [1.0, 2.0]
    stream.push(np.array([4.0]), now=0.1)
    assert stream.take_new(multiple=2).tolist() == [3.0, 4.0]


def test_push_tracks_last_seen():
    stream = PatientStream("p1")
    stream.push(np.zeros(5), now=12.5)
    assert stream.last_seen == 12.5


@pytest.mark.parametrize("bpm", [30, 48, 72, 110, 150, 180])
def test_heart_rate_on_synthetic_ecg(bpm):
    assert heart_rate(synthetic_ecg(8, bpm=bpm)) == pytest.approx(bpm, rel=0.05)


def test_heart_rate_is_none_for_flat_signal():
    assert heart_rate(np.zeros(8 * LIVE_FS)) is None


def test_heart_rate_ignores_polarity():
    assert heart_rate(-synthetic_ecg(8, bpm=80)) == pytest.approx(80, rel=0.05)


def test_vitals_ok():
    v = vitals_for("p1", synthetic_ecg(8, bpm=75), last_seen=10.0, now=10.5)
    assert v.status == "ok" and v.heart_rate_bpm == pytest.approx(75, rel=0.05)


def test_vitals_warming_up_with_too_little_data():
    v = vitals_for("p1", synthetic_ecg(1), last_seen=1.0, now=1.0)
    assert v.status == "warming_up" and v.heart_rate_bpm is None


def test_vitals_no_signal_when_device_goes_quiet():
    v = vitals_for("p1", synthetic_ecg(8), last_seen=0.0, now=10.0)
    assert v.status == "no_signal" and v.heart_rate_bpm is None


def test_vitals_no_signal_for_flat_line():
    v = vitals_for("p1", np.zeros(8 * LIVE_FS), last_seen=5.0, now=5.0)
    assert v.status == "no_signal"


def irregular_ecg(seconds: float, mean_bpm: float = 80, jitter: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(int(seconds * LIVE_FS))
    t = 0.1
    while t < seconds:
        x[int(t * LIVE_FS)] = 1.0
        t += 60 / mean_bpm * (1 + jitter * rng.uniform(-1, 1))
    return np.convolve(x, np.hanning(10), mode="same")


def test_regular_rhythm_has_low_irregularity():
    _, irregularity = rhythm(synthetic_ecg(8, bpm=70))
    assert irregularity is not None and irregularity < 0.05


def test_irregular_rhythm_is_detected():
    _, irregularity = rhythm(irregular_ecg(8))
    assert irregularity is not None and irregularity > 0.15


def test_irregularity_needs_enough_beats():
    assert rhythm(synthetic_ecg(4, bpm=50)) == (pytest.approx(50, rel=0.05), None)


def test_vitals_include_irregularity():
    assert vitals_for("p1", irregular_ecg(8), last_seen=1.0, now=1.0).rr_irregularity > 0.15
