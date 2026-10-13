import numpy as np

from cardiosignal.beats import (detect_qrs, detect_r_peaks, ecg_polarity, heart_rate_from_envelope,
                                heart_rate_from_peaks, match_beats, peak_weight_mask, periodicity)

FS = 2000


def _ecg(bpm=75, seconds=6):
    x = np.zeros(FS * seconds)
    period = int(FS * 60 / bpm)
    x[period // 2::period] = 1.0
    x += 0.3 * np.roll(x, 300)   # smaller T-wave bumps, must not count as beats
    return np.convolve(x, np.hanning(40), mode="same")


def test_detects_r_peaks_not_t_waves():
    peaks = detect_r_peaks(_ecg(75))
    assert len(peaks) == 7  # 6 s at 75 bpm -> 7 or 8 beats, first at 0.4 s
    assert heart_rate_from_peaks(peaks) == np.float64(60 * FS / 1600)


def _t_dominant_inverted_ecg(bpm=70, seconds=6):
    """Like recording 0045: sharp *downward* QRS, then a tall, wide, upward T-wave."""
    period = int(FS * 60 / bpm)
    qrs, t_wave = np.zeros(FS * seconds), np.zeros(FS * seconds)
    qrs[period // 2::period] = 1.0
    t_wave[period // 2 + 500::period] = 1.0
    return (-0.9 * np.convolve(qrs, np.hanning(50), mode="same")
            + 1.0 * np.convolve(t_wave, np.hanning(400), mode="same"))


def test_polarity_uses_qrs_not_tall_t_wave():
    x = _t_dominant_inverted_ecg()
    assert x.max() > abs(x.min())            # the T-wave is the tallest thing in the signal
    assert ecg_polarity(x) == -1             # ...but the QRS points down, so the ECG is inverted


def test_r_peaks_land_on_qrs_not_t_wave():
    x = -_t_dominant_inverted_ecg(bpm=70)
    period = int(FS * 60 / 70)
    true_qrs = np.arange(period // 2, len(x), period)
    peaks = detect_r_peaks(x)
    assert match_beats(true_qrs, peaks, tolerance=40) == len(true_qrs) == len(peaks)


def test_beat_cut_at_segment_edge_does_not_hide_other_beats():
    x = np.zeros(FS * 6)
    x[::1666] = 1.0                                    # first beat sits exactly on sample 0
    x = np.convolve(x, np.hanning(40), mode="same")
    assert len(detect_r_peaks(x)) >= 7


def test_detectors_handle_flat_signal():
    assert len(detect_qrs(np.zeros(FS * 6))) == 0
    assert ecg_polarity(np.zeros(FS * 6)) == 1


def test_periodicity_separates_rhythm_from_noise():
    fs_env = 40.0
    t = np.arange(int(6 * fs_env)) / fs_env
    rhythmic = np.maximum(0, np.sin(2 * np.pi * 1.2 * t)) ** 4
    noise = np.abs(np.random.default_rng(0).standard_normal(len(t)))
    assert periodicity(rhythmic, fs_env) > 0.6 > periodicity(noise, fs_env)


def test_peak_weight_mask():
    w = peak_weight_mask(20, np.array([5, -1]), half_width=2, boost=3.0)
    assert w[3:8].tolist() == [4.0] * 5 and w[0] == 1.0 and w[10] == 1.0


def test_match_beats_tolerance_and_one_to_one():
    ref = np.array([100, 1000, 2000])
    assert match_beats(ref, np.array([110, 1090, 2500]), tolerance=100) == 2
    assert match_beats(ref, np.array([100, 101]), tolerance=100) == 1  # one reference beat matched once
    assert match_beats(np.array([]), np.array([5]), tolerance=10) == 0


def test_heart_rate_from_envelope():
    fs_env = 40.0
    t = np.arange(int(6 * fs_env)) / fs_env
    env = np.maximum(0, np.sin(2 * np.pi * (72 / 60) * t)) ** 4
    assert abs(heart_rate_from_envelope(env, fs_env) - 72) < 4
