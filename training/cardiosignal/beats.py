"""Heartbeat-level tools: QRS detection, ECG polarity, beat matching, heart rate.

QRS complexes are found by their *steepness* (Pan-Tompkins style: band-pass,
derivative, square, moving-window integration), not by height. Tall, wide
T-waves are slow, so they no longer get mistaken for heartbeats.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from .config import TARGET_FS

MIN_RR_SECONDS = 0.3        # fastest plausible beat-to-beat interval (200 bpm)
QRS_BAND_HZ = (5.0, 15.0)
INTEGRATION_SECONDS = 0.12
ENERGY_THRESHOLD = 0.3      # fraction of the segment's robust peak QRS energy
REFINE_SECONDS = 0.05       # search +-50 ms around the energy peak for the R-peak
T_WAVE_WINDOW_SECONDS = 0.36  # a weak candidate this close to a beat is a T-wave
T_WAVE_ENERGY_RATIO = 0.5
EDGE_SECONDS = 0.25           # ignored when setting the detection threshold
HR_RANGE = (35, 200)
# Heart-sound quality cut-off, fixed on the validation set (no model output used): above it,
# the heart rate read from the sound alone matches the ECG in 97% of segments; below it, 50%.
PCG_QUALITY_THRESHOLD = 0.3


def _qrs_energy(x: np.ndarray, fs: int) -> np.ndarray:
    b, a = butter(2, QRS_BAND_HZ, btype="band", fs=fs)
    slope = np.gradient(filtfilt(b, a, x))
    window = max(1, int(INTEGRATION_SECONDS * fs))
    return np.convolve(slope ** 2, np.ones(window) / window, mode="same")


def detect_qrs(ecg: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """Approximate sample positions of QRS complexes, independent of polarity."""
    x = np.asarray(ecg, dtype=np.float64)
    if x.size < fs // 2 or np.allclose(x, x[0]):
        return np.array([], dtype=int)
    energy = _qrs_energy(x - np.median(x), fs)
    # A beat cut in half at the segment edge makes a filter transient several times
    # stronger than real beats; measure the threshold on the interior only.
    edge = int(EDGE_SECONDS * fs)
    interior = energy[edge:-edge] if len(energy) > 4 * edge else energy
    ceiling = np.percentile(interior, 99.5)
    if ceiling <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(energy, height=ENERGY_THRESHOLD * ceiling, distance=int(MIN_RR_SECONDS * fs))
    return _reject_t_waves(peaks, energy, fs)


def _reject_t_waves(peaks: np.ndarray, energy: np.ndarray, fs: int) -> np.ndarray:
    """Pan-Tompkins rule: a candidate soon after a beat, with much less slope energy, is a T-wave.

    Candidates are compared against the stronger neighbour on either side, so a
    T-wave is rejected even when it comes first in the segment.
    """
    window = int(T_WAVE_WINDOW_SECONDS * fs)
    kept = []
    for i, p in enumerate(peaks):
        neighbours = [q for q in peaks[max(0, i - 1): i + 2] if q != p and abs(int(q) - int(p)) < window]
        if any(energy[p] < T_WAVE_ENERGY_RATIO * energy[q] for q in neighbours):
            continue
        kept.append(p)
    return np.array(kept, dtype=int)


def _qrs_extremes(x: np.ndarray, qrs: np.ndarray, fs: int) -> np.ndarray:
    """Signed value of the largest deflection near each QRS."""
    half = int(REFINE_SECONDS * fs)
    centered = x - np.median(x)
    values = []
    for q in qrs:
        window = centered[max(0, q - half): q + half + 1]
        values.append(window[np.argmax(np.abs(window))])
    return np.array(values)


def ecg_polarity(ecg: np.ndarray, fs: int = TARGET_FS) -> int:
    """+1 if the QRS complexes point up, -1 if the ECG is inverted."""
    x = np.asarray(ecg, dtype=np.float64)
    qrs = detect_qrs(x, fs)
    if len(qrs) == 0:
        centered = x - np.mean(x)
        return -1 if np.mean(centered ** 3) < 0 else 1
    return -1 if np.sum(_qrs_extremes(x, qrs, fs)) < 0 else 1


def detect_r_peaks(ecg: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """R-peak sample indices of an upright ECG: QRS by steepness, refined to the local maximum."""
    x = np.asarray(ecg, dtype=np.float64)
    half = int(REFINE_SECONDS * fs)
    peaks = []
    for q in detect_qrs(x, fs):
        lo = max(0, q - half)
        peaks.append(lo + int(np.argmax(x[lo: q + half + 1])))
    return np.unique(np.array(peaks, dtype=int))


def match_beats(reference: np.ndarray, predicted: np.ndarray, tolerance: int) -> int:
    """Number of predicted beats within `tolerance` samples of a distinct reference beat (greedy, one-to-one)."""
    used = np.zeros(len(reference), dtype=bool)
    hits = 0
    for p in np.sort(predicted):
        if len(reference) == 0:
            break
        distance = np.abs(reference - p).astype(float)
        distance[used] = np.inf
        best = int(np.argmin(distance))
        if distance[best] <= tolerance:
            used[best] = True
            hits += 1
    return hits


def heart_rate_from_peaks(peaks: np.ndarray, fs: int = TARGET_FS) -> float | None:
    if len(peaks) < 2:
        return None
    bpm = 60.0 * fs / float(np.median(np.diff(peaks)))
    return bpm if HR_RANGE[0] <= bpm <= HR_RANGE[1] else None


def _envelope_autocorrelation(envelope: np.ndarray, envelope_fs: float):
    e = np.asarray(envelope, dtype=np.float64)
    e = e - e.mean()
    if np.allclose(e, 0):
        return None, 0, 0
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    ac = ac / ac[0]
    lo = int(envelope_fs * 60.0 / HR_RANGE[1])
    hi = min(len(ac) - 1, int(envelope_fs * 60.0 / HR_RANGE[0]))
    return ac, lo, hi


def heart_rate_from_envelope(envelope: np.ndarray, envelope_fs: float) -> float | None:
    """Dominant beat period of a loudness envelope via autocorrelation (works for heart sounds)."""
    ac, lo, hi = _envelope_autocorrelation(envelope, envelope_fs)
    if ac is None or hi <= lo:
        return None
    lag = lo + int(np.argmax(ac[lo:hi + 1]))
    return 60.0 * envelope_fs / lag


def periodicity(envelope: np.ndarray, envelope_fs: float) -> float:
    """Signal-quality score in [0, 1]: how rhythmic a heart-sound envelope is.

    Normalized autocorrelation at the dominant heartbeat lag. A clean recording
    repeats every beat (high); a silent or noise-dominated one does not (low).
    Uses only the input recording, so the app can compute it too.
    """
    ac, lo, hi = _envelope_autocorrelation(envelope, envelope_fs)
    if ac is None or hi <= lo:
        return 0.0
    return float(max(0.0, np.max(ac[lo:hi + 1])))


def peak_weight_mask(length: int, peaks: np.ndarray, half_width: int, boost: float) -> np.ndarray:
    """Per-sample loss weights: 1 everywhere, 1 + boost within +-half_width of each peak."""
    w = np.ones(length, dtype=np.float32)
    for p in peaks:
        if p >= 0:
            w[max(0, p - half_width): p + half_width + 1] = 1.0 + boost
    return w
