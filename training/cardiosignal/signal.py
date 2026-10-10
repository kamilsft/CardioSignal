"""Signal preprocessing shared by data preparation, training and the app."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import decimate

from .beats import ecg_polarity
from .config import DECIMATION, ENVELOPE_WINDOW, TARGET_FS


def downsample(x: np.ndarray, factor: int = DECIMATION) -> np.ndarray:
    """Anti-aliased decimation (8 kHz -> 2 kHz by default)."""
    return decimate(np.asarray(x, dtype=np.float64), factor, ftype="fir", zero_phase=True)


def scale_to_unit(x: np.ndarray) -> np.ndarray:
    """Remove the mean and divide by peak magnitude, so the signal lies in [-1, 1].

    The generator ends in tanh, so inputs and targets must live in the same range.
    Returns a new array; the input is not modified.
    """
    centered = np.asarray(x, dtype=np.float64) - np.mean(x)
    peak = np.max(np.abs(centered))
    if peak < 1e-8:
        return np.zeros_like(centered, dtype=np.float32)
    return (centered / peak).astype(np.float32)


def fix_ecg_polarity(x: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """Return the ECG flipped if needed so the QRS complexes point up.

    Lead placement can invert an ECG. A heart sound carries no information about
    that, so PCG->ECG cannot learn it; we standardize polarity instead.
    v3: decided by the sharp QRS deflections (steepness-based detection), not by
    skewness, which tall wide T-waves could fool.
    """
    x = np.asarray(x, dtype=np.float64)
    return -x if ecg_polarity(x, fs) < 0 else x.copy()


def envelope(x: torch.Tensor, window: int = ENVELOPE_WINDOW) -> torch.Tensor:
    """Smoothed loudness envelope of |x|. x: [B, 1, T] -> [B, 1, T // window]."""
    return F.avg_pool1d(x.abs(), window, stride=window // 2)
