"""Model service: preprocessing identical to training, windowed inference, beat analysis."""
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np
import torch
from scipy.signal import resample_poly

from cardiosignal.beats import (PCG_QUALITY_THRESHOLD, detect_r_peaks, heart_rate_from_envelope,
                                heart_rate_from_peaks, periodicity)
from cardiosignal.config import ENVELOPE_WINDOW, SEGMENT_LEN, TARGET_FS
from cardiosignal.inference import load_generators
from cardiosignal.signal import envelope, fix_ecg_polarity, scale_to_unit

Direction = Literal["ecg_to_pcg", "pcg_to_ecg"]
ENVELOPE_FS = TARGET_FS / (ENVELOPE_WINDOW // 2)


@dataclass(frozen=True)
class SignalAnalysis:
    heart_rate_bpm: float | None
    beats: list[int]                 # sample indices at TARGET_FS (ECG R-peaks; empty for PCG)
    quality: float | None = None     # heart-sound rhythm clarity, PCG only
    quality_ok: bool | None = None


@dataclass(frozen=True)
class Translation:
    direction: Direction
    sample_rate: int
    input_signal: np.ndarray         # preprocessed input at TARGET_FS, in [-1, 1]
    output_signal: np.ndarray
    input_analysis: SignalAnalysis
    output_analysis: SignalAnalysis
    ecg_was_flipped: bool


def resample(x: np.ndarray, from_fs: int, to_fs: int = TARGET_FS) -> np.ndarray:
    if from_fs == to_fs:
        return np.asarray(x, dtype=np.float64).copy()
    ratio = Fraction(to_fs, from_fs)   # exact; signal_io rejects rates with unwieldy ratios
    return resample_poly(np.asarray(x, dtype=np.float64), ratio.numerator, ratio.denominator)


def window_starts(length: int, window: int = SEGMENT_LEN) -> list[int]:
    """Non-overlapping 6 s windows; the last one is aligned to the end so the tail is covered."""
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, window))
    if starts[-1] + window < length:
        starts.append(length - window)
    return starts


def analyze_ecg(ecg: np.ndarray) -> SignalAnalysis:
    peaks = detect_r_peaks(ecg)
    return SignalAnalysis(heart_rate_bpm=_round(heart_rate_from_peaks(peaks)), beats=[int(p) for p in peaks])


def analyze_pcg(pcg: np.ndarray) -> SignalAnalysis:
    env = envelope(torch.from_numpy(pcg.astype(np.float32)).view(1, 1, -1)).flatten().numpy()
    quality = periodicity(env, ENVELOPE_FS)
    return SignalAnalysis(heart_rate_bpm=_round(heart_rate_from_envelope(env, ENVELOPE_FS)), beats=[],
                          quality=round(quality, 3), quality_ok=quality >= PCG_QUALITY_THRESHOLD)


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 1)


class Translator:
    def __init__(self, model_path: str, torch_threads: int = 4):
        torch.set_num_threads(max(1, torch_threads))
        self.g_e2p, self.g_p2e = load_generators(model_path, device="cpu")

    @torch.no_grad()
    def _run(self, generator, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Translate a 2 kHz signal window by window, each window scaled like the training data.

        Returns (model_input, model_output), both trimmed to len(x), so the input shown to
        the user and analysed for heart rate is exactly what the model saw.
        """
        padded = np.pad(x, (0, max(0, SEGMENT_LEN - len(x))), mode="reflect")
        model_in = np.zeros_like(padded, dtype=np.float32)
        model_out = np.zeros_like(padded, dtype=np.float32)
        for start in window_starts(len(padded)):
            chunk = scale_to_unit(padded[start:start + SEGMENT_LEN])
            model_in[start:start + SEGMENT_LEN] = chunk
            model_out[start:start + SEGMENT_LEN] = generator(torch.from_numpy(chunk).view(1, 1, -1)).flatten().numpy()
        return model_in[: len(x)], model_out[: len(x)]

    def translate(self, signal: np.ndarray, sample_rate: int, direction: Direction) -> Translation:
        x = resample(signal, sample_rate)
        flipped = False
        if direction == "ecg_to_pcg":
            upright = fix_ecg_polarity(x)
            flipped = bool(np.sign(np.dot(upright, x)) < 0)
            x = upright
        generator = self.g_e2p if direction == "ecg_to_pcg" else self.g_p2e
        x_scaled, y = self._run(generator, x)
        if direction == "ecg_to_pcg":
            return Translation(direction, TARGET_FS, x_scaled, y, analyze_ecg(x_scaled), analyze_pcg(y), flipped)
        return Translation(direction, TARGET_FS, x_scaled, y, analyze_pcg(x_scaled), analyze_ecg(y), flipped)
