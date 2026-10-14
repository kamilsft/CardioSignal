"""One patient's live ECG: a rolling buffer plus heart-rate analysis of the latest window."""
from dataclasses import dataclass
from typing import Literal

import numpy as np

from cardiosignal.beats import detect_qrs

LIVE_FS = 250                 # Hz; typical for wearable ECG and 8x less traffic than the model's 2 kHz
WINDOW_SECONDS = 8            # heart rate is the median beat interval over this window
MIN_ANALYSIS_SECONDS = 3      # fewer than ~3 beats gives an unstable heart rate
NO_SIGNAL_SECONDS = 5         # a device silent this long is treated as disconnected
# Wider than the training code's 35-200 bpm: a dangerously slow heart must read as slow, not as "no signal".
# The detector's 300 ms refractory period caps the top end at 200 bpm anyway.
LIVE_HR_RANGE = (20, 250)
MIN_BEATS_FOR_RHYTHM = 5      # irregularity needs at least 3 successive RR differences

Status = Literal["warming_up", "ok", "no_signal"]


@dataclass(frozen=True)
class Vitals:
    patient_id: str
    heart_rate_bpm: float | None
    status: Status
    rr_irregularity: float | None = None   # mean |successive RR difference| / mean RR; ~0.02-0.06 in sinus rhythm


class PatientStream:
    """Fixed-size ring buffer of the most recent ECG samples.

    Mutated in place on purpose: devices push every few hundred milliseconds, and
    copying a growing array on each push would cost O(n) per message.
    """

    def __init__(self, patient_id: str, fs: int = LIVE_FS, window_seconds: float = WINDOW_SECONDS):
        self.patient_id = patient_id
        self.fs = fs
        self._buffer = np.zeros(int(fs * window_seconds), dtype=np.float32)
        self._written = 0          # total samples ever pushed
        self._sent = 0             # samples already handed out by take_new()
        self.last_seen: float | None = None

    @property
    def capacity(self) -> int:
        return len(self._buffer)

    def push(self, samples: np.ndarray, now: float) -> None:
        x = np.asarray(samples, dtype=np.float32)[-self.capacity:]
        start = (self._written + len(samples) - len(x)) % self.capacity   # skip what an oversized push overwrote
        first = min(len(x), self.capacity - start)
        self._buffer[start:start + first] = x[:first]
        self._buffer[: len(x) - first] = x[first:]
        self._written += len(samples)
        self.last_seen = now

    def window(self) -> np.ndarray:
        """Copy of the buffered samples, oldest first."""
        return self._latest(min(self._written, self.capacity))

    def take_new(self, multiple: int = 1) -> np.ndarray:
        """Samples pushed since the previous call (at most one buffer's worth).

        Returns a multiple of `multiple` samples; the remainder waits for the next call, so a
        consumer that decimates in pairs never drops or misaligns a sample.
        """
        pending = self._written - self._sent
        held_back = pending % multiple
        count = min(pending - held_back, self.capacity - held_back)
        count -= count % multiple
        self._sent = self._written - held_back
        return self._latest(count, skip_newest=held_back)

    def _latest(self, count: int, skip_newest: int = 0) -> np.ndarray:
        if count <= 0:
            return np.zeros(0, dtype=np.float32)
        ordered = np.roll(self._buffer, -(self._written % self.capacity))   # oldest .. newest
        end = len(ordered) - skip_newest
        return ordered[end - count:end].copy()


def heart_rate(ecg: np.ndarray, fs: int = LIVE_FS) -> float | None:
    """Beats per minute from QRS steepness, so an upside-down lead still works."""
    return rhythm(ecg, fs)[0]


def rhythm(ecg: np.ndarray, fs: int = LIVE_FS) -> tuple[float | None, float | None]:
    """(heart rate in bpm, RR irregularity) of one window."""
    beats = detect_qrs(ecg, fs)
    if len(beats) < 2:
        return None, None
    rr = np.diff(beats).astype(np.float64)
    bpm = 60.0 * fs / float(np.median(rr))
    if not LIVE_HR_RANGE[0] <= bpm <= LIVE_HR_RANGE[1]:
        return None, None
    irregularity = None
    if len(beats) >= MIN_BEATS_FOR_RHYTHM:
        irregularity = round(float(np.mean(np.abs(np.diff(rr))) / np.mean(rr)), 3)
    return round(bpm, 1), irregularity


def vitals_for(patient_id: str, window: np.ndarray, last_seen: float | None, now: float,
               fs: int = LIVE_FS) -> Vitals:
    if last_seen is None or now - last_seen > NO_SIGNAL_SECONDS:
        return Vitals(patient_id, None, "no_signal")
    if len(window) < MIN_ANALYSIS_SECONDS * fs:
        return Vitals(patient_id, None, "warming_up")
    bpm, irregularity = rhythm(window, fs)
    return Vitals(patient_id, bpm, "ok" if bpm is not None else "no_signal", irregularity)
