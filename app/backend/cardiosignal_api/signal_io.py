"""Parse and validate user-supplied signal files and arrays."""
import io
from fractions import Fraction
from pathlib import PurePath

import numpy as np
from scipy.io import wavfile

from cardiosignal.config import TARGET_FS

SUPPORTED_EXTENSIONS = (".csv", ".txt", ".npy", ".wav")
MIN_SAMPLE_RATE = 250
MAX_SAMPLE_RATE = 48000
MAX_RESAMPLE_FACTOR = 1000


class SignalError(ValueError):
    """Invalid user input. The message is safe to show to the user."""


def validate_sample_rate(sample_rate: int) -> None:
    if not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
        raise SignalError(f"Sample rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz.")
    # Exact polyphase resampling needs a small up/down ratio; unusual rates (e.g. 47999 Hz)
    # would otherwise be approximated and silently distort timing and heart rate.
    ratio = Fraction(TARGET_FS, sample_rate)
    if max(ratio.numerator, ratio.denominator) > MAX_RESAMPLE_FACTOR:
        raise SignalError(f"Unsupported sample rate {sample_rate} Hz. Use a standard rate such as "
                          "250, 500, 1000, 2000, 4000, 8000, 22050 or 44100 Hz.")


def validate_signal(signal: np.ndarray, sample_rate: int, min_seconds: int, max_seconds: int) -> np.ndarray:
    """Return the signal as a 1-D float64 array, or raise SignalError with a user-friendly message."""
    validate_sample_rate(sample_rate)
    try:
        x = np.asarray(signal, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise SignalError("The signal must contain numbers only.") from exc
    if x.ndim == 2 and 1 in x.shape:
        x = x.reshape(-1)
    if x.ndim != 1:
        raise SignalError("Expected a single-channel signal (one column of numbers).")
    if not np.all(np.isfinite(x)):
        raise SignalError("The signal contains missing or non-numeric values (NaN/inf).")
    seconds = len(x) / sample_rate
    if seconds < min_seconds:
        raise SignalError(f"Signal is {seconds:.1f} s long; at least {min_seconds} s is needed.")
    if seconds > max_seconds:
        raise SignalError(f"Signal is {seconds:.1f} s long; the maximum is {max_seconds} s.")
    if np.std(x) < 1e-9:
        raise SignalError("The signal is flat (no variation).")
    return x


def parse_signal_file(filename: str, content: bytes, sample_rate: int | None) -> tuple[np.ndarray, int]:
    """Read a single-channel signal from CSV/TXT/NPY/WAV bytes. Returns (samples, sample_rate)."""
    extension = PurePath(filename or "").suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise SignalError(f"Unsupported file type '{extension or 'none'}'. Use one of: {', '.join(SUPPORTED_EXTENSIONS)}.")
    if extension == ".wav":
        signal, rate = _read_wav(content)
        validate_sample_rate(rate)
        return signal, rate
    if sample_rate is None:
        raise SignalError("Please provide the sample rate (Hz) for CSV, TXT and NPY files.")
    reader = _read_npy if extension == ".npy" else _read_text
    return reader(content), sample_rate


def _read_wav(content: bytes) -> tuple[np.ndarray, int]:
    try:
        rate, data = wavfile.read(io.BytesIO(content))
        data = np.asarray(data, dtype=np.float64)
    except (ValueError, EOFError, TypeError, OSError, IndexError, KeyError) as exc:
        raise SignalError("Could not read the WAV file.") from exc
    if data.ndim == 2:
        data = data[:, 0]   # first channel
    return data, int(rate)


def _read_npy(content: bytes) -> np.ndarray:
    try:
        return np.load(io.BytesIO(content), allow_pickle=False)
    except (ValueError, OSError, EOFError) as exc:
        raise SignalError("Could not read the NPY file (pickled objects are not accepted).") from exc


def _read_text(content: bytes) -> np.ndarray:
    """First numeric column of a CSV/TXT file; a non-numeric header row is skipped."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SignalError("The file is not valid UTF-8 text.") from exc
    values = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        cells = [c for c in line.replace(";", ",").replace("\t", ",").replace(" ", ",").split(",") if c]
        if not cells:
            continue
        try:
            values.append(float(cells[0]))
        except ValueError:
            if values:
                raise SignalError(f"Non-numeric value on line {line_number}: {cells[0]!r}.") from None
            # header row before any data: skip it
    if not values:
        raise SignalError("No numeric values found in the file.")
    return np.array(values)
