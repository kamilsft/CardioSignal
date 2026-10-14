"""Simulated bedside devices: replay real ECG recordings at real-time speed over WebSockets.

    LIVE_INGEST_TOKEN=... python -m cardiosignal_api.live.simulator --patients 200

Each patient loops one recording (cut at heartbeats so the loop is seamless), slowed
down or sped up a little so heart rates differ, with a bit of sensor noise.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from fractions import Fraction

import numpy as np
import websockets
from scipy.signal import resample_poly

from cardiosignal.beats import detect_r_peaks
from cardiosignal.config import TARGET_FS
from cardiosignal.signal import fix_ecg_polarity, scale_to_unit

from ..config import Settings
from ..samples import SampleStore
from .monitor import LIVE_FS

logger = logging.getLogger("cardiosignal_api.live.simulator")
STRETCH_RANGE = (0.75, 1.35)     # x0.75 speed .. x1.35: a 75 bpm recording plays at 56-100 bpm
SENSOR_NOISE = 0.02
MAX_PATIENTS = 5000
MAX_BACKOFF_SECONDS = 30
FATAL_CLOSE_CODES = {1003, 1008}   # bad data / bad token or id: retrying cannot help


def build_loop(ecg: np.ndarray, source_fs: int, stretch: float, out_fs: int = LIVE_FS) -> np.ndarray:
    """One recording as a seamless loop at out_fs; stretch > 1 plays it slower (lower heart rate)."""
    upright = fix_ecg_polarity(np.asarray(ecg, dtype=np.float64), source_fs)
    beats = detect_r_peaks(upright, source_fs)
    whole_beats = upright[beats[0]:beats[-1]] if len(beats) >= 2 else upright
    ratio = Fraction(out_fs, source_fs) * Fraction(stretch).limit_denominator(100)
    resampled = resample_poly(whole_beats - np.median(whole_beats), ratio.numerator, ratio.denominator)
    return scale_to_unit(resampled)


class SimulatedPatient:
    def __init__(self, patient_id: str, loop: np.ndarray, noise: float, seed: int, start: int = 0):
        self.patient_id = patient_id
        self._loop = np.asarray(loop, dtype=np.float32)
        self._noise = noise
        self._rng = np.random.default_rng(seed)
        self._position = start % len(self._loop)

    def next_chunk(self, n: int) -> np.ndarray:
        indices = (self._position + np.arange(n)) % len(self._loop)
        self._position = (self._position + n) % len(self._loop)
        chunk = self._loop[indices]
        if self._noise > 0:
            chunk = chunk + self._noise * self._rng.standard_normal(n).astype(np.float32)
        return chunk


def make_patients(store: SampleStore, count: int, seed: int) -> list[SimulatedPatient]:
    recordings = store.list()
    if not recordings:
        raise ValueError("No recordings to replay: check SAMPLES_PATH.")
    rng = np.random.default_rng(seed)
    patients = []
    for i in range(count):
        loop = build_loop(recordings[i % len(recordings)].ecg, TARGET_FS, float(rng.uniform(*STRETCH_RANGE)))
        patients.append(SimulatedPatient(f"patient-{i + 1:04d}", loop, SENSOR_NOISE,
                                         seed=int(rng.integers(2 ** 31)), start=int(rng.integers(len(loop)))))
    return patients


class FatalDeviceError(Exception):
    """The server refused this device (bad token, id or data): retrying will not help."""


async def _send_forever(ws, patient: SimulatedPatient, chunk_seconds: float) -> None:
    loop = asyncio.get_running_loop()
    chunk = round(LIVE_FS * chunk_seconds)
    next_send = loop.time()
    while True:
        await ws.send(json.dumps({"samples": np.round(patient.next_chunk(chunk), 4).tolist()}))
        next_send += chunk_seconds
        delay = next_send - loop.time()
        if delay < -1.0:              # far behind (machine overloaded): skip ahead rather than burst
            next_send = loop.time()
        await asyncio.sleep(max(0.0, delay))


async def stream_patient(patient: SimulatedPatient, url: str, token: str, chunk_seconds: float) -> None:
    """Stream one patient forever, reconnecting with exponential backoff after network errors."""
    endpoint = f"{url.rstrip('/')}/api/live/ingest/{patient.patient_id}"
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(endpoint, additional_headers={"X-Device-Token": token}) as ws:
                backoff = 1.0
                await _send_forever(ws, patient, chunk_seconds)
        except websockets.exceptions.ConnectionClosed as exc:
            if exc.rcvd is not None and exc.rcvd.code in FATAL_CLOSE_CODES:
                raise FatalDeviceError(f"{patient.patient_id}: {exc.rcvd.reason}") from exc
            logger.warning("%s: connection closed (%s), retrying in %.0f s", patient.patient_id, exc, backoff)
        except websockets.exceptions.InvalidStatus as exc:    # e.g. a proxy answering 502 while the backend restarts
            logger.warning("%s: HTTP %d from server, retrying in %.0f s",
                           patient.patient_id, exc.response.status_code, backoff)
        except OSError as exc:
            logger.warning("%s: cannot reach %s (%s), retrying in %.0f s", patient.patient_id, url, exc, backoff)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)


async def run(patients: list[SimulatedPatient], url: str, token: str, chunk_seconds: float) -> None:
    async def staggered(i: int, patient: SimulatedPatient) -> None:
        await asyncio.sleep(i * 0.005)          # avoid 1000 handshakes in the same millisecond
        await stream_patient(patient, url, token, chunk_seconds)

    logger.info("Streaming %d patients to %s", len(patients), url)
    async with asyncio.TaskGroup() as group:
        for i, patient in enumerate(patients):
            group.create_task(staggered(i, patient))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate ECG devices streaming to the CardioSignal live monitor.")
    parser.add_argument("--url", default=os.getenv("LIVE_URL", "ws://127.0.0.1:8000"), help="backend WebSocket base URL")
    parser.add_argument("--patients", type=int, default=50)
    parser.add_argument("--chunk-ms", type=int, default=250, help="how often each device sends data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not 1 <= args.patients <= MAX_PATIENTS:
        parser.error(f"--patients must be between 1 and {MAX_PATIENTS}")
    if not 50 <= args.chunk_ms <= 1000:
        parser.error("--chunk-ms must be between 50 and 1000")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(sys.argv[1:] if argv is None else argv)
    token = os.getenv("LIVE_INGEST_TOKEN", "")
    if not token:
        logger.error("Set LIVE_INGEST_TOKEN to the same value as the backend.")
        return 2
    patients = make_patients(SampleStore(Settings().samples_path), args.patients, args.seed)
    try:
        asyncio.run(run(patients, args.url, token, args.chunk_ms / 1000))
    except KeyboardInterrupt:
        return 0
    except ExceptionGroup as group:
        fatal = group.subgroup(FatalDeviceError)
        if fatal is None:
            raise
        logger.error("%s", fatal.exceptions[0])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
