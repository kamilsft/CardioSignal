"""Live hubs: devices push samples in, dashboards receive one snapshot per tick.

MemoryHub analyses in the API process (simple; one machine). RedisHub (redis_hub.py) only
forwards samples to Redis and merges results from separate worker processes (scales out).
Both share device bookkeeping and dashboard fan-out from LiveHub.
"""
import asyncio
import logging
import time
from typing import Any, Callable

import numpy as np

from .analyzer import ShardAnalyzer, compute_vitals, merge_results
from .monitor import LIVE_FS

logger = logging.getLogger("cardiosignal_api.live")


class LiveUnavailable(Exception):
    """The analysis pipeline cannot accept data right now (e.g. Redis is down)."""


class LiveHub:
    SUBSCRIBER_QUEUE = 2      # a slow dashboard skips old snapshots instead of falling behind

    def __init__(self, max_patients: int = 2000, max_waveforms: int = 24, fs: int = LIVE_FS,
                 clock: Callable[[], float] = time.time):
        self.max_patients = max_patients
        self.max_waveforms = max_waveforms
        self.fs = fs
        self.clock = clock               # wall clock: alert times are shown to people and shared across processes
        self.latest: dict[str, Any] | None = None
        self._connected: set[str] = set()
        self._subscribers: set[asyncio.Queue] = set()

    # --- devices -------------------------------------------------------------------------------
    def connect(self, patient_id: str) -> bool:
        """Register a device. Refuses a second connection for the same patient, or one over capacity."""
        if patient_id in self._connected or len(self._connected) >= self.max_patients:
            return False
        self._connected.add(patient_id)
        return True

    def disconnect(self, patient_id: str) -> None:
        self._connected.discard(patient_id)

    @property
    def devices_connected(self) -> int:
        return len(self._connected)

    async def ingest(self, patient_id: str, samples: np.ndarray, now: float) -> None:
        raise NotImplementedError

    async def run(self, interval: float = 1.0) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        """Release external resources (overridden by RedisHub)."""

    # --- dashboards ----------------------------------------------------------------------------
    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.SUBSCRIBER_QUEUE)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def publish(self, message: dict[str, Any]) -> None:
        for queue in self._subscribers:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(message)

    async def _every(self, interval: float, step: Callable[[float], Any]) -> None:
        while True:
            started = self.clock()
            try:
                snapshot = await step(started)
                if snapshot is not None:
                    self.latest = snapshot
                    self.publish(snapshot)
            except Exception:
                logger.exception("Live monitor tick failed")
            await asyncio.sleep(max(0.0, interval - (self.clock() - started)))


class MemoryHub(LiveHub):
    """Everything in one process: good for development, tests and small deployments."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.analyzer = ShardAnalyzer(self.fs, self.max_waveforms)

    def connect(self, patient_id: str) -> bool:
        if not super().connect(patient_id):
            return False
        self.analyzer.add(patient_id)        # visible as "connecting" before its first sample
        return True

    async def ingest(self, patient_id: str, samples: np.ndarray, now: float) -> None:
        if patient_id in self._connected:
            self.analyzer.ingest(patient_id, samples, now)

    def tick(self, now: float) -> dict[str, Any]:
        """Synchronous snapshot (tests); run() does the same with QRS detection off the event loop."""
        return self._snapshot(self.analyzer.tick(now, frozenset(self._connected)), now)

    async def run(self, interval: float = 1.0) -> None:
        async def step(now: float) -> dict[str, Any]:
            readings = self.analyzer.collect(now, frozenset(self._connected))   # copies: no race with ingest
            vitals, analysis_ms = await asyncio.to_thread(compute_vitals, readings, now, self.fs)
            return self._snapshot(self.analyzer.finalize(readings, vitals, analysis_ms, now), now)
        await self._every(interval, step)

    def _snapshot(self, result: dict[str, Any], now: float) -> dict[str, Any]:
        return merge_results([result], self.max_waveforms, now, {
            "devices_connected": self.devices_connected, "workers": 1, "pipeline_lag_ms": 0.0})
