"""Small in-memory sliding-window rate limiter (per client IP).

Good enough for a single-instance demo. Behind several replicas, use a shared
store such as Redis instead. Memory stays bounded: idle clients are evicted and
the number of tracked clients is capped.
"""
import threading
import time
from collections import deque

SWEEP_EVERY = 1000          # calls between sweeps of idle clients
DEFAULT_MAX_CLIENTS = 50_000


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float = 60.0, clock=time.monotonic,
                 max_clients: int = DEFAULT_MAX_CLIENTS):
        self.max_requests = max_requests
        self.window = window_seconds
        self.max_clients = max_clients
        self._clock = clock
        self._hits: dict[str, deque] = {}
        self._calls = 0
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            self._calls += 1
            if self._calls % SWEEP_EVERY == 0 or len(self._hits) >= self.max_clients:
                self._sweep(now)
            hits = self._hits.get(key)
            if hits is None:
                if len(self._hits) >= self.max_clients:
                    return False   # still full after sweeping: refuse rather than grow
                hits = self._hits[key] = deque()
            self._prune(hits, now)
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            return True

    def tracked_clients(self) -> int:
        with self._lock:
            return len(self._hits)

    def _prune(self, hits: deque, now: float) -> None:
        while hits and now - hits[0] >= self.window:
            hits.popleft()

    def _sweep(self, now: float) -> None:
        for key in list(self._hits):
            self._prune(self._hits[key], now)
            if not self._hits[key]:
                del self._hits[key]
