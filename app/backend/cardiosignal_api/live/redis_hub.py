"""Scale-out hub: the API only forwards samples to Redis; worker processes do the analysis.

    device --WS--> API --XADD--> live:ingest:<shard>  --XREADGROUP--> worker <shard>
    dashboard <--WS-- API <--merge-- live:results (pub/sub) <--PUBLISH-- every worker, once per second

Patients are sharded by a stable hash of their id, so each patient's rolling buffer and alert
state live in exactly one worker. Streams are capped (MAXLEN ~) so a dead worker cannot fill memory.
"""
import asyncio
import json
import logging
import zlib
from typing import Any

import numpy as np
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .analyzer import merge_results
from .hub import LiveHub, LiveUnavailable

logger = logging.getLogger("cardiosignal_api.live")
GROUP = "analyzers"
RESULTS_CHANNEL = "live:results"
SAMPLE_DTYPE = "<f4"
STREAM_MAXLEN = 200_000          # ~8 minutes of backlog for 100 patients at 4 messages/s per shard
RESULT_MAX_AGE_TICKS = 3         # a worker silent for 3 ticks is considered down


def stream_key(shard: int) -> str:
    return f"live:ingest:{shard}"


def shard_of(patient_id: str, shards: int) -> int:
    """Stable across processes and restarts (unlike Python's salted hash())."""
    return zlib.crc32(patient_id.encode()) % shards


def encode_samples(patient_id: str, samples: np.ndarray, now: float) -> dict[str, Any]:
    return {"p": patient_id, "t": f"{now:.3f}", "s": np.asarray(samples, dtype=SAMPLE_DTYPE).tobytes()}


def decode_samples(fields: dict[bytes, bytes]) -> tuple[str, np.ndarray, float]:
    return fields[b"p"].decode(), np.frombuffer(fields[b"s"], dtype=SAMPLE_DTYPE), float(fields[b"t"])


class RedisHub(LiveHub):
    def __init__(self, redis: Redis, shards: int, *args: Any, stream_maxlen: int = STREAM_MAXLEN, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if shards < 1:
            raise ValueError("shards must be at least 1")
        self.redis = redis
        self.shards = shards
        self.stream_maxlen = stream_maxlen
        self._results: dict[int, tuple[float, dict[str, Any]]] = {}    # shard -> (received at, result)

    async def ingest(self, patient_id: str, samples: np.ndarray, now: float) -> None:
        try:
            await self.redis.xadd(stream_key(shard_of(patient_id, self.shards)),
                                  encode_samples(patient_id, samples, now),
                                  maxlen=self.stream_maxlen, approximate=True)
        except (RedisError, OSError) as exc:
            raise LiveUnavailable("The analysis pipeline is unavailable.") from exc

    def receive_result(self, result: dict[str, Any], received_at: float) -> None:
        self._results[int(result["shard"])] = (received_at, result)

    def snapshot(self, now: float, interval: float) -> dict[str, Any]:
        fresh = [r for at, r in self._results.values() if now - at <= RESULT_MAX_AGE_TICKS * interval]
        return merge_results(fresh, self.max_waveforms, now, {
            "devices_connected": self.devices_connected,
            "workers": len(fresh),
            "pipeline_lag_ms": max((r.get("lag_ms", 0.0) for r in fresh), default=0.0),
        })

    async def run(self, interval: float = 1.0) -> None:
        pubsub = self.redis.pubsub()
        reader = asyncio.create_task(self._read_results(pubsub))

        async def step(now: float) -> dict[str, Any]:
            return self.snapshot(now, interval)
        try:
            await self._every(interval, step)
        finally:
            reader.cancel()
            await pubsub.aclose()

    async def _read_results(self, pubsub) -> None:
        subscribed = False
        while True:
            try:
                if not subscribed:
                    await pubsub.subscribe(RESULTS_CHANNEL)
                    subscribed = True
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is not None:
                    self.receive_result(json.loads(message["data"]), self.clock())
            except (RedisError, OSError) as exc:
                logger.warning("Lost the results channel (%s); retrying in 1 s", exc)
                subscribed = False
                await asyncio.sleep(1)
            except (ValueError, KeyError) as exc:
                logger.warning("Ignoring malformed worker result: %s", exc)

    async def close(self) -> None:
        await self.redis.aclose()
