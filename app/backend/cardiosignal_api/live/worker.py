"""Analysis worker for one shard of patients.

    REDIS_URL=redis://localhost:6379/0 python -m cardiosignal_api.live.worker --shard 0

Reads samples with a consumer group (so unacknowledged messages survive a crash and are
re-read on restart), keeps each patient's rolling buffer and alert state, and publishes one
result per tick for the API to merge.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Callable

from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError

from ..config import Settings
from .analyzer import ShardAnalyzer, compute_vitals
from .redis_hub import GROUP, RESULTS_CHANNEL, decode_samples, stream_key

logger = logging.getLogger("cardiosignal_api.live.worker")
READ_BATCH = 2000
READ_BLOCK_MS = 200
RETRY_SECONDS = 1.0


class ShardWorker:
    def __init__(self, redis: Redis, shard: int, max_waveforms: int = 24, clock: Callable[[], float] = time.time):
        self.redis = redis
        self.shard = shard
        self.stream = stream_key(shard)
        self.consumer = f"worker-{shard}"      # stable name: a restarted worker re-reads its own pending messages
        self.clock = clock
        self.analyzer = ShardAnalyzer(max_waveforms=max_waveforms)
        self._cursor = "0"                     # "0" = our pending (unacknowledged) messages first, then ">" = new
        self._max_lag = 0.0

    async def setup(self) -> None:
        try:
            await self.redis.xgroup_create(self.stream, GROUP, id="$", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def consume_once(self, block_ms: int = READ_BLOCK_MS) -> int:
        """Read one batch into the analyzer and acknowledge it. Returns the number of messages."""
        response = await self.redis.xreadgroup(GROUP, self.consumer, {self.stream: self._cursor},
                                               count=READ_BATCH, block=block_ms)
        messages = response[0][1] if response else []
        if not messages:
            if self._cursor == "0":            # pending backlog drained: switch to new messages now
                self._cursor = ">"
                return await self.consume_once(block_ms)
            return 0
        now = self.clock()
        for _, fields in messages:
            try:
                patient_id, samples, sent_at = decode_samples(fields)
            except (KeyError, ValueError, UnicodeDecodeError) as exc:
                logger.warning("Skipping malformed message on %s: %s", self.stream, exc)
                continue
            self.analyzer.ingest(patient_id, samples, sent_at)
            self._max_lag = max(self._max_lag, now - sent_at)
        await self.redis.xack(self.stream, GROUP, *[message_id for message_id, _ in messages])
        return len(messages)

    async def tick_once(self, now: float) -> dict[str, Any]:
        readings = self.analyzer.collect(now)
        vitals, analysis_ms = await asyncio.to_thread(compute_vitals, readings, now, self.analyzer.fs)
        result = self.analyzer.finalize(readings, vitals, analysis_ms, now)
        result = {**result, "shard": self.shard, "lag_ms": round(self._max_lag * 1000, 1)}
        self._max_lag = 0.0
        await self.redis.publish(RESULTS_CHANNEL, json.dumps(result))
        return result

    async def run(self, tick_seconds: float = 1.0) -> None:
        await self._retrying(self.setup)
        logger.info("Worker for shard %d reading %s", self.shard, self.stream)
        await asyncio.gather(self._consume_forever(), self._tick_forever(tick_seconds))

    async def _consume_forever(self) -> None:
        while True:
            await self._retrying(self.consume_once)

    async def _tick_forever(self, tick_seconds: float) -> None:
        while True:
            started = self.clock()
            try:
                await self.tick_once(started)
            except (RedisError, OSError) as exc:
                logger.warning("Could not publish results (%s)", exc)
            await asyncio.sleep(max(0.0, tick_seconds - (self.clock() - started)))

    async def _retrying(self, operation: Callable[[], Any]) -> Any:
        while True:
            try:
                return await operation()
            except (RedisError, OSError) as exc:
                logger.warning("Redis unavailable (%s); retrying in %.0f s", exc, RETRY_SECONDS)
                await asyncio.sleep(RETRY_SECONDS)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CardioSignal live-monitor analysis worker.")
    parser.add_argument("--shard", type=int, required=True, help="which shard of patients to analyse (0-based)")
    args = parser.parse_args(argv)
    if args.shard < 0:
        parser.error("--shard must be 0 or more")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(sys.argv[1:] if argv is None else argv)
    settings = Settings()
    if args.shard >= settings.live_shards:
        logger.error("--shard %d is out of range: LIVE_SHARDS is %d", args.shard, settings.live_shards)
        return 2
    redis = Redis.from_url(os.getenv("REDIS_URL", settings.redis_url))
    worker = ShardWorker(redis, args.shard, settings.live_max_waveforms)
    try:
        asyncio.run(worker.run(settings.live_tick_seconds))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
