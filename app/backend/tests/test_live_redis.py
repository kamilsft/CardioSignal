"""Redis mode: API hub -> Redis streams -> shard workers -> results channel -> API hub (fakeredis)."""
import asyncio
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import fakeredis
import numpy as np
import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError
from starlette.websockets import WebSocketDisconnect

from cardiosignal_api.config import Settings
from cardiosignal_api.live import routes
from cardiosignal_api.live.hub import LiveUnavailable
from cardiosignal_api.live.monitor import LIVE_FS
from cardiosignal_api.live.redis_hub import (GROUP, RESULTS_CHANNEL, RedisHub, decode_samples, shard_of,
                                             stream_key)
from cardiosignal_api.live.worker import ShardWorker, main as worker_main, parse_args
from cardiosignal_api.main import create_app
from test_live_monitor import synthetic_ecg

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples.npz"


def fake_redis(server: fakeredis.FakeServer | None = None) -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(server=server or fakeredis.FakeServer())


def test_shard_of_is_stable_and_spreads_patients():
    assert shard_of("patient-0001", 4) == shard_of("patient-0001", 4)
    counts = Counter(shard_of(f"patient-{i:04d}", 4) for i in range(1000))
    assert set(counts) == {0, 1, 2, 3} and min(counts.values()) > 200


def test_hub_ingest_writes_to_the_patients_shard_stream():
    async def scenario():
        redis = fake_redis()
        hub = RedisHub(redis, shards=3)
        await hub.ingest("bed-7", np.array([0.5, -0.25], dtype=np.float32), now=12.5)
        [(_, fields)] = await redis.xrange(stream_key(shard_of("bed-7", 3)))
        return decode_samples(fields)
    pid, samples, sent_at = asyncio.run(scenario())
    assert pid == "bed-7" and samples.tolist() == [0.5, -0.25] and sent_at == 12.5


def test_hub_ingest_reports_unavailable_when_redis_is_down():
    class DownRedis:
        async def xadd(self, *args, **kwargs):
            raise RedisConnectionError("connection refused")
    with pytest.raises(LiveUnavailable):
        asyncio.run(RedisHub(DownRedis(), shards=1).ingest("p", np.zeros(5), now=0.0))


def test_shards_must_be_positive():
    with pytest.raises(ValueError):
        RedisHub(fake_redis(), shards=0)


async def stream_patients(hub: RedisHub, patients: dict[str, float], seconds: int, start: float = 100.0) -> None:
    """Push `seconds` of ECG for each patient (bpm by id), timestamped as if sent in real time."""
    signals = {pid: synthetic_ecg(seconds, bpm=bpm) for pid, bpm in patients.items()}   # continuous rhythm
    for second in range(seconds):
        for pid, signal in signals.items():
            await hub.ingest(pid, signal[second * LIVE_FS:(second + 1) * LIVE_FS], now=start + second)


def test_workers_analyse_their_shards_and_hub_merges_results():
    async def scenario():
        server = fakeredis.FakeServer()
        hub = RedisHub(fake_redis(server), shards=2, clock=lambda: 120.0)
        workers = [ShardWorker(fake_redis(server), shard, clock=lambda: 120.0) for shard in (0, 1)]
        for w in workers:
            await w.setup()
        patients = {f"p{i}": 60 + 10 * i for i in range(6)}
        await stream_patients(hub, patients, seconds=8, start=112.0)
        for w in workers:
            while await w.consume_once(block_ms=10):
                pass
            hub.receive_result(await w.tick_once(now=120.0), received_at=120.0)
        return hub.snapshot(now=120.0, interval=1.0), workers
    snap, workers = asyncio.run(scenario())
    assert snap["summary"]["workers"] == 2 and snap["summary"]["patients"] == 6 and snap["summary"]["ok"] == 6
    by_id = {p["id"]: p["heart_rate_bpm"] for p in snap["patients"]}
    assert by_id["p0"] == pytest.approx(60, rel=0.05) and by_id["p5"] == pytest.approx(110, rel=0.05)
    assert all(len(w.analyzer) > 0 for w in workers)                       # both shards got patients
    assert snap["summary"]["pipeline_lag_ms"] == pytest.approx(8000, abs=1)  # oldest message (t=112) read at 120


def test_racing_heart_alert_travels_through_redis():
    async def scenario():
        server = fakeredis.FakeServer()
        hub = RedisHub(fake_redis(server), shards=1)
        worker = ShardWorker(fake_redis(server), 0)
        await worker.setup()
        result = None
        for second in range(20):
            now = 200.0 + second
            await hub.ingest("racing", synthetic_ecg(1, bpm=170, seed=second), now=now)
            await worker.consume_once(block_ms=10)
            if second >= 7:
                result = await worker.tick_once(now)
        hub.receive_result(result, received_at=219.0)
        return hub.snapshot(now=219.0, interval=1.0)
    snap = asyncio.run(scenario())
    assert snap["patients"][0]["alerts"] == ["high_heart_rate"]
    assert snap["alerts"]["active"][0]["kind"] == "high_heart_rate"


def test_stale_worker_results_are_ignored():
    hub = RedisHub(fake_redis(), shards=2)
    result = {"shard": 0, "time": 0, "analysis_ms": 1.0, "patients": [{"id": "p", "heart_rate_bpm": 70.0,
              "status": "ok", "rr_irregularity": 0.02, "alerts": []}], "waveforms": {},
              "alerts": {"active": [], "recent": []}, "lag_ms": 5.0}
    hub.receive_result(result, received_at=100.0)
    assert hub.snapshot(now=101.0, interval=1.0)["summary"]["workers"] == 1
    assert hub.snapshot(now=110.0, interval=1.0)["summary"] | {} == {
        "patients": 0, "ok": 0, "warming_up": 0, "no_signal": 0, "alerts_active": 0, "analysis_ms": 0.0,
        "devices_connected": 0, "workers": 0, "pipeline_lag_ms": 0.0}


def test_restarted_worker_rereads_unacknowledged_messages():
    async def scenario():
        server = fakeredis.FakeServer()
        hub = RedisHub(fake_redis(server), shards=1)
        crashed = ShardWorker(fake_redis(server), 0)
        await crashed.setup()
        await hub.ingest("p", np.ones(10), now=1.0)
        # The first worker reads but dies before acknowledging.
        await crashed.redis.xreadgroup(GROUP, crashed.consumer, {crashed.stream: ">"}, count=10)
        restarted = ShardWorker(fake_redis(server), 0)
        await restarted.setup()                                   # group exists already: no error
        recovered = await restarted.consume_once(block_ms=10)
        pending = await restarted.redis.xpending(restarted.stream, GROUP)
        return recovered, pending["pending"], len(restarted.analyzer)
    assert asyncio.run(scenario()) == (1, 0, 1)


def test_worker_skips_malformed_messages():
    async def scenario():
        redis = fake_redis()
        worker = ShardWorker(redis, 0)
        await worker.setup()
        await redis.xadd(worker.stream, {"p": "x"})                # no samples or timestamp
        return await worker.consume_once(block_ms=10), len(worker.analyzer)
    assert asyncio.run(scenario()) == (1, 0)


def test_worker_publishes_results_on_the_channel():
    async def scenario():
        redis = fake_redis()
        pubsub = redis.pubsub()
        await pubsub.subscribe(RESULTS_CHANNEL)
        await pubsub.get_message(timeout=0.1)                     # subscribe confirmation
        worker = ShardWorker(redis, 3)
        await worker.setup()
        await worker.tick_once(now=1.0)
        message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
        return json.loads(message["data"])
    result = asyncio.run(scenario())
    assert result["shard"] == 3 and result["patients"] == []


def test_hub_run_merges_published_results():
    async def scenario():
        server = fakeredis.FakeServer()
        hub = RedisHub(fake_redis(server), shards=1)
        queue = hub.subscribe()
        task = asyncio.create_task(hub.run(interval=0.05))
        worker = ShardWorker(fake_redis(server), 0)
        await worker.setup()
        await hub.ingest("p", synthetic_ecg(8), now=hub.clock())
        await asyncio.sleep(0.1)                                  # let the hub subscribe
        await worker.consume_once(block_ms=10)
        await worker.tick_once(hub.clock())
        for _ in range(40):
            snap = await asyncio.wait_for(queue.get(), timeout=2)
            if snap["summary"]["workers"]:
                break
        task.cancel()
        return snap
    snap = asyncio.run(scenario())
    assert snap["summary"]["workers"] == 1 and snap["patients"][0]["status"] == "ok"


def test_api_in_redis_mode_closes_devices_when_redis_is_down():
    class DownRedis(fakeredis.FakeAsyncRedis):
        async def xadd(self, *args, **kwargs):
            raise RedisConnectionError("connection refused")
    settings = replace(Settings(), samples_path=SAMPLES, live_ingest_token="t", live_backend="redis",
                       live_tick_seconds=0.05)
    with TestClient(create_app(settings, translator=object(), redis_client=DownRedis())) as client:
        with client.websocket_connect("/api/live/ingest/bed-1", headers={"X-Device-Token": "t"}) as ws:
            ws.send_json({"samples": [0.0] * (LIVE_FS // 4)})
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
    assert exc.value.code == routes.TRY_AGAIN_LATER


def test_invalid_backend_setting_is_rejected():
    with pytest.raises(ValueError, match="LIVE_BACKEND"):
        replace(Settings(), live_backend="kafka")


def test_worker_cli_validates_shard(monkeypatch):
    monkeypatch.setenv("LIVE_SHARDS", "2")
    assert worker_main(["--shard", "5"]) == 2
    with pytest.raises(SystemExit):
        parse_args(["--shard", "-1"])


def test_worker_run_loop_feeds_the_hub_end_to_end():
    async def scenario():
        server = fakeredis.FakeServer()
        hub = RedisHub(fake_redis(server), shards=1)
        hub_task = asyncio.create_task(hub.run(interval=0.05))
        worker_task = asyncio.create_task(ShardWorker(fake_redis(server), 0).run(tick_seconds=0.05))
        await asyncio.sleep(0.2)                                  # worker creates its group
        signal = synthetic_ecg(8, bpm=75)
        for second in range(8):
            await hub.ingest("p", signal[second * LIVE_FS:(second + 1) * LIVE_FS], now=hub.clock())
        for _ in range(60):
            await asyncio.sleep(0.05)
            if hub.latest and hub.latest["patients"] and hub.latest["patients"][0]["status"] == "ok":
                break
        hub_task.cancel()
        worker_task.cancel()
        return hub.latest
    latest = asyncio.run(scenario())
    assert latest["patients"][0]["heart_rate_bpm"] == pytest.approx(75, rel=0.05)


def test_worker_retries_while_redis_is_unavailable(caplog):
    class FlakyRedis(fakeredis.FakeAsyncRedis):
        failures = 2

        async def xgroup_create(self, *args, **kwargs):
            if FlakyRedis.failures:
                FlakyRedis.failures -= 1
                raise RedisConnectionError("connection refused")
            return await super().xgroup_create(*args, **kwargs)

    async def scenario(monkeypatch_retry):
        worker = ShardWorker(FlakyRedis(server=fakeredis.FakeServer()), 0)
        await worker._retrying(worker.setup)
    import cardiosignal_api.live.worker as worker_module
    original = worker_module.RETRY_SECONDS
    worker_module.RETRY_SECONDS = 0.01
    try:
        asyncio.run(scenario(None))
    finally:
        worker_module.RETRY_SECONDS = original
    assert FlakyRedis.failures == 0 and caplog.text.count("Redis unavailable") == 2
