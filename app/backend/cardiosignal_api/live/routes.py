"""Live monitor endpoints.

    WS  /api/live/ingest/{patient_id}   device -> server, header X-Device-Token, messages {"samples": [...]}
    WS  /api/live/stream                server -> dashboard, one snapshot per tick
    GET /api/live/patients              latest heart rates (no waveforms)
"""
import asyncio
import hmac
import json
import logging
import re
from collections import Counter
from typing import Any, Callable

import numpy as np
from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect

from ..config import Settings
from ..schemas import ApiResponse
from .hub import LiveHub, LiveUnavailable
from .monitor import LIVE_FS

logger = logging.getLogger("cardiosignal_api.live")
PATIENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
MAX_MESSAGE_CHARS = 32_000          # 250 numbers need ~3 kB
MAX_ABS_VALUE = 1e4
IDLE_TIMEOUT_SECONDS = 15        # a device silent this long is disconnected, freeing its slot
GOING_AWAY = 1001                # retryable
UNSUPPORTED_DATA = 1003          # bad message: fix the device
POLICY_VIOLATION = 1008          # bad token / id / rate: fix the device, do not retry
TRY_AGAIN_LATER = 1013           # monitor or viewer limit: retry later
REPLACED = 4000                  # superseded by a newer connection for the same patient


class InvalidMessage(ValueError):
    pass


class RateBudget:
    """Token bucket: a device may send at most ~real time, plus a short backlog after a reconnect."""

    BURST_SECONDS = 10
    SPEED_TOLERANCE = 1.25

    def __init__(self, fs: int, now: float):
        self.capacity = fs * self.BURST_SECONDS
        self.refill_per_second = fs * self.SPEED_TOLERANCE
        self.tokens = float(self.capacity)
        self.updated = now

    def spend(self, samples: int, now: float) -> bool:
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.refill_per_second)
        self.updated = now
        if samples > self.tokens:
            return False
        self.tokens -= samples
        return True


def parse_samples(text: str | None) -> np.ndarray:
    if text is None:
        raise InvalidMessage("Send JSON text messages, not binary.")
    if len(text) > MAX_MESSAGE_CHARS:
        raise InvalidMessage("Message too large.")
    try:
        body = json.loads(text)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise InvalidMessage("Message is not valid JSON.") from exc
    samples = body.get("samples") if isinstance(body, dict) else None
    if not isinstance(samples, list) or not 1 <= len(samples) <= LIVE_FS:
        raise InvalidMessage(f'Expected {{"samples": [...]}} with 1-{LIVE_FS} numbers.')
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in samples):
        raise InvalidMessage("Samples must be numbers.")
    x = np.asarray(samples, dtype=np.float32)
    if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > MAX_ABS_VALUE:
        raise InvalidMessage("Samples must be finite numbers.")
    return x


def _device_refusal(settings: Settings, ws: WebSocket, patient_id: str) -> str | None:
    if not settings.live_ingest_token:
        return "Live ingest is disabled: set LIVE_INGEST_TOKEN on the server."
    given = ws.headers.get("x-device-token", "")
    if not hmac.compare_digest(given.encode(), settings.live_ingest_token.encode()):
        return "Invalid device token."
    if not PATIENT_ID.fullmatch(patient_id):
        return "Patient id must be 1-32 letters, digits, '-' or '_'."
    return None


def live_router(settings: Settings, rate_limited: Callable[[Request], None]) -> APIRouter:
    router = APIRouter(prefix="/api/live")
    devices: dict[str, WebSocket] = {}           # patient_id -> its current connection
    viewers: Counter[str] = Counter()            # client IP -> open dashboards

    @router.get("/patients", response_model=ApiResponse[dict[str, Any]], dependencies=[Depends(rate_limited)])
    async def patients(request: Request):
        latest = request.app.state.hub.latest
        data = {"summary": latest["summary"], "patients": latest["patients"]} if latest else             {"summary": None, "patients": []}
        return ApiResponse[dict[str, Any]](success=True, data=data)

    @router.websocket("/ingest/{patient_id}")
    async def ingest(ws: WebSocket, patient_id: str):
        hub: LiveHub = ws.app.state.hub
        # Accept before refusing, so the device sees the close code and reason (not a bare HTTP 403).
        await ws.accept()
        refusal = _device_refusal(settings, ws, patient_id)
        if refusal is not None:
            await ws.close(POLICY_VIOLATION, refusal)
            return
        previous = devices.get(patient_id)
        if previous is not None:
            # A reconnecting device replaces its old (possibly half-open) connection instead of being locked out.
            devices[patient_id] = ws
            await _close_quietly(previous, REPLACED, "Replaced by a newer connection.")
        elif not hub.connect(patient_id):
            await ws.close(TRY_AGAIN_LATER, "The monitor is full.")
            return
        devices[patient_id] = ws
        budget = RateBudget(hub.fs, hub.clock())
        try:
            while True:
                message = await asyncio.wait_for(ws.receive(), timeout=IDLE_TIMEOUT_SECONDS)
                if message["type"] == "websocket.disconnect":
                    break
                samples = parse_samples(message.get("text"))
                now = hub.clock()
                if not budget.spend(len(samples), now):
                    await ws.close(POLICY_VIOLATION, "Sending faster than real time.")
                    break
                await hub.ingest(patient_id, samples, now)
        except InvalidMessage as exc:
            await _close_quietly(ws, UNSUPPORTED_DATA, str(exc))
        except LiveUnavailable as exc:
            logger.warning("Dropping device %s: %s", patient_id, exc.__cause__ or exc)
            await _close_quietly(ws, TRY_AGAIN_LATER, str(exc))
        except TimeoutError:
            await _close_quietly(ws, GOING_AWAY, f"No data for {IDLE_TIMEOUT_SECONDS} s.")
        except (WebSocketDisconnect, RuntimeError):     # RuntimeError: replaced connection already closed
            pass
        finally:
            if devices.get(patient_id) is ws:           # a replacement keeps the patient registered
                del devices[patient_id]
                hub.disconnect(patient_id)

    @router.websocket("/stream")
    async def stream(ws: WebSocket):
        hub: LiveHub = ws.app.state.hub
        client = ws.client.host if ws.client else "unknown"
        await ws.accept()
        origin = ws.headers.get("origin")
        if origin is not None and origin not in settings.allowed_origins:
            await ws.close(POLICY_VIOLATION, "Origin not allowed.")
            return
        if viewers[client] >= settings.live_max_viewers_per_ip or viewers.total() >= settings.live_max_viewers:
            await ws.close(TRY_AGAIN_LATER, "Too many viewers; try again later.")
            return
        viewers[client] += 1
        queue = hub.subscribe()
        try:
            while True:
                await ws.send_json(await queue.get())
        except (WebSocketDisconnect, RuntimeError):     # RuntimeError: send after the client went away
            pass
        finally:
            hub.unsubscribe(queue)
            viewers[client] -= 1
            if viewers[client] <= 0:
                del viewers[client]

    return router


async def _close_quietly(ws: WebSocket, code: int, reason: str) -> None:
    try:
        await ws.close(code, reason)
    except RuntimeError:            # already closed
        pass
