"""FastAPI application: routes, error handling, CORS, rate limiting."""
import asyncio
import logging
from contextlib import asynccontextmanager, suppress

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from cardiosignal.config import TARGET_FS

from .config import Settings
from .live.hub import LiveHub, MemoryHub
from .live.redis_hub import RedisHub
from .live.routes import live_router
from .rate_limit import RateLimiter
from .samples import SampleStore
from .schemas import (AnalysisOut, ApiResponse, Direction, HealthOut, SampleOut, SampleSummary,
                      TranslateRequest, TranslationOut)
from .signal_io import SignalError, parse_signal_file, validate_signal
from .translator import Translation, Translator

logger = logging.getLogger("cardiosignal_api")
DECIMALS = 4


def _rounded(x: np.ndarray) -> list[float]:
    return np.round(np.asarray(x, dtype=np.float64), DECIMALS).tolist()


def _warnings(t: Translation) -> list[str]:
    out = []
    pcg_analysis = t.input_analysis if t.direction == "pcg_to_ecg" else None
    if pcg_analysis is not None and pcg_analysis.quality_ok is False:
        out.append("The heart-sound recording has a weak or irregular rhythm (low quality). "
                   "The generated ECG is likely unreliable.")
    if t.ecg_was_flipped:
        out.append("The ECG appeared upside down (inverted lead) and was flipped before translation.")
    return out


def to_response(t: Translation) -> TranslationOut:
    return TranslationOut(
        direction=t.direction, sample_rate=t.sample_rate,
        input_signal=_rounded(t.input_signal), output_signal=_rounded(t.output_signal),
        input_analysis=AnalysisOut(**t.input_analysis.__dict__),
        output_analysis=AnalysisOut(**t.output_analysis.__dict__),
        ecg_was_flipped=t.ecg_was_flipped, warnings=_warnings(t))


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=ApiResponse[None](success=False, error=message).model_dump())


def make_live_hub(settings: Settings, redis_client: Redis | None = None) -> LiveHub:
    limits = (settings.live_max_patients, settings.live_max_waveforms)
    if settings.live_backend == "redis":
        return RedisHub(redis_client or Redis.from_url(settings.redis_url), settings.live_shards, *limits)
    return MemoryHub(*limits)


def create_app(settings: Settings | None = None, translator: Translator | None = None,
               redis_client: Redis | None = None) -> FastAPI:
    settings = settings or Settings()
    limiter = RateLimiter(settings.rate_limit_per_minute)
    # Inference is CPU-bound: cap how many run at once so the server (and /api/health) stays responsive.
    inference_slots = asyncio.Semaphore(max(1, settings.max_concurrent_inferences))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if translator is not None:
            app.state.translator = translator
        else:
            if not settings.model_path.exists():
                raise RuntimeError(f"Model file not found: {settings.model_path}. Set MODEL_PATH.")
            app.state.translator = Translator(str(settings.model_path), settings.torch_threads)
            logger.info("Loaded model from %s", settings.model_path)
        app.state.samples = SampleStore(settings.samples_path)
        app.state.inference_slots = inference_slots
        app.state.hub = make_live_hub(settings, redis_client)
        logger.info("Live monitor backend: %s", settings.live_backend)
        monitor = asyncio.create_task(app.state.hub.run(settings.live_tick_seconds))
        try:
            yield
        finally:
            monitor.cancel()
            with suppress(asyncio.CancelledError):
                await monitor
            await app.state.hub.close()

    app = FastAPI(title="CardioSignal API", version="1.0.0", lifespan=lifespan,
                  description="Translate between ECG and heart-sound (PCG) signals. Research demo, not for diagnosis.")
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins),
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.exception_handler(SignalError)
    async def signal_error(_: Request, exc: SignalError):
        return _error(422, str(exc))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        return _error(422, f"Invalid request: {field} {first.get('msg', '')}".strip())

    @app.exception_handler(StarletteHTTPException)   # also covers unknown routes (404) and wrong methods (405)
    async def http_error(_: Request, exc: StarletteHTTPException):
        return _error(exc.status_code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return _error(500, "Something went wrong while processing the signal. Please try again.")

    def rate_limited(request: Request):
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(429, "Too many requests. Please wait a minute and try again.")

    async def run_translation(signal: np.ndarray, sample_rate: int, direction: Direction) -> ApiResponse[TranslationOut]:
        x = validate_signal(signal, sample_rate, settings.min_seconds, settings.max_seconds)
        if inference_slots.locked():
            raise HTTPException(503, "The server is busy translating other signals. Please try again in a few seconds.")
        async with inference_slots:
            result = await run_in_threadpool(app.state.translator.translate, x, sample_rate, direction)
        return ApiResponse[TranslationOut](success=True, data=to_response(result))

    @app.get("/api/health", response_model=ApiResponse[HealthOut])
    async def health():
        return ApiResponse[HealthOut](success=True, data=HealthOut(
            status="ok", model_loaded=app.state.translator is not None, samples=len(app.state.samples.list())))

    @app.get("/api/samples", response_model=ApiResponse[list[SampleSummary]])
    async def list_samples():
        items = [SampleSummary(id=s.id, label=s.label, description=s.description) for s in app.state.samples.list()]
        return ApiResponse[list[SampleSummary]](success=True, data=items)

    @app.get("/api/samples/{sample_id}", response_model=ApiResponse[SampleOut])
    async def get_sample(sample_id: str):
        sample = app.state.samples.get(sample_id)
        if sample is None:
            raise HTTPException(404, f"Sample '{sample_id}' not found.")
        return ApiResponse[SampleOut](success=True, data=SampleOut(
            id=sample.id, label=sample.label, description=sample.description, sample_rate=TARGET_FS,
            ecg=_rounded(sample.ecg), pcg=_rounded(sample.pcg)))

    @app.post("/api/translate", response_model=ApiResponse[TranslationOut], dependencies=[Depends(rate_limited)])
    async def translate(body: TranslateRequest):
        return await run_translation(np.asarray(body.signal), body.sample_rate, body.direction)

    @app.post("/api/translate/upload", response_model=ApiResponse[TranslationOut], dependencies=[Depends(rate_limited)])
    async def translate_upload(file: UploadFile = File(...), direction: Direction = Form(...),
                               sample_rate: int | None = Form(None)):
        content = await file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise SignalError(f"File is too large; the limit is {settings.max_upload_bytes // (1024 * 1024)} MB.")
        signal, rate = parse_signal_file(file.filename or "", content, sample_rate)
        return await run_translation(signal, rate, direction)

    app.include_router(live_router(settings, rate_limited))
    return app
