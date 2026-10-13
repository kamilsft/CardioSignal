"""Runtime settings, read from environment variables with safe defaults."""
import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    model_path: Path = field(default_factory=lambda: Path(os.getenv("MODEL_PATH", BACKEND_DIR / "models" / "model.pt")))
    samples_path: Path = field(default_factory=lambda: Path(os.getenv("SAMPLES_PATH", BACKEND_DIR / "data" / "samples.npz")))
    allowed_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()))
    max_upload_bytes: int = field(default_factory=lambda: _int_env("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))
    min_seconds: int = field(default_factory=lambda: _int_env("MIN_SECONDS", 3))
    max_seconds: int = field(default_factory=lambda: _int_env("MAX_SECONDS", 60))
    rate_limit_per_minute: int = field(default_factory=lambda: _int_env("RATE_LIMIT_PER_MINUTE", 30))
    torch_threads: int = field(default_factory=lambda: _int_env("TORCH_THREADS", 4))
    max_concurrent_inferences: int = field(default_factory=lambda: _int_env("MAX_CONCURRENT_INFERENCES", 2))
    # Live monitor. Devices must send this token; empty disables device ingest entirely.
    live_ingest_token: str = field(default_factory=lambda: os.getenv("LIVE_INGEST_TOKEN", ""))
    live_tick_seconds: float = field(default_factory=lambda: _float_env("LIVE_TICK_SECONDS", 1.0))
    live_max_patients: int = field(default_factory=lambda: _int_env("LIVE_MAX_PATIENTS", 2000))
    live_max_waveforms: int = field(default_factory=lambda: _int_env("LIVE_MAX_WAVEFORMS", 24))
    live_max_viewers: int = field(default_factory=lambda: _int_env("LIVE_MAX_VIEWERS", 50))
    live_max_viewers_per_ip: int = field(default_factory=lambda: _int_env("LIVE_MAX_VIEWERS_PER_IP", 5))
    # "memory": analyse in this process. "redis": forward to Redis; `live.worker` processes analyse.
    live_backend: str = field(default_factory=lambda: os.getenv("LIVE_BACKEND", "memory"))
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    live_shards: int = field(default_factory=lambda: _int_env("LIVE_SHARDS", 2))

    def __post_init__(self) -> None:
        if self.live_backend not in ("memory", "redis"):
            raise ValueError(f"LIVE_BACKEND must be 'memory' or 'redis', got {self.live_backend!r}")
        if self.live_shards < 1:
            raise ValueError("LIVE_SHARDS must be at least 1")
