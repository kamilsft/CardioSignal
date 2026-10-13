"""Request and response models. Every response uses the same envelope."""
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")
Direction = Literal["ecg_to_pcg", "pcg_to_ecg"]
# JSON is for samples and short signals; long high-rate recordings should be uploaded as files.
# Roughly matches nginx's 6 MB body limit (about 600k numbers).
MAX_JSON_SAMPLES = 600_000


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    error: str | None = None


class TranslateRequest(BaseModel):
    direction: Direction
    sample_rate: int = Field(ge=250, le=48000, description="Sample rate of `signal` in Hz")
    signal: list[float] = Field(min_length=1, max_length=MAX_JSON_SAMPLES)


class AnalysisOut(BaseModel):
    heart_rate_bpm: float | None
    beats: list[int]
    quality: float | None = None
    quality_ok: bool | None = None


class TranslationOut(BaseModel):
    direction: Direction
    sample_rate: int
    input_signal: list[float]
    output_signal: list[float]
    input_analysis: AnalysisOut
    output_analysis: AnalysisOut
    ecg_was_flipped: bool
    warnings: list[str]


class SampleSummary(BaseModel):
    id: str
    label: str
    description: str


class SampleOut(SampleSummary):
    sample_rate: int
    ecg: list[float]
    pcg: list[float]


class HealthOut(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    samples: int
