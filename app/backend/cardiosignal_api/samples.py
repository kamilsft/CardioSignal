"""Demo recordings from the test split (not used in training), shipped with the app."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Sample:
    id: str
    label: str           # "healthy" or "cvd"
    description: str
    ecg: np.ndarray
    pcg: np.ndarray


class SampleStore:
    def __init__(self, path: Path):
        self._samples: dict[str, Sample] = {}
        if not path.exists():
            return
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in ("ids", "labels", "descriptions", "ecg", "pcg")}  # decompress each once
        for i, sample_id in enumerate(arrays["ids"]):
            self._samples[str(sample_id)] = Sample(
                id=str(sample_id), label=str(arrays["labels"][i]), description=str(arrays["descriptions"][i]),
                ecg=arrays["ecg"][i].astype(np.float32), pcg=arrays["pcg"][i].astype(np.float32))

    def list(self) -> list[Sample]:
        return list(self._samples.values())

    def get(self, sample_id: str) -> Sample | None:
        return self._samples.get(sample_id)
