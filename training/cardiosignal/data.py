"""Dataset over the compact .npz splits produced by prepare_data.py."""
import numpy as np
import torch
from torch.utils.data import Dataset

from .beats import peak_weight_mask
from .config import TARGET_FS

PEAK_HALF_WIDTH = int(0.04 * TARGET_FS)  # +-40 ms around each R-peak


class PairedSignals(Dataset):
    """Paired, already-scaled ECG/PCG segments as float32 tensors of shape [1, T].

    Returns (ecg, pcg), or (ecg, pcg, ecg_weight) when peak_boost > 0: per-sample
    loss weights that emphasize the heartbeat spikes (needs r_peaks in the file;
    without them the weights are all 1).
    """

    def __init__(self, npz_path: str, augment: bool = False, peak_boost: float = 0.0):
        with np.load(npz_path) as data:
            self.ecg = data["ecg"]
            self.pcg = data["pcg"]
            self.is_cvd = data["is_cvd"]
            self.recording = data["recording"]
            self.r_peaks = data["r_peaks"] if "r_peaks" in data.files else None
            self.person = data["person"] if "person" in data.files else self.recording
        if self.ecg.shape != self.pcg.shape:
            raise ValueError(f"ECG/PCG shape mismatch in {npz_path}: {self.ecg.shape} vs {self.pcg.shape}")
        self.augment = augment
        self.peak_boost = peak_boost

    def __len__(self):
        return len(self.ecg)

    def __getitem__(self, index):
        ecg = torch.from_numpy(self.ecg[index].astype(np.float32)).unsqueeze(0)
        pcg = torch.from_numpy(self.pcg[index].astype(np.float32)).unsqueeze(0)
        if self.augment:
            ecg, pcg = _augment(ecg, pcg)
        if self.peak_boost <= 0:
            return ecg, pcg
        peaks = self.r_peaks[index] if self.r_peaks is not None else np.array([], dtype=int)
        weight = peak_weight_mask(ecg.shape[-1], peaks, PEAK_HALF_WIDTH, self.peak_boost)
        return ecg, pcg, torch.from_numpy(weight).unsqueeze(0)


def recording_weights(recordings, power: float = 0.5) -> np.ndarray:
    """Per-segment sampling weights that soften the dominance of long recordings.

    Each recording's total weight becomes n**(1 - power): power=0 keeps natural
    frequencies, power=1 makes every recording equally likely, 0.5 is in between.
    """
    _, inverse, counts = np.unique(np.asarray(recordings), return_inverse=True, return_counts=True)
    return counts[inverse].astype(np.float64) ** -power


def _augment(ecg: torch.Tensor, pcg: torch.Tensor):
    """Same random gain on both signals: keeps the pair aligned, adds amplitude variety.

    Kept below 1 so the scaled signals stay inside the generator's tanh range.
    """
    gain = float(torch.empty(1).uniform_(0.7, 1.0))
    return ecg * gain, pcg * gain
