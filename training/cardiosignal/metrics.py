"""Evaluation metrics. All functions take [B, 1, T] tensors and return per-sample values."""
import torch

from .signal import envelope


def pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.flatten(1).double()
    b = b.flatten(1).double()
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)


def envelope_pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return pearson(envelope(a), envelope(b))


def mae(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().flatten(1).mean(1).double()


METRICS = {"pearson": pearson, "envelope_pearson": envelope_pearson, "mae": mae}


def compute_all(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Per-sample metric values as python lists, keyed by metric name."""
    return {name: fn(pred, target).cpu().tolist() for name, fn in METRICS.items()}
