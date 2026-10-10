"""Loading checkpoints and running generators. Used by training, evaluation and the app."""
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from .metrics import compute_all
from .models import Generator

DEFAULT_MODEL_CONFIG = {"channels": [64, 128, 256], "strides": [4, 3, 2], "heads": 4, "layers": 4, "ff_dim": 512}


def build_generator(model_config: dict | None = None) -> Generator:
    cfg = {**DEFAULT_MODEL_CONFIG, **(model_config or {})}
    return Generator(channels=tuple(cfg["channels"]), strides=tuple(cfg["strides"]),
                     heads=cfg["heads"], layers=cfg["layers"], ff_dim=cfg["ff_dim"])


def load_generators(checkpoint_path: str, device: str = "cpu"):
    """Return (ecg_to_pcg, pcg_to_ecg) in eval mode from a best.pt / last.pt checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    generators = []
    for key in ("g_e2p", "g_p2e"):
        g = build_generator(ckpt.get("model_config"))
        g.load_state_dict(ckpt[key])
        generators.append(g.to(device).eval())
    return tuple(generators)


@torch.no_grad()
def evaluate_generators(g_e2p, g_p2e, dataset, device, batch_size: int = 32, max_batches: int | None = None):
    """Per-sample metrics for both directions plus 'copy the input' baselines."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    results = defaultdict(lambda: defaultdict(list))
    for i, (ecg, pcg) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        ecg, pcg = ecg.to(device), pcg.to(device)
        pairs = {
            "ecg_to_pcg": (g_e2p(ecg).float(), pcg),
            "pcg_to_ecg": (g_p2e(pcg).float(), ecg),
            "baseline_ecg_as_pcg": (ecg, pcg),
            "baseline_pcg_as_ecg": (pcg, ecg),
        }
        for name, (pred, target) in pairs.items():
            for metric, values in compute_all(pred, target).items():
                results[name][metric].extend(values)
    return {name: dict(metrics) for name, metrics in results.items()}


def summarize(per_sample: dict, mask=None) -> dict:
    """Mean of each metric, optionally over a boolean mask of samples."""
    out = {}
    for name, metrics in per_sample.items():
        out[name] = {}
        for metric, values in metrics.items():
            t = torch.tensor(values, dtype=torch.float64)
            if mask is not None:
                t = t[torch.as_tensor(mask[: len(t)], dtype=torch.bool)]
            out[name][metric] = round(float(t.mean()), 4) if len(t) else None
    return out


def selection_score(summary: dict) -> float:
    """Single number for picking the best checkpoint: higher is better."""
    return summary["ecg_to_pcg"]["envelope_pearson"] + summary["pcg_to_ecg"]["pearson"]
