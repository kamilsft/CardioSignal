"""Evaluate a trained checkpoint on the test split.

Writes test_metrics.json (all / healthy / CVD) and sample_<i>.png comparison plots.

Usage:
    python evaluate.py --checkpoint <best.pt> --data_dir <folder with test.npz> --out_dir <folder>
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

from cardiosignal.config import TARGET_FS
from cardiosignal.data import PairedSignals
from cardiosignal.inference import evaluate_generators, load_generators, summarize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PLOT_SECONDS = 3


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_plots", type=int, default=6)
    return p.parse_args(argv)


@torch.no_grad()
def plot_sample(g_e2p, g_p2e, ecg, pcg, title, path, device):
    fake_pcg = g_e2p(ecg.unsqueeze(0).to(device)).cpu().squeeze()
    fake_ecg = g_p2e(pcg.unsqueeze(0).to(device)).cpu().squeeze()
    n = min(ecg.shape[-1], PLOT_SECONDS * TARGET_FS)
    t = np.arange(n) / TARGET_FS
    panels = [
        ("ECG → PCG", [(pcg.squeeze(), "real PCG", "#888"), (fake_pcg, "generated PCG", "#d1495b")]),
        ("PCG → ECG", [(ecg.squeeze(), "real ECG", "#888"), (fake_ecg, "generated ECG", "#2e86ab")]),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 6), sharex=True)
    for ax, (name, lines) in zip(axes, panels):
        for signal, label, color in lines:
            ax.plot(t, signal[:n].numpy(), lw=0.8, label=label, color=color)
        ax.set_title(name, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("seconds")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = PairedSignals(str(Path(args.data_dir) / f"{args.split}.npz"))
    g_e2p, g_p2e = load_generators(args.checkpoint, device=str(device))
    per_sample = evaluate_generators(g_e2p, g_p2e, dataset, device, args.batch_size)

    is_cvd = np.asarray(dataset.is_cvd, dtype=bool)
    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "counts": {"all": int(len(is_cvd)), "healthy": int((~is_cvd).sum()), "cvd": int(is_cvd.sum())},
        "all": summarize(per_sample),
        "healthy": summarize(per_sample, ~is_cvd),
        "cvd": summarize(per_sample, is_cvd),
    }
    (out / "test_metrics.json").write_text(json.dumps(report, indent=2))

    picks = np.linspace(0, len(dataset) - 1, num=min(args.num_plots, len(dataset)), dtype=int)
    for i, idx in enumerate(picks):
        ecg, pcg = dataset[int(idx)]
        label = "CVD" if is_cvd[idx] else "healthy"
        plot_sample(g_e2p, g_p2e, ecg, pcg, f"{dataset.recording[idx]} ({label})", out / f"sample_{i}.png", device)

    print(json.dumps(report["all"], indent=2))
    print(f"Saved report and {len(picks)} plots to {out}")


if __name__ == "__main__":
    main()
