"""Side-by-side comparison of two trained checkpoints on the same test set.

Produces:
  scores.png   bar chart of test metrics (all / healthy / CVD) for each model
  overlay_<i>.png  the same test segments translated by each model, over the real signal

Usage:
    python compare.py --data_dir <dir with test.npz> --out_dir <dir>
                      --model v1 <best.pt> <test_metrics.json> --model v2 <best.pt> <test_metrics.json>
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

from cardiosignal.config import TARGET_FS
from cardiosignal.data import PairedSignals
from cardiosignal.inference import load_generators

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#9aa5b1", "#d1495b", "#2e86ab", "#edae49"]
PANELS = [("ecg_to_pcg", "envelope_pearson", "ECG → PCG\nheart-sound timing (envelope r)"),
          ("pcg_to_ecg", "pearson", "PCG → ECG\nwaveform match (r)")]
GROUPS = ["all", "healthy", "cvd"]
PLOT_SECONDS = 3


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", nargs=3, action="append", metavar=("NAME", "CHECKPOINT", "METRICS_JSON"), required=True)
    p.add_argument("--num_overlays", type=int, default=3)
    return p.parse_args(argv)


def plot_scores(models, path):
    reports = {name: json.loads(Path(metrics).read_text()) for name, _, metrics in models}
    first = next(iter(reports.values()))
    fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 4.5))
    x = np.arange(len(GROUPS))
    bars = [("no model", None)] + [(name, name) for name in reports]
    width = 0.8 / len(bars)
    for ax, (direction, metric, title) in zip(axes, PANELS):
        baseline_key = "baseline_ecg_as_pcg" if direction == "ecg_to_pcg" else "baseline_pcg_as_ecg"
        for i, (label, key) in enumerate(bars):
            values = [(first[g][baseline_key] if key is None else reports[key][g][direction])[metric] for g in GROUPS]
            rects = ax.bar(x + (i - (len(bars) - 1) / 2) * width, values, width, label=label, color=COLORS[i % len(COLORS)])
            ax.bar_label(rects, fmt="%.2f", fontsize=8, padding=2)
        ax.set_xticks(x, [f"{g}\n(n={first['counts'][g]})" for g in GROUPS])
        ax.set_title(title, fontsize=11)
        ax.set_ylim(min(0, ax.get_ylim()[0]), 1)
        ax.axhline(0, color="black", lw=0.6)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Test set: higher is better (1.0 = perfect)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def plot_overlays(models, dataset, out, count):
    gens = {name: load_generators(ckpt) for name, ckpt, _ in models}
    is_cvd = np.asarray(dataset.is_cvd, dtype=bool)
    healthy_idx = np.flatnonzero(~is_cvd)
    picks = list(healthy_idx[np.linspace(0, len(healthy_idx) - 1, max(1, count - 1), dtype=int)])
    if is_cvd.any():
        picks.append(int(np.flatnonzero(is_cvd)[len(np.flatnonzero(is_cvd)) // 2]))
    n = PLOT_SECONDS * TARGET_FS
    t = np.arange(n) / TARGET_FS
    for i, idx in enumerate(picks):
        ecg, pcg = dataset[int(idx)]
        fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
        rows = [(axes[0], "ECG → PCG", pcg, lambda g: g[0](ecg.unsqueeze(0))),
                (axes[1], "PCG → ECG", ecg, lambda g: g[1](pcg.unsqueeze(0)))]
        for ax, title, real, run in rows:
            ax.plot(t, real.squeeze()[:n].numpy(), color="black", lw=1.2, alpha=0.35, label="real")
            for j, (name, g) in enumerate(gens.items()):
                ax.plot(t, run(g).squeeze()[:n].numpy(), lw=0.9, color=COLORS[(j + 1) % len(COLORS)], label=name)
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
        axes[-1].set_xlabel("seconds")
        label = "CVD" if is_cvd[idx] else "healthy"
        fig.suptitle(f"Same test segment, each model's output: {dataset.recording[idx]} ({label})", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"overlay_{i}.png", dpi=110)
        plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_scores(args.model, out / "scores.png")
    plot_overlays(args.model, PairedSignals(str(Path(args.data_dir) / "test.npz")), out, args.num_overlays)
    print(f"Saved comparison plots to {out}")


if __name__ == "__main__":
    main()
