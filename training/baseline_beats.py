"""Non-AI baseline: find heartbeats directly in the heart sound, no model.

Each heartbeat makes a loud first heart sound (S1, "lub") shortly after the ECG's
R-peak. The baseline smooths the heart-sound loudness, picks its peaks, and shifts
them earlier by a fixed offset to estimate the R-peaks. Its settings are tuned on
the *training* people only, then it is scored once on the test people with exactly
the same metric as the model (beat_accuracy.py).

Usage:
    python baseline_beats.py --data_dir <dir with train.npz/test.npz> --out <json>
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from beat_accuracy import beat_scores, per_group_f1
from cardiosignal.beats import PCG_QUALITY_THRESHOLD, detect_r_peaks, periodicity
from cardiosignal.config import ENVELOPE_WINDOW, TARGET_FS

ENVELOPE_FS = TARGET_FS / (ENVELOPE_WINDOW // 2)
GRID = {
    "smooth_ms": (20, 40, 80),
    "min_gap_s": (0.2, 0.25, 0.3, 0.4),
    "height": (0.35, 0.5, 0.65, 0.8),
    "offset_ms": tuple(range(-100, 1, 10)),    # negative: R-peak comes before the heart sound
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tune_segments", type=int, default=2000, help="training segments used for tuning")
    p.add_argument("--tolerance_ms", type=float, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def heart_sound_beats(pcg: np.ndarray, smooth_ms: float, min_gap_s: float, height: float, offset_ms: float,
                      fs: int = TARGET_FS) -> np.ndarray:
    """Estimated R-peak positions from a heart-sound recording, without any learned model."""
    window = max(1, int(smooth_ms / 1000 * fs))
    loudness = np.convolve(np.abs(np.asarray(pcg, dtype=np.float64)), np.ones(window) / window, mode="same")
    ceiling = np.percentile(loudness, 99)
    if ceiling <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(loudness, height=height * ceiling, distance=int(min_gap_s * fs))
    shifted = peaks + int(offset_ms / 1000 * fs)
    return shifted[(shifted >= 0) & (shifted < len(pcg))]


def load(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def tune(data, idx, tolerance):
    """Grid-search the baseline's settings on training segments; return the best settings and their F1."""
    real = {i: detect_r_peaks(data["ecg"][i].astype(np.float32)) for i in idx}
    pcg = {i: data["pcg"][i].astype(np.float32) for i in idx}
    best, best_f1 = None, -1.0
    for values in itertools.product(*GRID.values()):
        settings = dict(zip(GRID, values))
        f1 = beat_scores([(real[i], heart_sound_beats(pcg[i], **settings)) for i in idx], tolerance)["f1_pct"]
        if f1 > best_f1:
            best, best_f1 = settings, f1
    return best, best_f1


def evaluate(data, settings, tolerance):
    rows = []
    for i in range(len(data["ecg"])):
        pcg = data["pcg"][i].astype(np.float32)
        env = np.convolve(np.abs(pcg), np.ones(ENVELOPE_WINDOW) / ENVELOPE_WINDOW, mode="valid")[:: ENVELOPE_WINDOW // 2]
        rows.append({"recording": str(data["recording"][i]), "person": str(data["person"][i]),
                     "is_cvd": bool(data["is_cvd"][i]), "quality": periodicity(env, ENVELOPE_FS),
                     "real_peaks": detect_r_peaks(data["ecg"][i].astype(np.float32)),
                     "fake_peaks": heart_sound_beats(pcg, **settings)})
    groups = {"all": rows, "good_quality": [r for r in rows if r["quality"] >= PCG_QUALITY_THRESHOLD],
              "healthy": [r for r in rows if not r["is_cvd"]], "cvd": [r for r in rows if r["is_cvd"]]}
    return {name: {"segments": len(g),
                   "beats": beat_scores([(r["real_peaks"], r["fake_peaks"]) for r in g], tolerance),
                   "per_person": per_group_f1(g, tolerance, "person")}
            for name, g in groups.items() if g}


def main(argv=None):
    args = parse_args(argv)
    tolerance = int(args.tolerance_ms / 1000 * TARGET_FS)
    train = load(Path(args.data_dir) / "train.npz")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(train["ecg"]), size=min(args.tune_segments, len(train["ecg"])), replace=False)
    settings, train_f1 = tune(train, idx, tolerance)
    print(f"tuned on {len(idx)} training segments: {settings} (train F1 {train_f1}%)", flush=True)
    result = {"settings": settings, "train_f1_pct": train_f1,
              "test": evaluate(load(Path(args.data_dir) / "test.npz"), settings, tolerance)}
    Path(args.out).write_text(json.dumps(result, indent=2))
    for name, r in result["test"].items():
        print(f"{name:13s} baseline beat F1 {r['beats']['f1_pct']}% | per-person median {r['per_person']['median_f1_pct']}%")


if __name__ == "__main__":
    main()
