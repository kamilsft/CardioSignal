"""Heartbeat-level accuracy of a checkpoint on the test set.

PCG -> ECG: are the generated ECG's heartbeats where the real ones are (within +-tolerance)?
ECG -> PCG: does the generated heart sound beat at the right heart rate?
Reported for all segments, healthy / CVD, good-quality heart sounds only, and as
the median over recordings (long recordings otherwise dominate the pooled number).
A chance baseline pairs each real ECG with a *different* test segment's beats.

Usage:
    python beat_accuracy.py --checkpoint <pt> --data_dir <dir with test.npz> --out <json>
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cardiosignal.beats import (PCG_QUALITY_THRESHOLD, detect_r_peaks, heart_rate_from_envelope,
                                heart_rate_from_peaks, match_beats, periodicity)
from cardiosignal.config import ENVELOPE_WINDOW, TARGET_FS
from cardiosignal.data import PairedSignals
from cardiosignal.inference import load_generators
from cardiosignal.signal import envelope

ENVELOPE_FS = TARGET_FS / (ENVELOPE_WINDOW // 2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tolerance_ms", type=float, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


@torch.no_grad()
def collect(checkpoint, dataset, batch_size):
    g_e2p, g_p2e = load_generators(checkpoint)
    rows, k = [], 0
    for ecg, pcg in DataLoader(dataset, batch_size=batch_size):
        fake_ecg, fake_pcg = g_p2e(pcg), g_e2p(ecg)
        env_fake, env_real = envelope(fake_pcg), envelope(pcg)
        for i in range(len(ecg)):
            real_env = env_real[i, 0].numpy()
            rows.append({"recording": str(dataset.recording[k]), "person": str(dataset.person[k]), "is_cvd": bool(dataset.is_cvd[k]),
                         "quality": periodicity(real_env, ENVELOPE_FS),
                         "real_peaks": detect_r_peaks(ecg[i, 0].numpy()),
                         "fake_peaks": detect_r_peaks(fake_ecg[i, 0].numpy()),
                         "hr_fake_pcg": heart_rate_from_envelope(env_fake[i, 0].numpy(), ENVELOPE_FS),
                         "hr_real_pcg": heart_rate_from_envelope(real_env, ENVELOPE_FS)})
            k += 1
    return rows


def beat_scores(pairs, tolerance):
    hits = n_pred = n_ref = 0
    for ref, pred in pairs:
        hits += match_beats(ref, pred, tolerance)
        n_pred += len(pred)
        n_ref += len(ref)
    precision = hits / max(n_pred, 1)
    recall = hits / max(n_ref, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"recall_pct": round(100 * recall, 1), "precision_pct": round(100 * precision, 1),
            "f1_pct": round(100 * f1, 1), "real_beats": n_ref}


def hr_scores(reference, estimate):
    errors = np.array([abs(r - e) for r, e in zip(reference, estimate) if r is not None and e is not None])
    if len(errors) == 0:
        return None
    return {"median_abs_error_bpm": round(float(np.median(errors)), 1),
            "within_5_bpm_pct": round(100 * float(np.mean(errors <= 5)), 1),
            "within_10_bpm_pct": round(100 * float(np.mean(errors <= 10)), 1),
            "segments": int(len(errors))}


def per_group_f1(rows, tolerance, key):
    """Beat F1 computed separately for each recording or person, then summarized."""
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append((r["real_peaks"], r["fake_peaks"]))
    f1s = {g: beat_scores(pairs, tolerance)["f1_pct"] for g, pairs in groups.items()}
    values = list(f1s.values())
    return {"count": len(values), "median_f1_pct": round(float(np.median(values)), 1) if values else None,
            "at_least_80_pct": int(sum(f >= 80 for f in values)),
            **({"per_person_f1_pct": dict(sorted(f1s.items()))} if key == "person" else {})}


def report(rows, tolerance, rng):
    if not rows:
        return None
    shuffled = rng.permutation(len(rows))
    hr_true = [heart_rate_from_peaks(r["real_peaks"]) for r in rows]
    return {
        "segments": len(rows),
        "pcg_to_ecg_beats": beat_scores([(r["real_peaks"], r["fake_peaks"]) for r in rows], tolerance),
        "pcg_to_ecg_beats_per_recording": per_group_f1(rows, tolerance, "recording"),
        "pcg_to_ecg_beats_per_person": per_group_f1(rows, tolerance, "person"),
        "chance_beats": beat_scores([(rows[i]["real_peaks"], rows[j]["fake_peaks"]) for i, j in enumerate(shuffled)], tolerance),
        "pcg_to_ecg_heart_rate": hr_scores(hr_true, [heart_rate_from_peaks(r["fake_peaks"]) for r in rows]),
        "ecg_to_pcg_heart_rate": hr_scores(hr_true, [r["hr_fake_pcg"] for r in rows]),
        "real_pcg_heart_rate_reference": hr_scores(hr_true, [r["hr_real_pcg"] for r in rows]),
        "chance_heart_rate": hr_scores(hr_true, [hr_true[j] for j in shuffled]),
    }


def main(argv=None):
    args = parse_args(argv)
    dataset = PairedSignals(str(Path(args.data_dir) / "test.npz"))
    rows = collect(args.checkpoint, dataset, args.batch_size)
    tolerance = int(args.tolerance_ms / 1000 * TARGET_FS)
    rng = np.random.default_rng(args.seed)
    good = [r for r in rows if r["quality"] >= PCG_QUALITY_THRESHOLD]
    result = {
        "checkpoint": str(args.checkpoint), "tolerance_ms": args.tolerance_ms,
        "quality_threshold": PCG_QUALITY_THRESHOLD,
        "good_quality_share_pct": round(100 * len(good) / max(len(rows), 1), 1),
        "all": report(rows, tolerance, rng),
        "good_quality": report(good, tolerance, rng),
        "healthy": report([r for r in rows if not r["is_cvd"]], tolerance, rng),
        "cvd": report([r for r in rows if r["is_cvd"]], tolerance, rng),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("good_quality_share_pct",)}, indent=2))
    for group in ("all", "good_quality", "healthy", "cvd"):
        r = result[group]
        if r:
            print(f"{group:13s} beat F1 {r['pcg_to_ecg_beats']['f1_pct']}% | per-recording median "
                  f"{r['pcg_to_ecg_beats_per_recording']['median_f1_pct']}% | per-person median "
                  f"{r['pcg_to_ecg_beats_per_person']['median_f1_pct']}% | chance {r['chance_beats']['f1_pct']}%")


if __name__ == "__main__":
    main()
