"""Convert the lab's 8 kHz .pt recordings into compact 2 kHz train/val/test .npz files.

Splits are made per *person* (see splits.py), never per recording or segment, so
no person's heartbeats appear in both training and test data.

Usage:
    python prepare_data.py --healthy <dir> --cvd <dir> --out <dir>
"""
import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from cardiosignal.beats import detect_r_peaks
from cardiosignal.config import SEGMENT_LEN, SOURCE_FS
from cardiosignal.signal import downsample, fix_ecg_polarity, scale_to_unit
from splits import assert_no_person_overlap, identify, load_ephnogram_subjects, split_people

DATA_VERSION = "v4-person-split"
MAX_BEATS = 32  # 6 s at 200 bpm is 20 beats; padded with -1
SOURCE_LEN = SOURCE_FS * 6
DEFAULT_EPHNOGRAM_META = Path(__file__).resolve().parent / "metadata" / "ephnogram_subjects.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--healthy", required=True, help="folder with healthy recordings (ECGPCG* and Measure_*)")
    p.add_argument("--cvd", required=True, help="folder with heart-disease recordings (New_CVD)")
    p.add_argument("--out", required=True)
    p.add_argument("--ephnogram_meta", default=str(DEFAULT_EPHNOGRAM_META))
    p.add_argument("--healthy_per_file", type=int, default=0, help="max segments per healthy file (0 = all)")
    p.add_argument("--measure_val_test", type=int, nargs=2, default=(2, 3), metavar=("VAL", "TEST"),
                   help="people held out from the healthy Measure cohort (14 people)")
    p.add_argument("--ephnogram_val_test", type=int, nargs=2, default=(3, 5), metavar=("VAL", "TEST"),
                   help="people held out from the EPHNOGRAM cohort (24 people)")
    p.add_argument("--cvd_val_test", type=int, nargs=2, default=(1, 2), metavar=("VAL", "TEST"),
                   help="people held out from the heart-disease cohort (10 people)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def is_valid(segment: dict) -> bool:
    for key in ("ecg", "ppg"):
        x = segment.get(key)
        if x is None:
            return False
        x = torch.as_tensor(x).flatten()
        if x.numel() != SOURCE_LEN or not torch.isfinite(x).all() or x.std() < 1e-6:
            return False
    return True


def process_file(job):
    """Load one .pt file, pick segments, downsample and scale. Runs in a worker process."""
    path, max_segments, seed = job
    segments = torch.load(path, map_location="cpu", weights_only=True)
    valid = [s for s in segments if isinstance(s, dict) and is_valid(s)]
    rng = random.Random(f"{seed}-{Path(path).name}")
    if max_segments and len(valid) > max_segments:
        valid = rng.sample(valid, max_segments)
    if not valid:
        return str(path), None, None, None, len(segments)
    ecg = np.stack([scale_to_unit(fix_ecg_polarity(downsample(torch.as_tensor(s["ecg"]).flatten().numpy())))
                    for s in valid])
    pcg = np.stack([scale_to_unit(downsample(torch.as_tensor(s["ppg"]).flatten().numpy())) for s in valid])
    return str(path), ecg, pcg, np.stack([padded_r_peaks(x) for x in ecg]), len(segments)


def padded_r_peaks(ecg: np.ndarray) -> np.ndarray:
    peaks = detect_r_peaks(ecg)[:MAX_BEATS]
    out = np.full(MAX_BEATS, -1, dtype=np.int16)
    out[: len(peaks)] = peaks
    return out


def build_manifest(args):
    healthy = sorted(Path(args.healthy).glob("*.pt"))
    cvd = sorted(Path(args.cvd).glob("*.pt"))
    if not healthy or not cvd:
        raise SystemExit(f"No .pt files found (healthy={len(healthy)}, cvd={len(cvd)})")
    ephnogram = load_ephnogram_subjects(args.ephnogram_meta)
    recordings = [identify(p, False, ephnogram) for p in healthy] + [identify(p, True, ephnogram) for p in cvd]
    splits = split_people(recordings, {"measure": tuple(args.measure_val_test),
                                       "ephnogram": tuple(args.ephnogram_val_test),
                                       "cvd": tuple(args.cvd_val_test)}, args.seed)
    assert_no_person_overlap(splits)
    return {split: {"files": [{"path": r.path, "is_cvd": r.is_cvd, "cohort": r.cohort, "person": r.person}
                              for r in recs],
                    "people": sorted({r.person for r in recs})}
            for split, recs in splits.items()}


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    (out / "manifest.json").write_text(json.dumps({"version": DATA_VERSION, "splits": manifest}, indent=2))

    for split, info in manifest.items():
        jobs = [(f["path"], 0 if f["is_cvd"] else args.healthy_per_file, args.seed) for f in info["files"]]
        cvd_flags = {f["path"]: f["is_cvd"] for f in info["files"]}
        person_of = {f["path"]: f["person"] for f in info["files"]}
        ecgs, pcgs, peaks, flags, recordings, people = [], [], [], [], [], []
        with ProcessPoolExecutor(args.workers) as pool:
            for i, (path, ecg, pcg, r_peaks, total) in enumerate(pool.map(process_file, jobs)):
                print(f"[{split}] {i + 1}/{len(jobs)} {Path(path).name}: kept {0 if ecg is None else len(ecg)}/{total}", flush=True)
                if ecg is None:
                    continue
                ecgs.append(ecg)
                pcgs.append(pcg)
                peaks.append(r_peaks)
                flags += [cvd_flags[path]] * len(ecg)
                recordings += [Path(path).stem] * len(ecg)
                people += [person_of[path]] * len(ecg)

        ecg_all = np.concatenate(ecgs).astype(np.float16)
        pcg_all = np.concatenate(pcgs).astype(np.float16)
        assert ecg_all.shape[1] == SEGMENT_LEN, ecg_all.shape
        np.savez(out / f"{split}.npz", ecg=ecg_all, pcg=pcg_all, r_peaks=np.concatenate(peaks),
                 is_cvd=np.array(flags), recording=np.array(recordings), person=np.array(people))
        print(f"[{split}] saved {len(ecg_all)} segments ({int(np.sum(flags))} CVD) "
              f"from {len(info['people'])} people: {info['people']}", flush=True)


if __name__ == "__main__":
    main()
