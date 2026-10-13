"""Copy the trained model and a few test-set demo recordings into the backend.

Demo samples come from the *test* split (recordings not used in training), one
segment per recording: healthy recordings with a clear heart-sound rhythm plus
heart-disease (CVD) recordings, so the demo shows both strengths and limits.

Usage:
    python scripts/export_assets.py --checkpoint <generators .pt> --test_npz <test.npz>
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from cardiosignal.beats import PCG_QUALITY_THRESHOLD, periodicity
from cardiosignal.config import ENVELOPE_WINDOW, TARGET_FS
from cardiosignal.inference import load_generators
from cardiosignal.signal import envelope

BACKEND = Path(__file__).resolve().parent.parent
ENVELOPE_FS = TARGET_FS / (ENVELOPE_WINDOW // 2)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_npz", required=True)
    p.add_argument("--healthy", type=int, default=4)
    p.add_argument("--cvd", type=int, default=2)
    return p.parse_args()


def quality(pcg: np.ndarray) -> float:
    env = envelope(torch.from_numpy(pcg.astype(np.float32)).view(1, 1, -1)).flatten().numpy()
    return periodicity(env, ENVELOPE_FS)


def pick(data, want_cvd: bool, count: int):
    """One segment per *person* (falls back to per recording for data without person ids)."""
    owner = data["person"] if "person" in data else data["recording"]
    chosen, seen = [], set()
    order = np.argsort(owner, kind="stable")
    for i in order:
        key = str(owner[i])
        if key in seen or bool(data["is_cvd"][i]) != want_cvd:
            continue
        if not want_cvd and quality(data["pcg"][i]) < PCG_QUALITY_THRESHOLD:
            continue
        seen.add(key)
        chosen.append(int(i))
        if len(chosen) == count:
            break
    return chosen


def main():
    args = parse_args()
    load_generators(args.checkpoint)  # fail early if the checkpoint is not a generator file
    (BACKEND / "models").mkdir(exist_ok=True)
    shutil.copyfile(args.checkpoint, BACKEND / "models" / "model.pt")

    with np.load(args.test_npz) as data:
        data = {k: data[k] for k in data.files}
    healthy, cvd = pick(data, False, args.healthy), pick(data, True, args.cvd)
    idx = healthy + cvd
    labels = ["healthy"] * len(healthy) + ["cvd"] * len(cvd)
    numbers = list(range(1, len(healthy) + 1)) + list(range(1, len(cvd) + 1))
    descriptions = [f"Healthy patient {n} (unseen patient)" if lab == "healthy" else
                    f"Cardiovascular disease patient {n} (unseen patient)" for lab, n in zip(labels, numbers)]
    (BACKEND / "data").mkdir(exist_ok=True)
    np.savez_compressed(BACKEND / "data" / "samples.npz",
                        ids=np.array([f"{lab}-{n}" for lab, n in zip(labels, numbers)]),
                        labels=np.array(labels), descriptions=np.array(descriptions),
                        ecg=data["ecg"][idx].astype(np.float16), pcg=data["pcg"][idx].astype(np.float16))
    print(f"Exported model and {len(idx)} samples ({len(healthy)} healthy, {len(cvd)} CVD)")


if __name__ == "__main__":
    main()
