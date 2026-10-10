"""End-to-end smoke test: tiny fake dataset -> train 2 epochs -> resume -> evaluate."""
import json

import numpy as np
import torch

import evaluate
import train
from cardiosignal.data import PairedSignals
from cardiosignal.inference import load_generators

LEN = 1200


def _write_split(path, n, rng):
    t = np.linspace(0, 6, LEN)
    ecg = np.stack([np.sin(2 * np.pi * (1 + rng.random()) * t) for _ in range(n)]) * 0.9
    pcg = np.sign(ecg) * ecg ** 2
    r_peaks = np.full((n, 32), -1, dtype=np.int16)
    r_peaks[:, :3] = [100, 500, 900]
    np.savez(path, ecg=ecg.astype(np.float16), pcg=pcg.astype(np.float16), r_peaks=r_peaks,
             is_cvd=np.arange(n) % 2 == 0, recording=np.array([f"rec{i // 2}" for i in range(n)]))


def test_dataset_shapes(tmp_path):
    _write_split(tmp_path / "train.npz", 4, np.random.default_rng(0))
    ds = PairedSignals(str(tmp_path / "train.npz"), augment=True)
    ecg, pcg = ds[0]
    assert ecg.shape == (1, LEN) and pcg.shape == (1, LEN) and ecg.dtype == torch.float32
    assert ecg.abs().max() <= 1.0


def test_dataset_peak_weights(tmp_path):
    _write_split(tmp_path / "train.npz", 2, np.random.default_rng(0))
    ecg, _, weight = PairedSignals(str(tmp_path / "train.npz"), peak_boost=4.0)[0]
    assert weight.shape == ecg.shape
    assert weight[0, 100] == 5.0 and weight[0, 300] == 1.0


def test_train_resume_evaluate(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 8), ("val", 4), ("test", 4)):
        _write_split(tmp_path / f"{split}.npz", n, rng)
    out = tmp_path / "run"
    common = ["--data_dir", str(tmp_path), "--out_dir", str(out), "--batch_size", "4",
              "--num_workers", "0", "--adv_warmup_epochs", "1"]

    train.main(common + ["--epochs", "1"])
    train.main(common + ["--epochs", "2"])  # resumes from last.pt and runs epoch 2

    history = (out / "history.csv").read_text().strip().splitlines()
    assert len(history) == 3  # header + 2 epochs
    assert (out / "best.pt").exists() and (out / "last.pt").exists() and (out / "final_ema.pt").exists()
    g_final, _ = load_generators(str(out / "final_ema.pt"))
    assert sum(p.numel() for p in g_final.parameters()) > 0

    g_e2p, g_p2e = load_generators(str(out / "best.pt"))
    with torch.no_grad():
        assert g_e2p(torch.zeros(1, 1, LEN)).shape == (1, 1, LEN)

    best = torch.load(out / "best.pt", weights_only=True)
    assert set(best["epochs"]) == {"g_e2p", "g_p2e"}
    assert all(1 <= e <= 2 for e in best["epochs"].values())

    evaluate.main(["--checkpoint", str(out / "best.pt"), "--data_dir", str(tmp_path),
                   "--out_dir", str(out / "eval"), "--num_plots", "1"])
    report = json.loads((out / "eval" / "test_metrics.json").read_text())
    assert {"all", "healthy", "cvd"} <= set(report)
    assert "ecg_to_pcg" in report["all"]
    assert (out / "eval" / "sample_0.png").exists()


def test_resume_ignored_when_data_version_changes(tmp_path, capsys):
    rng = np.random.default_rng(1)
    for split, n in (("train", 8), ("val", 4)):
        _write_split(tmp_path / f"{split}.npz", n, rng)
    out = tmp_path / "run"
    common = ["--data_dir", str(tmp_path), "--out_dir", str(out), "--batch_size", "4", "--num_workers", "0"]

    (tmp_path / "manifest.json").write_text(json.dumps({"version": "old"}))
    train.main(common + ["--epochs", "1"])
    (tmp_path / "manifest.json").write_text(json.dumps({"version": "new"}))
    train.main(common + ["--epochs", "1"])

    assert "Starting fresh" in capsys.readouterr().out
    assert len((out / "history.csv").read_text().strip().splitlines()) == 3  # header + 1 epoch per run
