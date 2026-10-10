"""Train paired ECG <-> PCG translation generators.

Fixes over the original lab train.py:
  * ECG/PCG unpacked in the right order
  * discriminators get gradients from real samples (no torch.no_grad on real)
  * paired supervision (L1 + multi-resolution STFT + envelope), since the data is paired
  * no identity loss: it rewarded copying the input, which is what the old model learned
  * conditional discriminators, LSGAN loss, adversarial warm-up
  * validation once per epoch, best-checkpoint selection, resumable, time-limited
  * split by subject, fixed seed, so results are reproducible
  * v2: recording-balanced sampling, best checkpoint chosen per direction,
        never resumes from a checkpoint trained on a different dataset version
  * v3: extra ECG loss weight on heartbeat spikes, EMA of generator weights
        (validated, selected and shipped), final_ema.pt = the pre-chosen final model

Usage:
    python train.py --data_dir <folder with train.npz/val.npz> --out_dir <folder>
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from cardiosignal.data import PairedSignals, recording_weights
from cardiosignal.ema import EMA
from cardiosignal.inference import (DEFAULT_MODEL_CONFIG, build_generator, evaluate_generators,
                                    selection_score, summarize)
from cardiosignal.losses import envelope_loss, lsgan_d_loss, lsgan_g_loss, multi_resolution_stft_loss
from cardiosignal.models import Discriminator, count_parameters

# Validation metric used to pick each direction's best checkpoint.
DIRECTION_METRICS = {"g_e2p": ("ecg_to_pcg", "envelope_pearson"), "g_p2e": ("pcg_to_ecg", "pearson")}
GENERATORS = ("g_e2p", "g_p2e")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_l1", type=float, default=10.0)
    p.add_argument("--lambda_stft", type=float, default=2.0)
    p.add_argument("--lambda_env", type=float, default=10.0)
    p.add_argument("--lambda_cycle", type=float, default=5.0)
    p.add_argument("--lambda_adv", type=float, default=1.0)
    p.add_argument("--adv_warmup_epochs", type=int, default=3, help="epochs of supervised-only training first")
    p.add_argument("--time_limit_hours", type=float, default=11.0, help="stop before Kaggle's 12h session limit")
    p.add_argument("--peak_boost", type=float, default=4.0,
                   help="extra ECG L1 weight within +-40 ms of each R-peak (0 = off)")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--balance_power", type=float, default=0.5,
                   help="0 = natural sampling, 1 = every recording equally likely")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--max_train_batches", type=int, default=None, help="for quick smoke tests")
    p.add_argument("--max_val_batches", type=int, default=None, help="for quick smoke tests")
    return p.parse_args(argv)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_training_state(args, device):
    nets = {
        "g_e2p": build_generator(DEFAULT_MODEL_CONFIG).to(device),
        "g_p2e": build_generator(DEFAULT_MODEL_CONFIG).to(device),
        "d_pcg": Discriminator().to(device),   # judges (ecg, pcg) pairs
        "d_ecg": Discriminator().to(device),   # judges (pcg, ecg) pairs
    }
    betas = (0.5, 0.999)
    opts = {
        "g": torch.optim.AdamW(list(nets["g_e2p"].parameters()) + list(nets["g_p2e"].parameters()),
                               lr=args.lr, betas=betas, weight_decay=1e-4),
        "d": torch.optim.AdamW(list(nets["d_pcg"].parameters()) + list(nets["d_ecg"].parameters()),
                               lr=args.lr, betas=betas, weight_decay=1e-4),
    }
    scheds = {k: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.epochs, eta_min=args.lr * 0.05)
              for k, o in opts.items()}
    return nets, opts, scheds


def supervised_losses(fake, real, weight=None):
    fake, real = fake.float(), real.float()
    if weight is None:
        l1 = F.l1_loss(fake, real)
    else:
        l1 = (weight * (fake - real).abs()).sum() / weight.sum()
    return l1, multi_resolution_stft_loss(fake, real), envelope_loss(fake, real)


def train_step(batch, nets, opts, scaler, args, use_adv, device, amp):
    batch = [x.to(device, non_blocking=True) for x in batch]
    ecg, pcg = batch[0], batch[1]
    ecg_weight = batch[2] if len(batch) > 2 else None
    autocast = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp)

    # Discriminators
    d_loss = torch.zeros((), device=device)
    if use_adv:
        with autocast, torch.no_grad():
            fake_pcg, fake_ecg = nets["g_e2p"](ecg), nets["g_p2e"](pcg)
        with autocast:
            d_loss = (lsgan_d_loss(nets["d_pcg"](ecg, pcg).float(), nets["d_pcg"](ecg, fake_pcg).float())
                      + lsgan_d_loss(nets["d_ecg"](pcg, ecg).float(), nets["d_ecg"](pcg, fake_ecg).float()))
        opts["d"].zero_grad(set_to_none=True)
        scaler.scale(d_loss).backward()
        scaler.step(opts["d"])

    # Generators
    with autocast:
        fake_pcg, fake_ecg = nets["g_e2p"](ecg), nets["g_p2e"](pcg)
        cycle_ecg, cycle_pcg = nets["g_p2e"](fake_pcg), nets["g_e2p"](fake_ecg)
        adv = (lsgan_g_loss(nets["d_pcg"](ecg, fake_pcg).float()) + lsgan_g_loss(nets["d_ecg"](pcg, fake_ecg).float())
               if use_adv else torch.zeros((), device=device))
    l1_p, stft_p, env_p = supervised_losses(fake_pcg, pcg)
    l1_e, stft_e, env_e = supervised_losses(fake_ecg, ecg, ecg_weight)
    cycle = F.l1_loss(cycle_ecg.float(), ecg) + F.l1_loss(cycle_pcg.float(), pcg)
    g_loss = (args.lambda_l1 * (l1_p + l1_e) + args.lambda_stft * (stft_p + stft_e)
              + args.lambda_env * (env_p + env_e) + args.lambda_cycle * cycle + args.lambda_adv * adv)

    opts["g"].zero_grad(set_to_none=True)
    scaler.scale(g_loss).backward()
    scaler.unscale_(opts["g"])
    torch.nn.utils.clip_grad_norm_(list(nets["g_e2p"].parameters()) + list(nets["g_p2e"].parameters()), 1.0)
    scaler.step(opts["g"])
    scaler.update()

    return {"g_total": g_loss.item(), "d_total": d_loss.item(), "adv": adv.item(), "cycle": cycle.item(),
            "l1_pcg": l1_p.item(), "l1_ecg": l1_e.item(), "stft_pcg": stft_p.item(), "env_pcg": env_p.item()}


def new_best_tracker():
    return {k: {"score": float("-inf"), "epoch": None, "state": None} for k in DIRECTION_METRICS}


def update_best(best, nets, val_summary, epoch):
    """Keep a CPU copy of each generator at its own best validation epoch. Returns names that improved."""
    improved = []
    for name, (direction, metric) in DIRECTION_METRICS.items():
        score = val_summary[direction][metric]
        if score is not None and score > best[name]["score"]:
            state = {k: v.detach().cpu().clone() for k, v in nets[name].state_dict().items()}
            best[name] = {"score": score, "epoch": epoch + 1, "state": state}
            improved.append(name)
    return improved


def save_best(path, best):
    """Generators only (each from its own best epoch): the small file the app loads."""
    torch.save({"model_config": DEFAULT_MODEL_CONFIG,
                "epochs": {k: v["epoch"] for k, v in best.items()},
                "scores": {k: v["score"] for k, v in best.items()},
                **{k: v["state"] for k, v in best.items()}}, path)


def read_data_version(data_dir) -> str:
    manifest = Path(data_dir) / "manifest.json"
    if not manifest.exists():
        return "unknown"
    return json.loads(manifest.read_text()).get("version", "v1")


def save_generators(path, models: dict, epoch: int):
    """Generators-only checkpoint in the format load_generators() and the app expect."""
    torch.save({"model_config": DEFAULT_MODEL_CONFIG, "epochs": {k: epoch + 1 for k in GENERATORS},
                **{k: models[k].state_dict() for k in GENERATORS}}, path)


def save_checkpoint(path, nets, ema, opts, scheds, scaler, epoch, best, val_summary, args, data_version):
    torch.save({
        "epoch": epoch, "best": best, "val": val_summary, "args": vars(args),
        "data_version": data_version, "model_config": DEFAULT_MODEL_CONFIG,
        **{k: v.state_dict() for k, v in nets.items()},
        **{f"ema_{k}": v.state_dict() for k, v in ema.items()},
        "opt_g": opts["g"].state_dict(), "opt_d": opts["d"].state_dict(),
        "sched_g": scheds["g"].state_dict(), "sched_d": scheds["d"].state_dict(),
        "scaler": scaler.state_dict(),
    }, path)


def try_resume(path, nets, ema, opts, scheds, scaler, data_version):
    if not path.exists():
        return 0, new_best_tracker()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("data_version") != data_version or "best" not in ckpt:
        print(f"Ignoring {path}: trained on data '{ckpt.get('data_version', 'v1')}', "
              f"current data is '{data_version}'. Starting fresh.", flush=True)
        return 0, new_best_tracker()
    for k, net in nets.items():
        net.load_state_dict(ckpt[k])
    for k, averaged in ema.items():
        averaged.load_state_dict(ckpt.get(f"ema_{k}", ckpt[k]))
    opts["g"].load_state_dict(ckpt["opt_g"]); opts["d"].load_state_dict(ckpt["opt_d"])
    scheds["g"].load_state_dict(ckpt["sched_g"]); scheds["d"].load_state_dict(ckpt["sched_d"])
    scaler.load_state_dict(ckpt["scaler"])
    print(f"Resumed from {path} at epoch {ckpt['epoch'] + 2}", flush=True)
    return ckpt["epoch"] + 1, ckpt["best"]


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = PairedSignals(str(Path(args.data_dir) / "train.npz"), augment=True, peak_boost=args.peak_boost)
    val_ds = PairedSignals(str(Path(args.data_dir) / "val.npz"))
    sampler = WeightedRandomSampler(recording_weights(train_ds.recording, args.balance_power),
                                    num_samples=len(train_ds), replacement=True)
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, drop_last=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.num_workers > 0)

    data_version = read_data_version(args.data_dir)
    nets, opts, scheds = build_training_state(args, device)
    ema = {k: EMA(nets[k], args.ema_decay) for k in GENERATORS}
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    start_epoch, best = try_resume(out / "last.pt", nets, ema, opts, scheds, scaler, data_version)
    has_peaks = train_ds.r_peaks is not None and args.peak_boost > 0
    print(f"device={device} amp={amp} data={data_version} train={len(train_ds)} val={len(val_ds)} "
          f"recordings={len(set(train_ds.recording))} peak_weighting={has_peaks} ema={args.ema_decay} "
          f"generator_params={count_parameters(nets['g_e2p']):,}", flush=True)

    started, history_path = time.time(), out / "history.csv"
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        for net in nets.values():
            net.train()
        use_adv = epoch >= args.adv_warmup_epochs
        running = {}
        for i, batch in enumerate(loader):
            if args.max_train_batches is not None and i >= args.max_train_batches:
                break
            stats = train_step(batch, nets, opts, scaler, args, use_adv, device, amp)
            for k in GENERATORS:
                ema[k].update(nets[k])
            running = {k: running.get(k, 0.0) + v for k, v in stats.items()}
            if i % 100 == 0:
                print(f"  epoch {epoch + 1} step {i}/{len(loader)} " + " ".join(f"{k}={v:.3f}" for k, v in stats.items()), flush=True)
        for s in scheds.values():
            s.step()

        ema_models = {k: ema[k].model for k in GENERATORS}   # validate and ship the averaged weights
        val_summary = summarize(evaluate_generators(ema_models["g_e2p"], ema_models["g_p2e"], val_ds, device,
                                                    args.batch_size, args.max_val_batches))
        score = selection_score(val_summary)
        steps = max(1, min(len(loader), args.max_train_batches or len(loader)))
        row = {"epoch": epoch + 1, "minutes": round((time.time() - epoch_start) / 60, 2), "val_score": round(score, 4),
               **{f"train_{k}": round(v / steps, 4) for k, v in running.items()},
               **{f"val_{d}_{m}": v for d, ms in val_summary.items() for m, v in ms.items()}}
        write_header = not history_path.exists()
        with history_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        improved = update_best(best, ema_models, val_summary, epoch)
        if improved:
            save_best(out / "best.pt", best)
            (out / "best_val_metrics.json").write_text(json.dumps(
                {k: {"epoch": v["epoch"], "score": v["score"]} for k, v in best.items()}, indent=2))
        save_generators(out / "final_ema.pt", ema_models, epoch)
        save_checkpoint(out / "last.pt", nets, ema, opts, scheds, scaler, epoch, best, val_summary, args, data_version)
        print(f"epoch {epoch + 1}/{args.epochs} {row['minutes']} min | "
              f"ECG->PCG env r={val_summary['ecg_to_pcg']['envelope_pearson']} (best {best['g_e2p']['score']:.4f}) | "
              f"PCG->ECG r={val_summary['pcg_to_ecg']['pearson']} (best {best['g_p2e']['score']:.4f})"
              + (f" | saved best {', '.join(improved)}" if improved else ""), flush=True)

        elapsed_h = (time.time() - started) / 3600
        if elapsed_h + 1.5 * (time.time() - epoch_start) / 3600 > args.time_limit_hours:
            print(f"Stopping at time limit ({elapsed_h:.2f} h). Re-run the same command to resume.", flush=True)
            break


if __name__ == "__main__":
    main()
