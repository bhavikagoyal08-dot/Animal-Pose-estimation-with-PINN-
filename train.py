"""
train.py — PINN Training Loop for Animal Kingdom Pose Estimation
================================================================
Trains PoseCNN with the combined PINN loss for 50 epochs.

Features:
    - Warm-up phase: encoder frozen for first 5 epochs (head-only training)
    - Cosine LR annealing after warm-up
    - Per-epoch logging of all loss components + PCK + jitter
    - Best model checkpointing (lowest val loss)
    - Loss curve saved to training_log.csv

Usage:
    python train.py                        # default settings
    python train.py --epochs 100           # longer training
    python train.py --batch_size 8         # larger batch
    python train.py --no_angle             # skip angle loss
    python train.py --lam_bone 0.3         # tune bone weight
"""

import os
import csv
import time
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import AnimalKingdomTripletDataset, NUM_KEYPOINTS
from model import PoseCNN, soft_argmax, BONES, ANGLE_TRIPLETS
from losses import pinn_loss, pck_metric, jitter_metric, estimate_bone_lengths


# ── Argument parsing ──────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="Train PINN Animal Kingdom pose estimator")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--img_size",      type=int,   default=256)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--lam_bone",      type=float, default=0.5)
    p.add_argument("--lam_smooth",    type=float, default=0.1)
    p.add_argument("--lam_angle",     type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int,   default=5,
                   help="Epochs to train head-only before unfreezing encoder")
    p.add_argument("--val_ratio",     type=float, default=0.2,
                   help="Fraction of triplets held out for validation")
    p.add_argument("--no_angle",      action="store_true",
                   help="Disable the joint angle loss term")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_file",       default="training_log.csv")
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--num_workers",    type=int,  default=0,
                   help="DataLoader workers (0 = main process, safe on Windows)")
    p.add_argument("--bone_estimate_n", type=int, default=2000,
                   help="Number of entries used to estimate bone lengths")
    p.add_argument("--max_triplets", type=int, default=None,
               help="Cap dataset size for quick iteration (e.g. 500)")
    return p.parse_args()


# ── One training epoch ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimiser, bone_lengths, joint_triples,
                    lam_bone, lam_smooth, lam_angle, device):
    model.train()
    totals    = {"loss_total": 0, "loss_data": 0,
                 "loss_bone": 0, "loss_smooth": 0, "loss_angle": 0}
    n_batches = 0

    for batch in loader:
        t0         = batch["img_t0"].to(device)
        t1         = batch["img_t1"].to(device)
        t2         = batch["img_t2"].to(device)
        gt_coords  = batch["kp_t1"].to(device)
        visibility = batch["vis_t1"].to(device)

        all_frames = torch.cat([t0, t1, t2], dim=0)
        all_coords = soft_argmax(model(all_frames))
        c0, c1, c2 = all_coords.chunk(3, dim=0)

        loss, log = pinn_loss(
            c0, c1, c2,
            gt_coords, visibility,
            bone_lengths,
            joint_triples=joint_triples,
            lam_bone=lam_bone,
            lam_smooth=lam_smooth,
            lam_angle=lam_angle,
        )

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimiser.step()

        for k in totals:
            totals[k] += log.get(k, 0.0)
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ── One validation epoch ──────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, bone_lengths, joint_triples,
             lam_bone, lam_smooth, lam_angle, device):
    model.eval()
    totals     = {"loss_total": 0, "loss_data": 0,
                  "loss_bone": 0, "loss_smooth": 0, "loss_angle": 0}
    pck_sum    = 0.0
    jitter_sum = 0.0
    n_batches  = 0

    for batch in loader:
        t0         = batch["img_t0"].to(device)
        t1         = batch["img_t1"].to(device)
        t2         = batch["img_t2"].to(device)
        gt_coords  = batch["kp_t1"].to(device)
        visibility = batch["vis_t1"].to(device)

        all_frames = torch.cat([t0, t1, t2], dim=0)
        all_coords = soft_argmax(model(all_frames))
        c0, c1, c2 = all_coords.chunk(3, dim=0)

        _, log = pinn_loss(
            c0, c1, c2,
            gt_coords, visibility,
            bone_lengths,
            joint_triples=joint_triples,
            lam_bone=lam_bone,
            lam_smooth=lam_smooth,
            lam_angle=lam_angle,
        )

        pck_sum    += pck_metric(c1, gt_coords, visibility, threshold=0.05)
        jitter_sum += jitter_metric(c0, c1)

        for k in totals:
            totals[k] += log.get(k, 0.0)
        n_batches += 1

    metrics = {k: v / max(n_batches, 1) for k, v in totals.items()}
    metrics["pck"]    = pck_sum    / max(n_batches, 1)
    metrics["jitter"] = jitter_sum / max(n_batches, 1)
    return metrics


# ── Pretty printer ────────────────────────────────────────────────────────────
def fmt(metrics, prefix=""):
    parts = [
        f"loss={metrics['loss_total']:.4f}",
        f"data={metrics['loss_data']:.4f}",
        f"bone={metrics['loss_bone']:.4f}",
        f"smooth={metrics['loss_smooth']:.5f}",
    ]
    if metrics.get("loss_angle", 0.0) > 0:
        parts.append(f"angle={metrics['loss_angle']:.4f}")
    if "pck" in metrics:
        parts.append(f"PCK={metrics['pck']:.1f}%")
    if "jitter" in metrics:
        parts.append(f"jitter={metrics['jitter']:.5f}")
    return prefix + "  ".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device               : {device}")
    print(f"Epochs               : {args.epochs}")
    print(f"Batch size           : {args.batch_size}")
    print(f"Image size           : {args.img_size}x{args.img_size}")
    print(f"Learning rate        : {args.lr}")
    print(f"Warm-up epochs       : {args.warmup_epochs}")
    print(f"Val ratio            : {args.val_ratio}")
    print(f"lam_bone / lam_smooth: {args.lam_bone} / {args.lam_smooth}")
    print(f"Angle loss           : {'disabled' if args.no_angle else f'enabled (lam={args.lam_angle})'}")

    # ── Dataset ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading dataset...")
    full_ds = AnimalKingdomTripletDataset(
    split="train",
    img_size=(args.img_size, args.img_size),
    max_triplets=args.max_triplets,)

    # Train / val split on triplets
    n_total = len(full_ds)
    n_val   = int(n_total * args.val_ratio)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"\nTrain triplets       : {n_train}")
    print(f"Val triplets         : {n_val}")

    # Estimate bone lengths from a subset of entries (not triplets)
    print(f"\nEstimating bone lengths (from {args.bone_estimate_n} entries)...")
    bone_lengths  = estimate_bone_lengths(
        full_ds.entries[:args.bone_estimate_n], BONES
    )
    joint_triples = None if args.no_angle else ANGLE_TRIPLETS

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True,  num_workers=args.num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=False
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Building model...")
    model = PoseCNN(
        num_keypoints=NUM_KEYPOINTS,
        pretrained=True,
        freeze_encoder=(args.warmup_epochs > 0),
    ).to(device)

    total, trainable = model.count_parameters()
    print(f"Total parameters     : {total:,}")
    print(f"Trainable (phase 1)  : {trainable:,}  (head only)")

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimiser = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    n_post_warmup = max(1, args.epochs - args.warmup_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_post_warmup, eta_min=1e-5
    )

    # ── Logging setup ─────────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_fields = [
        "epoch", "phase",
        "train_loss", "train_data", "train_bone", "train_smooth", "train_angle",
        "val_loss",   "val_data",   "val_bone",   "val_smooth",   "val_angle",
        "val_pck", "val_jitter", "lr", "epoch_time_s",
    ]
    with open(args.log_file, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_val   = float("inf")
    best_pck   = 0.0
    best_epoch = 0

    # ── Training loop ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Starting training...\n")

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()

        # Phase transition: unfreeze encoder after warm-up
        if epoch == args.warmup_epochs + 1:
            model.unfreeze_encoder()
            # Encoder gets 0.01x LR to avoid disrupting pretrained weights
            optimiser = optim.Adam([
                {"params": model.encoder.parameters(), "lr": args.lr * 0.01},
                {"params": list(model.up1.parameters()) +
                           list(model.up2.parameters()) +
                           list(model.up3.parameters()) +
                           list(model.heatmap_conv.parameters()),
                 "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=n_post_warmup, eta_min=1e-5
            )
            _, trainable = model.count_parameters()
            print(f"  -> Encoder unfrozen. Trainable: {trainable:,}\n")

        phase = "warmup" if epoch <= args.warmup_epochs else "train"

        train_metrics = train_one_epoch(
            model, train_loader, optimiser, bone_lengths, joint_triples,
            args.lam_bone, args.lam_smooth, args.lam_angle, device
        )
        val_metrics = validate(
            model, val_loader, bone_lengths, joint_triples,
            args.lam_bone, args.lam_smooth, args.lam_angle, device
        )

        if epoch > args.warmup_epochs:
            scheduler.step()
        current_lr = optimiser.param_groups[0]["lr"]
        epoch_time = time.time() - t_start

        print(f"Epoch {epoch:3d}/{args.epochs}  [{phase:6s}]  "
              f"lr={current_lr:.2e}  ({epoch_time:.1f}s)")
        print(f"  Train: {fmt(train_metrics)}")
        print(f"  Val:   {fmt(val_metrics)}")

        # Checkpoint by val loss
        if val_metrics["loss_total"] < best_val:
            best_val = val_metrics["loss_total"]

        # Checkpoint by PCK
        if val_metrics["pck"] > best_pck:
            best_pck   = val_metrics["pck"]
            best_epoch = epoch
            ckpt_path  = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optim_state":  optimiser.state_dict(),
                "val_loss":     val_metrics["loss_total"],
                "val_pck":      val_metrics["pck"],
                "val_jitter":   val_metrics["jitter"],
                "args":         vars(args),
                "bone_lengths": bone_lengths,
            }, ckpt_path)
            print(f"  ** Best saved -> {ckpt_path}  "
                  f"(PCK={best_pck:.1f}%  val_loss={best_val:.4f})")

        # Periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_metrics["loss_total"],
                "val_pck":     val_metrics["pck"],
            }, os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pt"))

        # CSV log
        with open(args.log_file, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                "epoch":        epoch,
                "phase":        phase,
                "train_loss":   round(train_metrics["loss_total"],       6),
                "train_data":   round(train_metrics["loss_data"],        6),
                "train_bone":   round(train_metrics["loss_bone"],        6),
                "train_smooth": round(train_metrics["loss_smooth"],      6),
                "train_angle":  round(train_metrics.get("loss_angle", 0), 6),
                "val_loss":     round(val_metrics["loss_total"],         6),
                "val_data":     round(val_metrics["loss_data"],          6),
                "val_bone":     round(val_metrics["loss_bone"],          6),
                "val_smooth":   round(val_metrics["loss_smooth"],        6),
                "val_angle":    round(val_metrics.get("loss_angle", 0),  6),
                "val_pck":      round(val_metrics["pck"],                2),
                "val_jitter":   round(val_metrics["jitter"],             6),
                "lr":           round(current_lr,                        8),
                "epoch_time_s": round(epoch_time,                        1),
            })
        print()

    # ── Done ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Training complete.")
    print(f"Best PCK             : {best_pck:.1f}%  (epoch {best_epoch})")
    print(f"Best val loss        : {best_val:.4f}")
    print(f"Checkpoint           : {os.path.join(args.checkpoint_dir, 'best_model.pt')}")
    print(f"Training log         : {args.log_file}")
    print("\nNext step: run evaluate.py")


if __name__ == "__main__":
    main()