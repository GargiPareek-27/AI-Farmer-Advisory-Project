#!/usr/bin/env python
"""Step 6 - Fast transfer-learning fine-tune of a small pretrained CNN.

Speed
  * Reads the cached uint8 arrays (no JPEG decoding, no DataLoader workers); the whole train
    set lives on the GPU and augmentation runs on the GPU.
  * Mixed precision (AMP), channels_last, OneCycle LR, head-only warm-up epochs, early stopping.
Leakage
  * Augmentation touches TRAIN only; class weights come from TRAIN labels only.
  * Early stopping / best checkpoint use VAL. The TEST split is never loaded here.
"""
import math
import time
import json
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from common import Paths, common_args, load_classes, load_xy, set_seed
from models import (IMAGENET_MEAN, IMAGENET_STD, build_model, forward_batches, get_device,
                    save_checkpoint, split_backbone_head)


def gpu_augment(x: torch.Tensor) -> torch.Tensor:
    """x: float (B,3,H,W) in [0,1]. Mild geometry + photometric jitter (hue is untouched on
    purpose: leaf colour is a disease cue)."""
    B, dev = x.size(0), x.device
    m = torch.rand(B, device=dev) < 0.5
    x = torch.where(m[:, None, None, None], x.flip(3), x)
    m = torch.rand(B, device=dev) < 0.5
    x = torch.where(m[:, None, None, None], x.flip(2), x)
    x = torch.rot90(x, random.randint(0, 3), (2, 3))
    # brightness / contrast +-20 %
    b = 1 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.4
    c = 1 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.4
    mean = x.mean((1, 2, 3), keepdim=True)
    x = ((x - mean) * c + mean) * b
    # random zoom-in crop (scale 0.75-1.0) + shift
    s = torch.empty(B, device=dev).uniform_(0.75, 1.0)
    tx = (1 - s) * torch.empty(B, device=dev).uniform_(-1, 1)
    ty = (1 - s) * torch.empty(B, device=dev).uniform_(-1, 1)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 1, 1], theta[:, 0, 2], theta[:, 1, 2] = s, s, tx, ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    return x.clamp(0, 1)


def main():
    def extra(p):
        import argparse
        p.add_argument("--model", default="mobilenet_v3_large",
                       choices=["mobilenet_v3_large", "efficientnet_b0", "resnet18", "resnet50"])
        p.add_argument("--epochs", type=int, default=15)
        p.add_argument("--warmup-epochs", type=int, default=2, help="epochs with frozen backbone")
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--lr-head", type=float, default=2e-3)
        p.add_argument("--lr-backbone", type=float, default=3e-4)
        p.add_argument("--patience", type=int, default=4)
        p.add_argument("--label-smoothing", type=float, default=0.1)
        p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
        p.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)

    args = common_args(__doc__.splitlines()[0], extra)
    P = Paths(args)
    set_seed(args.seed)
    device = get_device()
    torch.backends.cudnn.benchmark = True
    amp = device.type == "cuda"
    classes = load_classes(P)
    K = len(classes)

    Xtr, ytr, _ = load_xy(P, "train", mmap=False)
    Xva, yva, _ = load_xy(P, "val", mmap=False)
    if args.pretrained:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        st = json.loads(P.stats_json.read_text())
        mean, std = st["mean"], st["std"]
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)

    Xtr_t = torch.from_numpy(np.ascontiguousarray(Xtr)).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    N = len(Xtr_t)

    model = build_model(args.model, K, args.pretrained).to(device).to(memory_format=torch.channels_last)
    backbone, head, backbone_children = split_backbone_head(model, args.model)

    counts = np.bincount(ytr, minlength=K).astype(np.float64)
    w = N / (K * np.maximum(counts, 1)) if args.class_weights else np.ones(K)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device),
                               label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr_backbone},
                             {"params": head, "lr": args.lr_head}], weight_decay=1e-4)
    steps_per_epoch = math.ceil(N / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr_backbone, args.lr_head], total_steps=args.epochs * steps_per_epoch, pct_start=0.15)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    hist, best_f1, best_loss, bad = [], -1.0, 1e9, 0
    ckpt_path = P.models / "cnn_best.pt"
    for epoch in range(args.epochs):
        t0 = time.time()
        frozen = epoch < args.warmup_epochs
        for p_ in backbone:
            p_.requires_grad_(not frozen)
        model.train()
        if frozen:                       # keep frozen BatchNorm statistics untouched
            for c in backbone_children:
                c.eval()
        perm = torch.randperm(N, device=device)
        run_loss, seen = 0.0, 0
        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            if len(idx) < 2:
                continue
            xb = Xtr_t[idx].permute(0, 3, 1, 2).float().div_(255.0)
            xb = gpu_augment(xb)
            xb = ((xb - mean_t) / std_t).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = crit(model(xb), ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(idx)
            seen += len(idx)

        logits = forward_batches(model, Xva, device, mean, std, amp=amp)
        pred = logits.argmax(1)
        val_loss = F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(yva)).item()
        val_f1 = f1_score(yva, pred, average="macro")
        val_acc = accuracy_score(yva, pred)
        hist.append(dict(epoch=epoch + 1, train_loss=run_loss / max(seen, 1), val_loss=val_loss,
                         val_acc=val_acc, val_macro_f1=val_f1, seconds=time.time() - t0, frozen=frozen))
        print(f"epoch {epoch + 1:02d}/{args.epochs} {'[head]' if frozen else '[full]'} "
              f"train_loss={hist[-1]['train_loss']:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} val_macroF1={val_f1:.4f} ({hist[-1]['seconds']:.1f}s)")

        if val_f1 > best_f1 + 1e-4 or (abs(val_f1 - best_f1) <= 1e-4 and val_loss < best_loss):
            best_f1, best_loss, bad = val_f1, val_loss, 0
            save_checkpoint(ckpt_path, model, args.model, classes, args.img_size, mean, std,
                            extra={"epoch": epoch + 1, "val_macro_f1": val_f1})
        elif not frozen:
            bad += 1
            if bad >= args.patience:
                print(f"early stopping (no val improvement for {args.patience} epochs)")
                break

    h = pd.DataFrame(hist)
    h.to_csv(P.reports / "cnn_history.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set_title("loss"); ax[0].legend()
    ax[1].plot(h.epoch, h.val_macro_f1, label="val macro-F1"); ax[1].plot(h.epoch, h.val_acc, label="val acc")
    ax[1].set_title("validation"); ax[1].legend()
    plt.tight_layout(); plt.savefig(P.reports / "cnn_curves.png", dpi=130); plt.close()
    print(f"\nBest val macro-F1 = {best_f1:.4f} -> {ckpt_path}  (test set untouched)")


if __name__ == "__main__":
    main()
