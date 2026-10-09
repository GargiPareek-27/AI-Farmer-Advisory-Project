#!/usr/bin/env python
"""Step 7 - Fast transfer-learning fine-tune of a pretrained CNN.

Speed : cached uint8 arrays, whole train set on the GPU, GPU-side augmentation, AMP, channels_last.
Leakage: augmentation touches TRAIN only; class weights come from TRAIN labels; early stopping
         and the best checkpoint use VAL; the TEST split is never loaded here.
LR schedule: head = short warm-up + cosine; backbone stays frozen for `--warmup-epochs`, then its
         LR ramps up linearly over one epoch before the cosine decay. (The previous one-cycle
         schedule hit its peak LR exactly when the backbone was unfrozen, and validation F1
         collapsed at that moment in early experiments.)
"""
import json
import math
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from common import BACKBONES, Paths, common_args, load_classes, load_xy, set_seed
from schedules import lr_lambdas
from models import (IMAGENET_MEAN, IMAGENET_STD, build_model, forward_batches, get_device,
                    save_checkpoint, split_backbone_head)


def gpu_augment(x: torch.Tensor) -> torch.Tensor:
    """x: float (B,3,H,W) in [0,1]. Flips, 90-degree rotation (square images only), brightness /
    contrast +-20 %, random zoom-in crop. Hue is untouched on purpose: colour is a disease cue."""
    B, dev = x.size(0), x.device
    for dim in (3, 2):
        m = torch.rand(B, device=dev) < 0.5
        x = torch.where(m[:, None, None, None], x.flip(dim), x)
    if x.size(2) == x.size(3):
        x = torch.rot90(x, random.randint(0, 3), (2, 3))
    b = 1 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.4
    c = 1 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.4
    mean = x.mean((1, 2, 3), keepdim=True)
    x = ((x - mean) * c + mean) * b
    s = torch.empty(B, device=dev).uniform_(0.75, 1.0)
    tx = (1 - s) * torch.empty(B, device=dev).uniform_(-1, 1)
    ty = (1 - s) * torch.empty(B, device=dev).uniform_(-1, 1)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 1, 1], theta[:, 0, 2], theta[:, 1, 2] = s, s, tx, ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="reflection", align_corners=False).clamp(0, 1)


def cutmix(x, y, alpha):
    """Paste a random box from a shuffled copy of the batch; returns x, y_a, y_b, lam."""
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    H, W = x.shape[2:]
    rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2, x1, x2 = max(cy - rh // 2, 0), min(cy + rh // 2, H), max(cx - rw // 2, 0), min(cx + rw // 2, W)
    x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
    return x, y, y[perm], 1 - (y2 - y1) * (x2 - x1) / (H * W)


def main():
    def extra(p):
        p.add_argument("--model", default="mobilenet_v3_large", choices=BACKBONES)
        p.add_argument("--epochs", type=int, default=25)
        p.add_argument("--warmup-epochs", type=int, default=2, help="epochs with frozen backbone")
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--lr-head", type=float, default=2e-3)
        p.add_argument("--lr-backbone", type=float, default=3e-4)
        p.add_argument("--patience", type=int, default=6, help="epochs without val improvement (after warm-up)")
        p.add_argument("--label-smoothing", type=float, default=0.1)
        p.add_argument("--no-class-weights", action="store_true")
        p.add_argument("--cutmix", type=float, default=0.0, help="CutMix Beta alpha (0 = off); try 1.0 on val")
        p.add_argument("--cutmix-prob", type=float, default=0.5)
        p.add_argument("--tag", default="", help="suffix for the checkpoint name (e.g. a second seed)")

    args = common_args(__doc__.splitlines()[0], extra)
    P = Paths(args)
    set_seed(args.seed)
    device = get_device()
    torch.backends.cudnn.benchmark = True
    amp = device.type == "cuda"
    classes = load_classes(P)
    K = len(classes)
    run = f"{args.model}_{args.img_size}" + ("_cutmix" if args.cutmix > 0 else "") + (f"_{args.tag}" if args.tag else "")

    Xtr, ytr, _ = load_xy(P, "train")
    Xva, yva, _ = load_xy(P, "val")
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    Xtr_t, ytr_t = torch.from_numpy(Xtr).to(device), torch.from_numpy(ytr).to(device)
    N = len(Xtr_t)

    model = build_model(args.model, K).to(device).to(memory_format=torch.channels_last)
    backbone, head, backbone_children = split_backbone_head(model, args.model)
    counts = np.bincount(ytr, minlength=K).astype(np.float64)
    w = np.ones(K) if args.no_class_weights else N / (K * np.maximum(counts, 1))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device),
                               label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr_backbone},
                             {"params": head, "lr": args.lr_head}], weight_decay=1e-4)
    spe = math.ceil(N / args.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, list(lr_lambdas(spe, args.epochs, args.warmup_epochs)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    hist, best_f1, best_loss, bad = [], -1.0, 1e9, 0
    ckpt = P.models / f"cnn_{run}.pt"
    done_marker = P.models / f"cnn_{run}.json"
    done_marker.unlink(missing_ok=True)              # written only when training finishes (used by reproduce.py --resume)
    for epoch in range(args.epochs):
        t0 = time.time()
        frozen = epoch < args.warmup_epochs
        for p_ in backbone:
            p_.requires_grad_(not frozen)
        model.train()
        if frozen:                                   # keep frozen BatchNorm statistics untouched
            for c in backbone_children:
                c.eval()
        perm = torch.randperm(N, device=device)
        run_loss, seen = 0.0, 0
        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            if len(idx) < 2:
                continue
            xb = gpu_augment(Xtr_t[idx].permute(0, 3, 1, 2).float().div_(255.0))
            yb = ytr_t[idx]
            mixed = args.cutmix > 0 and random.random() < args.cutmix_prob
            if mixed:
                xb, ya, yb2, lam = cutmix(xb, yb, args.cutmix)
            xb = ((xb - mean_t) / std_t).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, enabled=amp):
                out = model(xb)
                loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb2) if mixed else crit(out, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(idx)
            seen += len(idx)

        logits = forward_batches(model, Xva, device, IMAGENET_MEAN, IMAGENET_STD, amp=amp)
        val_loss = F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(yva)).item()
        val_f1 = f1_score(yva, logits.argmax(1), average="macro")
        val_acc = accuracy_score(yva, logits.argmax(1))
        hist.append(dict(epoch=epoch + 1, train_loss=run_loss / max(seen, 1), val_loss=val_loss,
                         val_acc=val_acc, val_macro_f1=val_f1, seconds=time.time() - t0, frozen=frozen))
        print(f"epoch {epoch + 1:02d}/{args.epochs} {'[head]' if frozen else '[full]'} "
              f"train_loss={hist[-1]['train_loss']:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} val_macroF1={val_f1:.4f} ({hist[-1]['seconds']:.1f}s)")

        if val_f1 > best_f1 + 1e-4 or (abs(val_f1 - best_f1) <= 1e-4 and val_loss < best_loss):
            best_f1, best_loss, bad = val_f1, val_loss, 0
            save_checkpoint(ckpt, model, args.model, classes, args.img_size, IMAGENET_MEAN, IMAGENET_STD,
                            extra={"epoch": epoch + 1, "val_macro_f1": val_f1})
        elif not frozen:
            bad += 1
            if bad >= args.patience:
                print(f"early stopping (no val improvement for {args.patience} epochs)")
                break

    done_marker.write_text(json.dumps(
        dict(arch=args.model, img_size=args.img_size, val_macro_f1=best_f1, val_loss=best_loss,
             cutmix=args.cutmix, epochs_run=len(hist))))
    h = pd.DataFrame(hist)
    h.to_csv(P.reports / f"cnn_{run}_history.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set_title("loss"); ax[0].legend()
    ax[1].plot(h.epoch, h.val_macro_f1, label="val macro-F1"); ax[1].plot(h.epoch, h.val_acc, label="val acc")
    ax[1].set_title("validation"); ax[1].legend()
    plt.tight_layout(); plt.savefig(P.reports / f"cnn_{run}_curves.png", dpi=130); plt.close()
    print(f"\nBest val macro-F1 = {best_f1:.4f} -> {ckpt}  (test set untouched)")


if __name__ == "__main__":
    main()
