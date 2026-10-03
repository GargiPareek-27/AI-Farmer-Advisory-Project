"""PyTorch helpers: backbones, embedding extraction, batched inference, checkpoints."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
HEAD_ATTR = {"mobilenet_v3_large": "classifier", "efficientnet_b0": "classifier",
             "resnet18": "fc", "resnet50": "fc"}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _base(name: str, pretrained: bool):
    if name not in HEAD_ATTR:
        raise SystemExit(f"Unsupported backbone {name}; choose from {sorted(HEAD_ATTR)}")
    return getattr(tvm, name)(weights="DEFAULT" if pretrained else None)


def build_model(name: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    m = _base(name, pretrained)
    if HEAD_ATTR[name] == "classifier":
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    else:
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def build_embedder(name: str, pretrained: bool = True) -> nn.Module:
    m = _base(name, pretrained)
    setattr(m, HEAD_ATTR[name], nn.Identity())
    return m


def split_backbone_head(model: nn.Module, name: str):
    head = getattr(model, HEAD_ATTR[name])
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    backbone_children = [c for c in model.children() if c is not head]
    return backbone, list(head.parameters()), backbone_children


@torch.no_grad()
def forward_batches(model, X, device, mean, std, bs: int = 128, tta: bool = False, amp: bool = True):
    """X: uint8 (N,H,W,3) array/memmap -> numpy outputs (logits or embeddings)."""
    model.eval()
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    outs = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(device)
        xb = xb.permute(0, 3, 1, 2).float().div_(255.0)
        xb = ((xb - mean_t) / std_t).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            o = model(xb)
            if tta:
                o = (o + model(xb.flip(3)) + model(xb.flip(2))) / 3
        outs.append(o.float().cpu())
    return torch.cat(outs).numpy()


def save_checkpoint(path, model, name, classes, img_size, mean, std, extra=None):
    torch.save({"arch": name, "state": model.state_dict(), "classes": classes,
                "img_size": img_size, "mean": mean, "std": std, **(extra or {})}, path)


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["arch"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state"])
    return model.to(device).to(memory_format=torch.channels_last).eval(), ck
