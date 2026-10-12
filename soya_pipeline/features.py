"""Hand-crafted features tuned for crop-health imagery (colour, vegetation indices, texture)."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def hand_features(img: np.ndarray) -> dict:
    """img: uint8 RGB (H, W, 3) -> ordered dict {feature_name: value}."""
    f: dict[str, float] = {}
    rgb = img.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    for name, ch in zip("rgb", (r, g, b)):
        f[f"rgb_{name}_mean"], f[f"rgb_{name}_std"] = ch.mean(), ch.std()
    for space, arr in (("hsv", hsv), ("lab", lab)):
        for c, nm in enumerate(space):
            ch = arr[..., c].astype(np.float32) / 255.0
            f[f"{space}_{nm}_mean"], f[f"{space}_{nm}_std"] = ch.mean(), ch.std()

    for nm, c, bins, rng in (("h", 0, 12, (0, 180)), ("s", 1, 8, (0, 256)), ("v", 2, 8, (0, 256))):
        hist = np.histogram(hsv[..., c], bins=bins, range=rng)[0] / hsv[..., c].size
        for i, v in enumerate(hist):
            f[f"hsv_{nm}_hist{i}"] = v

    eps = 1e-6
    idx = {
        "exg": 2 * g - r - b,
        "exr": 1.4 * r - g,
        "vari": np.clip((g - r) / (g + r - b + eps), -1, 1),
        "gli": (2 * g - r - b) / (2 * g + r + b + eps),
    }
    for nm, v in idx.items():
        f[f"{nm}_mean"], f[f"{nm}_std"] = v.mean(), v.std()
        f[f"{nm}_p10"], f[f"{nm}_p90"] = np.percentile(v, 10), np.percentile(v, 90)

    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    f["frac_green"] = ((h >= 35) & (h <= 85) & (s > 40)).mean()
    f["frac_yellow"] = ((h >= 15) & (h < 35) & (s > 40)).mean()
    f["frac_brown"] = ((h >= 5) & (h < 25) & (s > 60) & (v < 150)).mean()
    f["frac_dark"] = (v < 60).mean()

    q = (gray // 8).astype(np.uint8)  # 32 grey levels
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        vals = graycoprops(glcm, prop).mean(1)
        for d, val in zip((1, 3), vals):
            f[f"glcm_{prop}_d{d}"] = val

    lbp = local_binary_pattern(gray, 8, 1, "uniform").astype(np.int64)
    for i, val in enumerate(np.bincount(lbp.ravel(), minlength=10) / lbp.size):
        f[f"lbp_{i}"] = val
    f["lap_var_log"] = np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var())
    f["edge_density"] = (cv2.Canny(gray, 100, 200) > 0).mean()
    return f


def hand_vector(img: np.ndarray) -> np.ndarray:
    v = np.fromiter(hand_features(img).values(), dtype=np.float32)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def feature_names(img: np.ndarray) -> list[str]:
    return list(hand_features(img).keys())
