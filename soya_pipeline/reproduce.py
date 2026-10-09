#!/usr/bin/env python
"""Reproduce a published experiment end to end from a recipe in configs/recipes.json.

    python reproduce.py --list
    python reproduce.py leaf_ensemble3 --dry-run                 # print the commands only
    python reproduce.py leaf_ensemble3                           # run everything (GPU recommended)
    python reproduce.py leaf_ensemble3 --resume --sync-dir DIR   # survive a Colab disconnect (see below)
    python reproduce.py leaf_baseline --no-torch --img-size 64 --folds 3   # CPU smoke test, no PyTorch

Every step is one of the numbered scripts, called exactly as documented in the README. When the recipe finishes, a
reproduction report (source zip, commit if any, library versions, GPU, per-model CV summaries) is written to
artifacts/<dataset>/reports/cv/reproduction_<recipe>.json - compare it with docs/RESULTS.md.

--resume     skip steps whose outputs already exist (a CNN counts as finished only when its completion marker exists).
             Use it to CONTINUE the same run. For a different configuration (other recipe, --folds, --img-size) use a
             fresh --artifacts folder, otherwise old outputs are reused.
--sync-dir   copy results (everything except the image cache) to DIR/<dataset> after every step, and restore them from
             there at start. Point it at a Google Drive folder so a disconnected Colab run can be resumed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGES = ["numpy", "pandas", "scikit-learn", "scikit-image", "opencv-python-headless", "pillow", "joblib",
            "torch", "torchvision"]
CORE_MODULES = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "skimage": "scikit-image",
                "cv2": "opencv-python-headless", "PIL": "pillow", "joblib": "joblib", "matplotlib": "matplotlib"}
DATASET_DIRS = {"leaf": "Soyabean_Leaf_Image_Dataset"}   # same as common.py


@dataclass
class Step:
    cmd: list
    done: list = field(default_factory=list)      # glob patterns under artifacts/<dataset>; all must match = step finished


def ckpt_stem(model: str, size: int, cutmix: float) -> str:
    """Must match the run name built in 07_finetune_cnn.py."""
    return f"{model}_{size}" + ("_cutmix" if cutmix > 0 else "")


def plan(recipe: dict, size=None, folds=None, no_torch=False, skip_data=False, artifacts="artifacts"):
    """-> (list of Step, list of CV model tags to summarise)."""
    ds, size, K = recipe["dataset"], size or recipe["img_size"], folds or recipe["folds"]
    base = ["--dataset", ds, "--img-size", str(size), "--artifacts", artifacts]

    def step(script, *args, done=()):
        return Step([sys.executable, str(HERE / script), *base, *map(str, args)], list(done))

    steps = []
    if not skip_data:
        deep_done = [] if no_torch else ["features/all_deep_mobilenet_v3_large.npy.meta.json"]
        steps += [step("01_build_manifest.py", "--folds", K, done=["manifest.csv"]),
                  step("02_cache_images.py", done=[f"cache/all_x{size}.npy.meta.json"]),
                  step("03_eda.py", done=["eda/eda_summary.json"]),
                  step("04_extract_features.py", *(["--skip-deep"] if no_torch else []),
                       done=["features/all_hand.npy.meta.json", *deep_done])]
    steps.append(step("05_audit_split.py", done=["reports/main/audit.txt"]))

    members = recipe["cnn"]
    stems = [ckpt_stem(m["model"], size, m["cutmix"]) for m in members]
    for f in range(K):
        steps.append(step("06_train_classical.py", "--fold", f, *(["--feature-sets", "hand"] if no_torch else []),
                          done=[f"models/fold{f}/classical_best.joblib"]))
        steps.append(step("08_evaluate.py", "--fold", f, "--which", "classical",
                          done=[f"reports/fold{f}/classical_*_pred.csv"]))
        if no_torch:
            continue
        for m, stem in zip(members, stems):
            extra = (["--cutmix", m["cutmix"]] if m["cutmix"] > 0 else []) + m["extra"]
            steps.append(step("07_finetune_cnn.py", "--fold", f, "--model", m["model"], *recipe["train_args"], *extra,
                              done=[f"models/fold{f}/cnn_{stem}.json"]))
        for stem in stems:
            steps.append(step("08_evaluate.py", "--fold", f, "--which", "cnn", "--cnn", stem, "--tta",
                              done=[f"reports/fold{f}/cnn_{stem}_tta_pred.csv"]))
        if len(members) > 1:                          # explicit member list: stray checkpoints cannot change the ensemble
            steps.append(step("08_evaluate.py", "--fold", f, "--which", "cnn", "--cnn", ",".join(stems), "--tta",
                              done=[f"reports/fold{f}/cnn_ensemble{len(members)}_tta_pred.csv"]))

    tags = ["classical"]
    if not no_torch:
        tags += [f"cnn_{s}_tta" for s in stems]
        if len(members) > 1:
            tags.append(f"cnn_ensemble{len(members)}_tta")
    steps += [step("09_cv_summary.py", "--model-tag", t) for t in tags]       # cheap: always re-run
    return steps, tags


def preflight(recipe: dict, no_torch: bool):
    """Cheap checks before a long run -> (problems, warnings)."""
    problems, warnings = [], []
    data = Path(os.environ.get("SOYA_ROOT", "./data")) / DATASET_DIRS[recipe["dataset"]]
    if not data.is_dir():
        problems.append(f"dataset folder not found: {data}\n  set SOYA_ROOT to the folder that CONTAINS {DATASET_DIRS[recipe['dataset']]}/")
    for mod, pkg in CORE_MODULES.items():
        if importlib.util.find_spec(mod) is None:
            problems.append(f"missing package {pkg}  (pip install -r requirements-core.txt)")
    if not no_torch:
        for mod in ("torch", "torchvision"):
            if importlib.util.find_spec(mod) is None:
                problems.append(f"missing package {mod}  (pip install -r requirements.txt, or use --no-torch)")
        if not problems:
            import torch
            if not torch.cuda.is_available():
                warnings.append("no GPU detected: CNN training on CPU takes many hours. In Colab: Runtime > Change runtime type > T4 GPU.")
    return problems, warnings


def is_done(step: Step, base: Path) -> bool:
    return bool(step.done) and all(any(base.glob(p)) for p in step.done)


def copy_tree(src: Path, dst: Path) -> int:
    """Copy new / changed files from src to dst, skipping the image cache. Returns the number of files copied."""
    n = 0
    if not src.is_dir():
        return n
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if rel.parts and rel.parts[0] == "cache" or p.is_dir() or p.name.endswith(".partial.npy"):
            continue
        q = dst / rel
        s = p.stat()
        if q.exists() and q.stat().st_size == s.st_size and q.stat().st_mtime_ns >= s.st_mtime_ns:
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        n += 1
    return n


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": {}}
    for p in PACKAGES:
        try:
            env["packages"][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            env["packages"][p] = None
    try:
        import torch
        env["cuda"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        env["cuda"], env["gpu"] = None, None
    env["source"] = os.environ.get("SOYA_SOURCE")        # set by the notebook: name + md5 of the project zip it unpacked
    try:
        env["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True,
                                           check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        env["git_commit"] = None
    return env


def label(s: Step) -> str:
    return " ".join(Path(a).name if i == 1 else a for i, a in enumerate(s.cmd) if i != 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--no-torch", action="store_true", help="skip every PyTorch step (CPU smoke test)")
    ap.add_argument("--skip-data", action="store_true", help="reuse manifest, cache and features")
    ap.add_argument("--resume", action="store_true", help="skip steps whose outputs already exist")
    ap.add_argument("--sync-dir", default=None, help="back results up to DIR/<dataset> after every step; restore from it at start")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()

    recipes = json.loads((HERE / "configs" / "recipes.json").read_text())
    if args.list or not args.recipe:
        for name, r in recipes.items():
            print(f"{name:16s} {r['description']}")
        return
    if args.recipe not in recipes:
        raise SystemExit(f"unknown recipe '{args.recipe}'. Available: {', '.join(recipes)}")
    recipe = recipes[args.recipe]
    args.artifacts = str(Path(args.artifacts).resolve())              # relative to where you ran this, for every step
    steps, tags = plan(recipe, args.img_size, args.folds, args.no_torch, args.skip_data, args.artifacts)
    K = args.folds or recipe["folds"]
    base = Path(args.artifacts) / recipe["dataset"]

    if args.dry_run:
        for s in steps:
            print(label(s))
        return

    problems, warnings = preflight(recipe, args.no_torch)
    for w in warnings:
        print(f"WARNING: {w}")
    if problems:
        raise SystemExit("cannot start:\n- " + "\n- ".join(problems))

    sync = Path(args.sync_dir) / recipe["dataset"] if args.sync_dir else None
    if sync is not None:
        restored = copy_tree(sync, base) if sync.is_dir() else 0
        if restored:
            print(f"restored {restored} files from {sync}")
    if args.resume and (base / "manifest.csv").exists():
        import pandas as pd
        have = int(pd.read_csv(base / "manifest.csv", usecols=["fold"]).fold.max()) + 1
        if have != K:
            raise SystemExit(f"--resume: the existing manifest has {have} folds but this run wants {K}; "
                             "use a fresh --artifacts folder for a new configuration")

    t0, skipped = time.time(), 0
    for i, s in enumerate(steps, 1):
        if args.resume and is_done(s, base):
            skipped += 1
            print(f"\n[{i}/{len(steps)}] [skip: outputs exist] {label(s)}", flush=True)
            continue
        print(f"\n[{i}/{len(steps)}] {label(s)}", flush=True)
        subprocess.run(s.cmd, check=True)
        if sync is not None:
            copy_tree(base, sync)

    cv_dir = base / "reports" / "cv"
    results = {t: json.loads((cv_dir / f"{t}_cv_summary.json").read_text()) for t in tags}
    report = {"recipe": args.recipe, "no_torch": args.no_torch, "img_size": args.img_size or recipe["img_size"],
              "folds": K, "minutes": round((time.time() - t0) / 60, 1), "steps_skipped_by_resume": skipped,
              "environment": environment(), "results": results}
    out = cv_dir / f"reproduction_{args.recipe}.json"
    out.write_text(json.dumps(report, indent=2))
    if sync is not None:
        copy_tree(base, sync)
    print(f"\n{'model':32s} {'pooled macro-F1':>16s}  95% CI")
    for t, r in results.items():
        lo, hi = r["pooled_macro_f1_ci95"]
        print(f"{t:32s} {r['pooled_macro_f1']:16.4f}  {lo:.3f}-{hi:.3f}")
    print(f"\nreport -> {out}\nGPU training is not bit-exact: expect about +/-0.01 macro-F1 between reruns.")


if __name__ == "__main__":
    main()
