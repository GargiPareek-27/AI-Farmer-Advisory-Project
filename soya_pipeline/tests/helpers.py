"""Shared test helpers."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(script, *args, env_root, check=True):
    env = {**os.environ, "SOYA_ROOT": str(env_root)}
    return subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], cwd=ROOT, env=env,
                          capture_output=True, text=True, check=check)


def make_args(dataset, root, artifacts, size=64, fold=-1, seed=42):
    return argparse.Namespace(dataset=dataset, data_root=str(root), artifacts=str(artifacts), img_size=size,
                              fold=fold, seed=seed, backbone="mobilenet_v3_large")
