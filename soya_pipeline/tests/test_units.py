"""Fast unit tests (no PyTorch, no data)."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metrics import cluster_bootstrap_f1, ece_score, softmax
from schedules import lr_lambdas

ROOT = Path(__file__).resolve().parents[1]


def load_script(filename):
    """Import a script whose file name is not a valid module name (01_..., 08_...)."""
    name = "script_" + filename.split(".")[0]
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                                   # dataclasses look the module up here
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def manifest_mod():
    return load_script("01_build_manifest.py")


# ---- metrics --------------------------------------------------------------------------------
def test_softmax_rows_sum_to_one():
    p = softmax(np.random.default_rng(0).normal(size=(5, 4)) * 50)
    assert np.allclose(p.sum(1), 1) and (p >= 0).all()


def test_ece_perfect_and_overconfident():
    y = np.array([0, 1, 0, 1])
    perfect = np.eye(2)[y]
    assert ece_score(y, perfect) == pytest.approx(0.0)
    wrong = np.eye(2)[1 - y]                                  # confident and always wrong
    assert ece_score(y, wrong) == pytest.approx(1.0)


def test_cluster_bootstrap_brackets_point_estimate_and_is_deterministic():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    pred = np.where(rng.random(300) < 0.8, y, rng.integers(0, 3, 300))
    groups = rng.integers(0, 60, 300)
    a = cluster_bootstrap_f1(y, pred, groups, 200, seed=3)
    assert a == cluster_bootstrap_f1(y, pred, groups, 200, seed=3)
    from sklearn.metrics import f1_score
    assert a[0] <= f1_score(y, pred, average="macro") <= a[1]


# ---- LR schedule ----------------------------------------------------------------------------
def test_lr_schedule_shape():
    spe, epochs, warm = 30, 25, 2
    bb, hd = lr_lambdas(spe, epochs, warm)
    assert all(bb(s) == 0 for s in range(warm * spe))                       # frozen during warm-up
    assert 0 < bb(warm * spe) < 0.1                                         # then ramps up, no jump
    assert max(bb(s) for s in range(epochs * spe)) <= 1.0
    assert hd(0) < 0.2 and hd(spe) > 0.9                                    # head warms up fast
    assert bb(epochs * spe) < 1e-6 and hd(epochs * spe) < 1e-6              # both decay to ~0
    assert all(not math.isnan(f(s)) for f in (bb, hd) for s in range(epochs * spe + 5))


# ---- source grouping ------------------------------------------------------------------------
def test_every_image_is_its_own_group_unless_a_regex_says_otherwise(manifest_mod):
    assert manifest_mod.source_info("X/a.jpg", "X", None) == "X/X/a.jpg"
    assert manifest_mod.source_info("X/b.jpg", "X", None) != manifest_mod.source_info("X/a.jpg", "X", None)
    import re
    rx = re.compile(r"(plot\d+)_")
    g = manifest_mod.source_info("X/plot7_img3.jpg", "X", rx)
    assert g == "X/plot7" and g == manifest_mod.source_info("X/plot7_img9.jpg", "X", rx)   # same plot -> same group
    assert manifest_mod.source_info("X/other.jpg", "X", rx) == "X/X/other.jpg"            # no match -> own group


def test_union_find_merges_hash_links_and_groups(manifest_mod):
    H = np.zeros((4, 1), dtype=np.uint64)
    H[0, 0], H[1, 0] = 0, 1                                                  # 1 bit apart -> near-duplicates
    H[2, 0], H[3, 0] = 0xFFFFFFFF00000000, 0x00000000FFFFFFFF               # >= 32 bits from everything else
    groups = np.array(["x", "y", "z", "z"])                                  # 2 and 3 share a source group
    cl = manifest_mod.build_clusters(H, groups, thr=6)
    assert cl[0] == cl[1] and cl[2] == cl[3] and cl[0] != cl[2]


def test_fold_coverage_guard():
    import common
    rows = [("A", c, f) for c, f in [(0, 0), (1, 1), (2, 2), (3, 0), (4, 1), (5, 2)]]
    rows += [("B", c, f) for c, f in [(10, 0), (11, 1)]]                    # B: 2 clusters only
    df = pd.DataFrame(rows, columns=["label", "cluster", "fold"])
    with pytest.raises(SystemExit):                                         # fold 0 leaves B with 1 cluster outside
        common.check_fold_coverage(df, 3)
    ok = pd.DataFrame(rows[:6], columns=["label", "cluster", "fold"])
    common.check_fold_coverage(ok, 3)                                       # 4 clusters outside every fold: fine


def test_expand_paths(tmp_path):
    import common
    for n in ("a1.pt", "a2.pt", "b.pt"):
        (tmp_path / n).write_text("x")
    got = common.expand_paths([str(tmp_path / "a*.pt"), str(tmp_path / "b.pt"), str(tmp_path / "a1.pt")])
    assert [Path(g).name for g in got] == ["a1.pt", "a2.pt", "b.pt"]          # sorted, de-duplicated
    with pytest.raises(SystemExit):
        common.expand_paths([str(tmp_path / "zzz*.pt")])
    assert common.expand_paths(["plain/name.pt"]) == ["plain/name.pt"]         # no wildcard: passed through


def test_pick_checkpoints_explicit_lists_and_errors(tmp_path):
    import json
    from types import SimpleNamespace
    ev = load_script("08_evaluate.py")
    for stem, f1 in (("effb0_320", 0.70), ("effb0_320_cutmix", 0.75), ("convnext_320", 0.72)):
        (tmp_path / f"cnn_{stem}.pt").write_text("w")
        (tmp_path / f"cnn_{stem}.json").write_text(json.dumps({"img_size": 320, "val_macro_f1": f1}))
    (tmp_path / "cnn_other_224.pt").write_text("w")
    (tmp_path / "cnn_other_224.json").write_text(json.dumps({"img_size": 224, "val_macro_f1": 0.9}))
    P = SimpleNamespace(models=tmp_path)
    assert ev.pick_checkpoints(P, "best", 320) == ["effb0_320_cutmix"]
    assert sorted(ev.pick_checkpoints(P, "all", 320)) == ["convnext_320", "effb0_320", "effb0_320_cutmix"]   # 224 one excluded
    assert ev.pick_checkpoints(P, "effb0_320,convnext_320", 320) == ["effb0_320", "convnext_320"]
    with pytest.raises(SystemExit):
        ev.pick_checkpoints(P, "effb0_320,does_not_exist", 320)
    assert ev.pick_checkpoints(P, "best", 512) == []                           # nothing at that size


def test_reproduce_plan_resume_markers_and_copy_tree(tmp_path):
    rp = load_script("reproduce.py")
    recipe = {"dataset": "leaf", "img_size": 320, "folds": 2, "train_args": ["--epochs", "1"],
              "cnn": [{"model": "efficientnet_b0", "cutmix": 0.0, "extra": []},
                      {"model": "efficientnet_b0", "cutmix": 1.0, "extra": []}]}
    steps, tags = rp.plan(recipe, artifacts=str(tmp_path))
    assert tags == ["classical", "cnn_efficientnet_b0_320_tta", "cnn_efficientnet_b0_320_cutmix_tta", "cnn_ensemble2_tta"]
    train0 = [s for s in steps if "07_finetune_cnn.py" in s.cmd[1] and "0" == s.cmd[s.cmd.index("--fold") + 1]]
    assert [s.done for s in train0] == [["models/fold0/cnn_efficientnet_b0_320.json"],
                                        ["models/fold0/cnn_efficientnet_b0_320_cutmix.json"]]
    base = tmp_path / "leaf"
    assert not rp.is_done(train0[0], base)
    (base / "models" / "fold0").mkdir(parents=True)
    (base / "models" / "fold0" / "cnn_efficientnet_b0_320.json").write_text("{}")
    assert rp.is_done(train0[0], base) and not rp.is_done(train0[1], base)
    # copy_tree: incremental, never copies the cache
    (base / "cache").mkdir(); (base / "cache" / "all_x320.npy").write_text("big")
    dst = tmp_path / "sync"
    assert rp.copy_tree(base, dst) == 1 and not (dst / "cache").exists()
    assert rp.copy_tree(base, dst) == 0                                         # nothing changed -> nothing copied
