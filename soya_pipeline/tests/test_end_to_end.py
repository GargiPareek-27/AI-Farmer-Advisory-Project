"""End-to-end run of the real scripts on synthetic data (CPU, no PyTorch): reproduce.py --no-torch."""
import json

import numpy as np
import pandas as pd
import pytest

import common
from helpers import make_args, run_script

SIZE = 64


def reproduce(recipe, ds_root, artifacts, folds):
    return run_script("reproduce.py", recipe, "--no-torch", "--img-size", SIZE, "--folds", folds,
                      "--artifacts", artifacts, env_root=ds_root)


@pytest.fixture(scope="session")
def leaf_run(dummy_root, tmp_path_factory):
    art = tmp_path_factory.mktemp("art_leaf")
    reproduce("leaf_baseline", dummy_root, art, 5)
    return art


def manifest(art, ds):
    return pd.read_csv(art / ds / "manifest.csv", keep_default_na=False)


@pytest.mark.slow
def test_no_group_cluster_or_file_crosses_splits_or_folds(leaf_run):
    m = manifest(leaf_run, "leaf")
    for key in ("cluster", "group", "md5"):
        assert (m.groupby(key).split.nunique() == 1).all(), key
        assert (m.groupby(key).fold.nunique() == 1).all(), key
    assert set(m.split) == {"train", "val", "test"}
    assert (pd.crosstab(m.label, m.split) > 0).all().all()


@pytest.mark.slow
def test_leaf_duplicates_are_removed_and_flipped_copies_are_clustered(leaf_run):
    dropped = pd.read_csv(leaf_run / "leaf" / "dropped.csv")
    assert (dropped.reason == "exact duplicate").sum() >= 6          # one exact copy per class in the dummy set
    m = manifest(leaf_run, "leaf")
    for label in m.label.unique():
        g = m[m.label == label].set_index("relpath")
        assert g.loc[f"{label}/img_001.jpg", "cluster"] == g.loc[f"{label}/img_flipped_copy.jpg", "cluster"]


@pytest.mark.slow
def test_fold_splits_partition_the_data(leaf_run, dummy_root):
    for f in range(5):
        P = common.Paths(make_args("leaf", dummy_root, leaf_run, fold=f))
        m = common.load_manifest(P)
        assert set(m.split) == {"train", "val", "test"}
        assert (m.split == "test").sum() == (m.fold == f).sum()
        assert (m.groupby("cluster").split.nunique() == 1).all()
        assert (pd.crosstab(m.label, m.split).loc[:, "val"] > 0).all()


@pytest.mark.slow
def test_cv_summary_and_reproduction_report(leaf_run):
    cv = leaf_run / "leaf" / "reports" / "cv"
    s = json.loads((cv / "classical_cv_summary.json").read_text())
    assert s["n"] == len(manifest(leaf_run, "leaf"))                # every image is in exactly one test fold
    assert 0.0 <= s["pooled_macro_f1"] <= 1.0
    lo, hi = s["pooled_macro_f1_ci95"]
    assert lo <= hi
    rep = json.loads((cv / "reproduction_leaf_baseline.json").read_text())
    assert rep["environment"]["python"] and "classical" in rep["results"]
    assert "source" in rep["environment"]                           # filled by the notebook via SOYA_SOURCE


@pytest.mark.slow
def test_every_image_is_tested_exactly_once(leaf_run):
    parts = [pd.read_csv(p) for p in sorted((leaf_run / "leaf" / "reports").glob("fold*/classical_*_pred.csv"))]
    allrows = pd.concat(parts)
    assert len(parts) == 5
    assert allrows.relpath.is_unique and len(allrows) == len(manifest(leaf_run, "leaf"))


@pytest.mark.slow
def test_stale_cache_is_refused(leaf_run, dummy_root):
    P = common.Paths(make_args("leaf", dummy_root, leaf_run))
    common.load_xy(P, "train")                                      # fresh cache loads
    original = P.manifest.read_text()
    try:
        P.manifest.write_text(original + "\n")                      # any manifest change invalidates the fingerprint
        assert common.meta_state(P, P.x_path) == "stale"
        with pytest.raises(SystemExit):
            common.load_xy(P, "train")
    finally:
        P.manifest.write_text(original)
    assert common.meta_state(P, P.x_path) == "ok"


@pytest.mark.slow
def test_rerunning_manifest_clears_stale_artifacts(dummy_root, tmp_path):
    reproduce("leaf_baseline", dummy_root, tmp_path, 3)
    assert any((tmp_path / "leaf" / "features").glob("all_hand.npy"))
    run_script("01_build_manifest.py", "--dataset", "leaf", "--artifacts", tmp_path, "--img-size", SIZE,
               "--folds", 3, env_root=dummy_root)
    assert not list((tmp_path / "leaf" / "cache").glob("all_*"))
    assert not list((tmp_path / "leaf" / "features").glob("all_*"))


def test_dry_run_lists_expected_commands(dummy_root):
    out = run_script("reproduce.py", "leaf_ensemble3", "--dry-run", env_root=dummy_root).stdout.splitlines()
    assert sum("07_finetune_cnn.py" in l for l in out) == 15        # 3 models x 5 folds
    members = "efficientnet_b0_320,efficientnet_b0_320_cutmix,convnext_tiny_320"
    assert sum(f"--cnn {members} --tta" in l for l in out) == 5     # explicit ensemble, once per fold
    assert any("--cutmix 1.0" in l for l in out) and any("--lr-backbone 1e-4" in l for l in out)
    assert out[-1].startswith("09_cv_summary.py") and "cnn_ensemble3_tta" in out[-1]


@pytest.mark.slow
def test_audit_report_is_saved(leaf_run):
    text = (leaf_run / "leaf" / "reports" / "main" / "audit.txt").read_text()
    assert text.startswith("split = main") and "(a) test: 0.0%" in text and "(c) mean-RGB-only probe" in text


@pytest.mark.slow
def test_cache_matches_direct_decoding_and_leaves_no_partial_file(leaf_run, dummy_root):
    P = common.Paths(make_args("leaf", dummy_root, leaf_run))
    m = common.load_manifest(P)
    X = np.load(P.x_path, mmap_mode="r")
    assert X.shape == (len(m), SIZE, SIZE, 3) and X.dtype == np.uint8
    for i in (0, len(m) // 2, len(m) - 1):
        assert np.array_equal(X[i], common.load_resized(P.img_dir / m.relpath[i], SIZE))
    assert not list(P.cache.glob("*.partial.npy"))


@pytest.mark.slow
def test_resume_and_sync_survive_a_lost_runtime(dummy_root, tmp_path):
    art, sync = tmp_path / "art", tmp_path / "sync"
    args = ("leaf_baseline", "--no-torch", "--img-size", SIZE, "--folds", 3, "--artifacts", art, "--sync-dir", sync)
    run_script("reproduce.py", *args, env_root=dummy_root)
    assert (sync / "leaf" / "manifest.csv").exists() and (sync / "leaf" / "models" / "fold0" / "classical_best.joblib").exists()
    assert not (sync / "leaf" / "cache").exists()                   # the image cache is never synced
    ref = json.loads((art / "leaf" / "reports" / "cv" / "classical_cv_summary.json").read_text())

    import shutil
    shutil.rmtree(art)                                              # the Colab runtime was reset
    second = run_script("reproduce.py", *args, "--resume", env_root=dummy_root)
    assert "restored" in second.stdout
    assert second.stdout.count("[skip: outputs exist]") >= 8        # manifest, eda, features, audit, per-fold training/eval
    assert "02_cache_images.py" in second.stdout and "[skip: outputs exist] 02_cache_images.py" not in second.stdout
    again = json.loads((art / "leaf" / "reports" / "cv" / "classical_cv_summary.json").read_text())
    assert again["pooled_macro_f1"] == ref["pooled_macro_f1"]       # same folds, same models: identical result
    rep = json.loads((art / "leaf" / "reports" / "cv" / "reproduction_leaf_baseline.json").read_text())
    assert rep["steps_skipped_by_resume"] >= 8


def test_resume_refuses_a_different_fold_count(dummy_root, tmp_path):
    reproduce("leaf_baseline", dummy_root, tmp_path, 3)
    r = run_script("reproduce.py", "leaf_baseline", "--no-torch", "--img-size", SIZE, "--folds", 5, "--resume",
                   "--artifacts", tmp_path, env_root=dummy_root, check=False)
    assert r.returncode != 0 and "fresh --artifacts" in (r.stderr + r.stdout)


def test_preflight_reports_missing_dataset(tmp_path):
    r = run_script("reproduce.py", "leaf_baseline", "--no-torch", "--artifacts", tmp_path / "a",
                   env_root=tmp_path / "nothing", check=False)
    assert r.returncode != 0 and "dataset folder not found" in (r.stderr + r.stdout)
    assert not (tmp_path / "a" / "leaf" / "manifest.csv").exists()  # failed fast, before any work


def test_unknown_recipe_and_removed_uav_recipe_are_rejected(dummy_root):
    r = run_script("reproduce.py", "uav_cv", env_root=dummy_root, check=False)
    assert r.returncode != 0 and "unknown recipe" in (r.stderr + r.stdout)
