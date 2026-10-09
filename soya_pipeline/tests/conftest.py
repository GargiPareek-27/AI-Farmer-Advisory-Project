import subprocess
import sys

import pytest

from helpers import ROOT


@pytest.fixture(scope="session")
def dummy_root(tmp_path_factory):
    out = tmp_path_factory.mktemp("dummy")
    subprocess.run([sys.executable, str(ROOT / "make_dummy_data.py"), "--out", str(out), "--per-class", "60"],
                   cwd=ROOT, check=True, capture_output=True)
    return out
