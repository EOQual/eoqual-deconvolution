"""Les exemples s'exécutent (petite image, budget réduit)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.slow
@pytest.mark.parametrize("name", ["quickstart", "tuning", "blind"])
def test_example_runs(name, tmp_path):
    env = dict(os.environ, EOQUAL_EXAMPLES_OUT=str(tmp_path), EOQUAL_EXAMPLES_SIZE="128",
               EOQUAL_EXAMPLES_N_CALLS="4")
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join([src, env.get("PYTHONPATH", "")])
    proc = subprocess.run([sys.executable, str(EXAMPLES / f"{name}.py")], env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert any(tmp_path.glob("*.html"))


def test_benchmark_compiles():
    compile((EXAMPLES / "benchmark.py").read_text(encoding="utf-8"), "benchmark.py", "exec")
