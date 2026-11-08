"""Les exemples s'exécutent (petite image, budget réduit)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.slow
@pytest.mark.parametrize("name", ["quickstart", "tuning", "blind", "blind_optics", "tutorial"])
def test_example_runs(name, tmp_path):
    env = dict(os.environ, EOQUAL_EXAMPLES_OUT=str(tmp_path), EOQUAL_EXAMPLES_SIZE="128",
               EOQUAL_EXAMPLES_N_CALLS="4")
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join([src, env.get("PYTHONPATH", "")])
    proc = subprocess.run([sys.executable, str(EXAMPLES / f"{name}.py")], env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert any(tmp_path.glob("*.html"))


@pytest.mark.parametrize("name", ["benchmark", "blind_optics_benchmark", "wiener_hunt_benchmark", "figures"])
def test_benchmark_compiles(name):
    compile((EXAMPLES / f"{name}.py").read_text(encoding="utf-8"), f"{name}.py", "exec")
