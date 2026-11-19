"""`tools/profile_training.py --mode torch` runs against the real engine.

It called `ClashRoyaleEnv()` with no decks, which the bindings refuse
(`TypeError: incompatible constructor arguments`), so the mode crashed before
measuring anything (docs/TODO.md L14). A one-step run is enough to cover the
constructor; it writes only into `tmp_path`.
"""
import os
import subprocess
import sys

import pytest

import python_ai

pytest.importorskip("clash_royale_env")


def test_torch_mode_runs_against_the_real_engine(tmp_path):
    tool = os.path.join(python_ai.REPO_ROOT, "python_ai", "tools", "profile_training.py")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run(
        [sys.executable, tool, "--mode", "torch", "--torch-steps", "1",
         "--torch-bwd", "1", "--out", str(tmp_path)],
        cwd=python_ai.REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "stubbed=False" in out.stdout      # the real engine, not the stub
    assert "one PPO update" in out.stdout     # got past the constructor to the end
