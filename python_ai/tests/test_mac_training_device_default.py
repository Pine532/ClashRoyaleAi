"""tools/mac_training.sh starts the trainer on MPS where the Mac has it; CLASH_DEVICE=cpu opts out.

The script runs against a fake interpreter: asked the MPS probe (`-c CODE`, two
arguments) it answers as FAKE_MPS says; launched as the trainer it records the
CLASH_DEVICE it was given and exits. CLASH_P2_PATTERN matches nothing, so the
launcher cannot adopt a real phase-2 trainer on this machine and linger.
"""
import os
import shutil
import subprocess
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAKE = """#!/bin/bash
if [ $# -eq 2 ]; then [ "${FAKE_MPS:-0}" = 1 ]; exit $?; fi
echo "device=${CLASH_DEVICE:-}" > "$FAKE_OUT"
"""


@pytest.mark.parametrize("mps, preset, want", [
    ("1", None, "mps"),        # the Mac has MPS: the default
    ("0", None, ""),           # no MPS: unset, so the trainer picks (CPU here)
    ("1", "cpu", "cpu"),       # the opt-out
])
def test_start_picks_mps_where_available(tmp_path, mps, preset, want):
    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    (root / "venv" / "bin").mkdir(parents=True)
    shutil.copy(os.path.join(REPO, "tools", "mac_training.sh"), root / "tools")
    fake = root / "venv" / "bin" / "python"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    out = tmp_path / "launched"
    env = {k: v for k, v in os.environ.items() if k != "CLASH_DEVICE"}
    env.update(FAKE_MPS=mps, FAKE_OUT=str(out), CLASH_LOGDIR=str(tmp_path / "logs" / "run"),
               CLASH_WEIGHTS=str(tmp_path / "w.pth"),
               CLASH_P2_PATTERN=f"no-such-phase2-trainer-{tmp_path.name}")
    if preset is not None:
        env["CLASH_DEVICE"] = preset
    try:
        subprocess.run(["bash", str(root / "tools" / "mac_training.sh"), "start"], env=env,
                       cwd=root, capture_output=True, text=True, timeout=60, check=True)
        for _ in range(100):
            if out.exists() and out.read_text().strip():
                break
            time.sleep(0.1)
    finally:
        subprocess.run(["pkill", "-f", str(root)])    # the launcher and its caffeinate
    assert out.read_text().strip() == f"device={want}"
