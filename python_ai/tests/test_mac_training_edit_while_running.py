"""Editing tools/mac_training.sh while a run is going must not change the running launcher.

bash reads a script as it executes it. The `_run` launcher waits on the trainer
for hours inside the script; when the trainer exits, bash reads on from the
byte where it stopped, so an edit made in place since then is executed from the
middle of a line. The script is one brace group ending in `exit`, which bash
parses whole before running any of it.

The test runs the launcher against a fake trainer that sleeps, appends a
command to the script while it waits, and checks the command never runs.

After the trainer exits, `_run` adopts any phase-2 trainer it finds with a
machine-wide pgrep, so these launchers get CLASH_P2_PATTERN set to a string
nothing matches; otherwise a real phase-2 run on this machine would hang them.
"""
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _launcher(tmp_path, trainer_seconds):
    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    (root / "venv" / "bin").mkdir(parents=True)
    script = root / "tools" / "mac_training.sh"
    shutil.copy(os.path.join(REPO, "tools", "mac_training.sh"), script)
    fake_python = root / "venv" / "bin" / "python"
    fake_python.write_text(f"#!/bin/bash\nsleep {trainer_seconds}\n")
    fake_python.chmod(0o755)
    (tmp_path / "logs").mkdir()
    env = dict(os.environ, CLASH_LOGDIR=str(tmp_path / "logs" / "run"),
               CLASH_WEIGHTS=str(tmp_path / "w.pth"),
               CLASH_P2_PATTERN=f"no-such-phase2-trainer-{tmp_path.name}")
    proc = subprocess.Popen(["bash", str(script), "_run"], env=env, cwd=root,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc, script, root


def test_an_edit_made_while_the_launcher_waits_never_runs(tmp_path):
    marker = tmp_path / "injected"
    launcher, script, root = _launcher(tmp_path, trainer_seconds=3)
    try:
        time.sleep(1.0)                          # the fake trainer is running
        assert launcher.poll() is None
        with open(script, "a") as f:             # an edit in place, as an editor makes
            f.write(f"\necho INJECTED > {marker}\n")
        out, _ = launcher.communicate(timeout=30)
    finally:
        subprocess.run(["pkill", "-f", str(root)])
    assert "trainer exited with code 0" in out
    assert not marker.exists(), f"the launcher ran text added after it started:\n{out}"


def test_a_test_launcher_ignores_a_phase2_trainer_elsewhere_on_the_machine(tmp_path):
    """A decoy whose command line names the phase-2 trainer must not hold the launcher."""
    decoy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                              "trainers/train_selfplay.py"])
    launcher, _, root = _launcher(tmp_path, trainer_seconds=1)
    try:
        out, _ = launcher.communicate(timeout=30)
    finally:
        decoy.kill()
        subprocess.run(["pkill", "-f", str(root)])
    assert "trainer exited with code 0" in out
