"""Editing tools/mac_training.sh while a run is going must not change the running launcher.

bash reads a script as it executes it. The `_run` launcher waits on the trainer
for hours inside the script; when the trainer exits, bash reads on from the
byte where it stopped, so an edit made in place since then is executed from the
middle of a line. The script is one brace group ending in `exit`, which bash
parses whole before running any of it.

The test runs the launcher against a fake trainer that sleeps, appends a
command to the script while it waits, and checks the command never runs.
"""
import os
import shutil
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_an_edit_made_while_the_launcher_waits_never_runs(tmp_path):
    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    (root / "venv" / "bin").mkdir(parents=True)
    script = root / "tools" / "mac_training.sh"
    shutil.copy(os.path.join(REPO, "tools", "mac_training.sh"), script)
    fake_python = root / "venv" / "bin" / "python"
    fake_python.write_text("#!/bin/bash\nsleep 3\n")
    fake_python.chmod(0o755)
    (tmp_path / "logs").mkdir()
    marker = tmp_path / "injected"
    env = dict(os.environ, CLASH_LOGDIR=str(tmp_path / "logs" / "run"),
               CLASH_WEIGHTS=str(tmp_path / "w.pth"))

    launcher = subprocess.Popen(["bash", str(script), "_run"], env=env, cwd=root,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(1.0)                          # the fake trainer is running
    assert launcher.poll() is None
    with open(script, "a") as f:             # an edit in place, as an editor makes
        f.write(f"\necho INJECTED > {marker}\n")
    out, _ = launcher.communicate(timeout=30)
    assert "trainer exited with code 0" in out
    assert not marker.exists(), f"the launcher ran text added after it started:\n{out}"
