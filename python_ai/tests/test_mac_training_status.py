"""tools/mac_training.sh reports the non-finite counters, which must stay 0.

`status` and the hourly `row` read the current log through the pointer file
next to CLASH_LOGDIR, so a temporary CLASH_LOGDIR isolates the test from the
live run. Two lines count (docs/TODO.md L11): an update that dropped a
minibatch to a non-finite gradient, and a sampled action whose log-prob was not
finite.
"""
import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(REPO, "tools", "mac_training.sh")

SKIP = ("  [WARN] 4 non-finite minibatch gradient(s) dropped this update. The weights are intact, "
        "but something upstream produced NaN/inf -- check the reward stream and the PPO ratio.")
SAMPLE = ("  [WARN] non-finite log-prob for a sampled action (env 5, episode 107454): card arm 4 "
          "log-prob 0.0, cell 370 log-prob -inf, abilities log-prob None. ...")
PROGRESS = "Episodes: 110300 | Avg(50): 1.51 | W/L/D: 0.45/0.55/0.00 | Decisive: 0.45 | Stage: 3"


def _run(tmp_path, command, lines):
    log = tmp_path / "fake_run.log"
    log.write_text("\n".join(lines) + "\n")
    (tmp_path / "fake.current_log").write_text(str(log) + "\n")
    env = dict(os.environ, CLASH_LOGDIR=str(tmp_path / "fake"),
               CLASH_WEIGHTS=str(tmp_path / "w.pth"))
    out = subprocess.run(["bash", SCRIPT, command], env=env, capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.mark.parametrize("lines, skips, samples", [
    ([PROGRESS], 0, 0),
    ([SKIP, PROGRESS, SKIP, SAMPLE], 2, 1),
])
def test_status_reports_the_counters_and_warns_on_any(tmp_path, lines, skips, samples):
    out = _run(tmp_path, "status", lines)
    assert "not running" in out
    assert (f"non-finite: {skips} update(s) with dropped minibatches, "
            f"{samples} bad sample(s) in this log") in out
    assert ("WARNING: non-finite counts must be 0" in out) == bool(skips or samples)


@pytest.mark.parametrize("lines, cells", [
    ([PROGRESS], ["0", "0", "ok"]),
    ([SKIP, SAMPLE, PROGRESS, SKIP, "x [FLOOR] y", "[DEMOTED] z"], ["2", "1", "**WARNING**"]),
])
def test_the_hourly_row_carries_the_counters(tmp_path, lines, cells):
    header = _run(tmp_path, "row-header", lines).splitlines()
    row = _run(tmp_path, "row", lines).strip()
    # A cell's own pipes are escaped (\|), as GitHub tables need.
    def cells_of(line):
        return [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
    names, values = cells_of(header[0]), cells_of(row)
    assert len(values) == len(names), (header[0], row)
    got = dict(zip(names, values))
    assert got["Trainer"] == "stopped"
    assert got["Last progress line"] == "`" + PROGRESS[:80].replace("|", "\\|") + "`"
    assert [got["Skip updates"], got["Bad samples"], got["Non-finite"]] == cells
    if len(lines) > 1:
        assert (got["[FLOOR]"], got["Demotions"]) == ("1", "1")
