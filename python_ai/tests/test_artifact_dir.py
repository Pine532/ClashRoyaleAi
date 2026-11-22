"""CLASH_ARTIFACT_DIR moves a run's shared pools out of the repository root.

    historical_checkpoints/   phase 2's PFSP pool (every phase-1 run adds to it)
    stage_checkpoints/        one snapshot per rung change
    replays/                  demo replays

All three are shared by every run on the machine. The user's rule (2026-11-22): an A/B arm writes
them under its own runs/ab_*/ folder, never into the live run's pools, so nothing shared needs
cleaning up afterwards. The setting is read at import, so each case runs in a fresh interpreter; and
each checks the destination before anything is written, so the old code fails without touching the
real pool.
"""
import json
import os
import subprocess
import sys

import python_ai
from python_ai.rl import checkpointing

_PROBE = r"""
import json, os, sys
from python_ai.rl import base_trainer, checkpointing as c
print(json.dumps({"historical": c.HISTORICAL_CHECKPOINT_DIR, "stage": c.STAGE_CHECKPOINT_DIR,
                  "replays": getattr(c, "REPLAY_DIR", None)}))
mode = sys.argv[1]
if mode == "snapshot":
    # Refuse before writing if the pool is not the arm's: the old code would write the real pool.
    if not c.HISTORICAL_CHECKPOINT_DIR.startswith(os.environ["ARM_ROOT"]):
        sys.exit("historical pool is not under the arm's folder; refusing to write")
    import torch
    path = c.save_historical_snapshot(torch.nn.Linear(1, 1), 7, "pipeline1")
    print(json.dumps({"saved": path}))
elif mode == "replay":
    # The trainer's own replay method, with the recorder captured instead of run.
    import types
    from python_ai.rl import replay
    seen = []
    replay.record_greedy_replay = lambda net, env, device, path: seen.append(path)
    stub = types.SimpleNamespace(build_replay_env=lambda: object(), net=None, device="cpu",
                                 replay_prefix="replay", episodes_completed=11)
    base_trainer.BaseTrainer.record_replay(stub)
    print(json.dumps({"replay": seen[0]}))
"""


def _probe(mode, arm_dir=None, tmp_path=None):
    env = {k: v for k, v in os.environ.items() if k != "CLASH_ARTIFACT_DIR"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if arm_dir is not None:
        env["CLASH_ARTIFACT_DIR"] = arm_dir
        env["ARM_ROOT"] = str(tmp_path)
    r = subprocess.run([sys.executable, "-c", _PROBE, mode], cwd=python_ai.REPO_ROOT, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    out = {}
    for line in r.stdout.splitlines():
        if line.startswith("{"):
            out.update(json.loads(line))
    return out


def test_unset_keeps_every_pool_at_the_repository_root():
    """The control: without the setting nothing moves (the live run relies on it)."""
    out = _probe("paths")
    root = python_ai.REPO_ROOT
    assert out["historical"] == os.path.join(root, "historical_checkpoints")
    assert out["stage"] == os.path.join(root, "stage_checkpoints")
    assert out["replays"] == os.path.join(root, "replays")


def test_an_arm_folder_receives_all_three_pools(tmp_path):
    arm = str(tmp_path / "ab_test" / "armB")
    out = _probe("paths", arm, tmp_path)
    assert out["historical"] == os.path.join(arm, "historical_checkpoints")
    assert out["stage"] == os.path.join(arm, "stage_checkpoints")
    assert out["replays"] == os.path.join(arm, "replays")


def test_a_historical_snapshot_lands_in_the_arm_folder_and_not_the_shared_pool(tmp_path):
    """Behavioural: the snapshot call with its default directory, as the trainer makes it."""
    arm = str(tmp_path / "ab_test" / "armB")
    out = _probe("snapshot", arm, tmp_path)
    assert out["saved"].startswith(os.path.join(arm, "historical_checkpoints") + os.sep)
    assert os.path.exists(out["saved"])
    # Episode 7 is this test's; a live run may add its own snapshots meanwhile.
    shared = checkpointing.run_path("historical_checkpoints")
    leaked = [n for n in (os.listdir(shared) if os.path.isdir(shared) else [])
              if n.endswith("_pipeline1_ep00000007.pth")]
    assert not leaked, leaked


def test_the_trainers_replay_goes_to_the_arm_folder(tmp_path):
    arm = str(tmp_path / "ab_test" / "armB")
    out = _probe("replay", arm, tmp_path)
    assert out["replay"] == os.path.join(arm, "replays", "replay_ep11.json")


def test_a_relative_setting_is_anchored_on_the_repository_root_not_the_cwd(tmp_path):
    out = _probe("paths", "runs/ab_example/armB", tmp_path)
    assert out["historical"] == os.path.join(python_ai.REPO_ROOT, "runs", "ab_example", "armB",
                                             "historical_checkpoints")


def test_the_setting_is_operational_so_a_resume_does_not_flag_it():
    changed, operational = checkpointing.settings_drift(
        {}, {"CLASH_ARTIFACT_DIR": "/tmp/ab_x/armB"})
    assert not changed
    assert [k for k, _, _ in operational] == ["CLASH_ARTIFACT_DIR"]
