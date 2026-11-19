"""A multi-day unattended run must survive what happens to one:

  * a checkpoint replace fails with PermissionError on Windows while any handle holds the file;
  * one copy of the training state, no backup;
  * the plateau tracker must survive a resume;
  * phase 2 must follow CLASH_WEIGHTS / CLASH_LOGDIR, and a child that dies at startup must be noticed;
  * phase 2's opponent pool is this lineage's snapshots only.
"""
import os
import time

import pytest
import torch

from python_ai.rl import checkpointing as CK
from python_ai.rl.curriculum import CurriculumManager


# --- checkpoint writes ---

def test_a_save_survives_a_transient_permission_error(tmp_path, monkeypatch):
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(5, "Access is denied")
        return real(src, dst)

    monkeypatch.setattr(CK.os, "replace", flaky)
    monkeypatch.setattr(CK, "_REPLACE_BACKOFF_S", 0.0)
    path = tmp_path / "w.pth"
    CK.atomic_save({"x": 1}, str(path))
    assert torch.load(path)["x"] == 1 and calls["n"] == 3


def test_a_save_keeps_the_previous_generation(tmp_path):
    path = tmp_path / "w.pth"
    CK.atomic_save({"gen": 1}, str(path))
    CK.atomic_save({"gen": 2}, str(path), keep_previous=True)
    assert torch.load(path)["gen"] == 2
    assert torch.load(str(path) + ".prev")["gen"] == 1


def _save_generations(path, n, keep):
    for gen in range(1, n + 1):
        CK.atomic_save({"gen": gen}, str(path), keep_previous=True)
        CK.keep_in_history(str(path), gen * 250, keep)


def test_the_last_few_saves_are_kept_as_full_checkpoints(tmp_path):
    # 2026-11-19: a deck-pool switch restarted the run and later periodic
    # saves overwrote the only checkpoint from before it; `.prev` holds one
    # generation, a few minutes deep.
    path = tmp_path / "livesync.pth"
    _save_generations(path, 7, keep=3)
    hist = tmp_path / "livesync_history"
    assert sorted(os.listdir(hist)) == [
        "ep00001250.pth", "ep00001500.pth", "ep00001750.pth"]
    # Each generation keeps its own content after the live file moved on.
    for gen in (5, 6, 7):
        assert torch.load(hist / f"ep{gen * 250:08d}.pth")["gen"] == gen
    assert torch.load(path)["gen"] == 7


def test_keeping_none_writes_no_history(tmp_path):
    path = tmp_path / "w.pth"
    _save_generations(path, 3, keep=0)
    assert not (tmp_path / "w_history").exists()


def test_a_second_save_at_the_same_episode_replaces_that_generation(tmp_path):
    # The stop save can land on the episode of the last periodic one.
    path = tmp_path / "w.pth"
    CK.atomic_save({"gen": 1}, str(path))
    CK.keep_in_history(str(path), 500, 3)
    CK.atomic_save({"gen": 2}, str(path))
    CK.keep_in_history(str(path), 500, 3)
    assert os.listdir(tmp_path / "w_history") == ["ep00000500.pth"]
    assert torch.load(tmp_path / "w_history" / "ep00000500.pth")["gen"] == 2


def test_the_history_depth_is_a_setting():
    assert CK.keep_checkpoints({}) == CK.KEEP_CHECKPOINTS_DEFAULT
    assert CK.keep_checkpoints({"CLASH_KEEP_CHECKPOINTS": "0"}) == 0
    assert CK.keep_checkpoints({"CLASH_KEEP_CHECKPOINTS": "12"}) == 12
    # Where a run writes, not what it learns: a change is not a drift.
    assert "CLASH_KEEP_CHECKPOINTS" in CK.OPERATIONAL_SETTINGS


# --- curriculum tracking survives a resume ---

def _mgr():
    return CurriculumManager(entry_win_rate=0.60, min_stage_for_phase2=8,
                             phase2_win_rate_gate=0.80, max_episodes_per_deck=1250,
                             random_opponent_budget=5000)


def test_the_plateau_tracker_is_restored_not_reset():
    m = _mgr()
    m.stage, m.stage_start_episode = 3, 1000
    m.best_rung_mean, m.best_rung_episode = 0.61, 2400
    m.rung_entry_progress = 0.42
    for _ in range(300):
        m.note_outcome(1)
    state = m.state_dict()
    n = _mgr()
    n.load_state_dict(state)
    assert n.best_rung_episode == 2400
    assert n.best_rung_mean == pytest.approx(0.61)
    assert n.rung_entry_progress == pytest.approx(0.42)
    assert len(n.rung_history) == 300


# --- phase 2 follows phase 1's redirection ---

def test_phase2_bootstraps_from_the_checkpoint_phase1_actually_wrote(tmp_path, monkeypatch):
    from python_ai.trainers import train_selfplay as TS
    monkeypatch.setenv("CLASH_WEIGHTS", str(tmp_path / "arm" / "model_weights.pth"))
    monkeypatch.setenv("CLASH_LOGDIR", str(tmp_path / "runs" / "arm"))
    weights, bootstrap, logdir = TS.selfplay_paths()
    assert bootstrap == str(tmp_path / "arm" / "model_weights.pth")
    assert os.path.dirname(weights) == str(tmp_path / "arm")
    assert logdir.startswith(str(tmp_path / "runs"))


def test_phase2_defaults_are_unchanged_without_redirection(monkeypatch):
    from python_ai.trainers import train_selfplay as TS
    monkeypatch.delenv("CLASH_WEIGHTS", raising=False)
    monkeypatch.delenv("CLASH_LOGDIR", raising=False)
    weights, bootstrap, logdir = TS.selfplay_paths()
    assert weights == CK.weights_path("model_weights_selfplay.pth")
    assert bootstrap == CK.weights_path("model_weights.pth")
    assert logdir == CK.run_path("runs/clash_royale_selfplay")


def test_a_phase2_child_that_dies_at_startup_is_reported(tmp_path, monkeypatch):
    from python_ai.trainers import train as T

    class Dead:
        returncode = 1

        def __init__(self, *a, **k):
            pass

        def poll(self):
            return 1

    monkeypatch.setattr(T.subprocess, "Popen", Dead)
    with pytest.raises(RuntimeError, match="pipeline #2 exited"):
        T.launch_pipeline2(log_dir=str(tmp_path), wait_seconds=0.01)


# --- the opponent pool is this lineage's ---

def test_snapshots_from_a_previous_lineage_are_not_opponents(tmp_path):
    from python_ai.trainers import league
    old = tmp_path / "20260801-000000_pipeline1_ep02000.pth"
    new = tmp_path / "20260915-000000_pipeline1_ep02000.pth"
    for p in (old, new):
        torch.save({"x": 1}, p)
    start = time.time()
    os.utime(old, (start - 86400, start - 86400))
    os.utime(new, (start + 5, start + 5))
    found = league.discover_historical_checkpoints(directory=str(tmp_path), since=start)
    assert [os.path.basename(p) for p in found] == [new.name]
