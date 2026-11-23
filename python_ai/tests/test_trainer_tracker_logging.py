"""Tracker/* scalars reach TensorBoard once per finished episode, in mode on only.

The elixir error compares the estimate with engine truth for LOGGING; it is an
info entry and never an observation.
"""
import gymnasium as gym
import numpy as np
import pytest

from python_ai.envs import gym_wrapper
from python_ai.tests.test_trainer_tracker_width import build as _build

KEYS = {"Tracker/Corrections", "Tracker/HandKnownShare", "Tracker/NextKnownShare",
        "Tracker/HandOrNextKnownShare", "Tracker/ElixirAbsErrMean", "Tracker/ElixirAbsErrMax"}


def short_env():
    from python_ai.trainers.train import PHASE1_OPPONENT
    return gym_wrapper.MicroRoyaleEnv(
        {"opponent": PHASE1_OPPONENT, "teacher_stage": 0, "max_ticks": 150})


def run(tmp_path, monkeypatch, mode):
    t = _build(tmp_path, monkeypatch, mode, env_factory=short_env, update_timestep=40)
    logged = {}
    real = t.writer.add_scalar

    def spy(tag, value, *a, **k):
        logged.setdefault(tag, []).append(float(value))
        return real(tag, value, *a, **k)
    t.writer.add_scalar = spy
    try:
        t.collect_rollout()
        return t, logged
    finally:
        t.envs.close()


def test_on_logs_the_tracker_scalars_per_episode(tmp_path, monkeypatch):
    t, logged = run(tmp_path, monkeypatch, "on")
    assert t.episodes_completed >= 1
    assert KEYS <= set(logged), set(logged)
    n = t.episodes_completed
    for k in KEYS:
        assert len(logged[k]) == n
    assert all(0.0 <= v <= 1.0 for v in logged["Tracker/HandOrNextKnownShare"])
    assert all(v >= 0.0 for v in logged["Tracker/ElixirAbsErrMean"])
    assert all(m >= a for m, a in zip(logged["Tracker/ElixirAbsErrMax"],
                                      logged["Tracker/ElixirAbsErrMean"]))


@pytest.mark.parametrize("mode", [None, "zero"])
def test_off_and_zero_log_nothing_new(tmp_path, monkeypatch, mode):
    t, logged = run(tmp_path, monkeypatch, mode)
    assert t.episodes_completed >= 1
    assert not [k for k in logged if k.startswith("Tracker/")]


def test_the_diagnostics_are_info_only_never_observation(monkeypatch):
    from python_ai.tests.test_env_tracker_tail import make, NOOP
    env = make(monkeypatch, "on")
    env.reset()
    obs, _, _, _, info = env.step(NOOP)
    assert "tracker_elixir_abs_err" in info
    env0 = make(monkeypatch, "zero")
    env0.reset()
    _, _, _, _, info0 = env0.step(NOOP)
    assert not [k for k in info0 if k.startswith("tracker_")]
