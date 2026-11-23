"""Trainers build the net from CLASH_OPP_TRACKER and assert the tail width.

A tiny phase-1 trainer (the test_replay_path_anchoring pattern): absolute
CLASH_WEIGHTS / CLASH_LOGDIR and a CLASH_ARTIFACT_DIR under tmp_path, so nothing
shared is written.
"""
import gymnasium as gym
import pytest
import torch

import clash_royale_env as E

from python_ai.envs import gym_wrapper
from python_ai.models.net import MicroRoyaleNet
from python_ai.tracking import config as tcfg

CE = E.ClashRoyaleEnv


def build(tmp_path, monkeypatch, mode, env_factory=None, update_timestep=4):
    from python_ai.rl.config import PPOConfig
    from python_ai.trainers.train import PHASE1_OPPONENT, Phase1Trainer
    monkeypatch.setenv("CLASH_WEIGHTS", str(tmp_path / "w.pth"))
    monkeypatch.setenv("CLASH_LOGDIR", str(tmp_path / "runs"))
    monkeypatch.setenv("CLASH_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    if mode is None:
        monkeypatch.delenv("CLASH_OPP_TRACKER", raising=False)
    else:
        monkeypatch.setenv("CLASH_OPP_TRACKER", mode)
    cfg = PPOConfig(num_envs=1, update_timestep=update_timestep, bptt_chunk=2, num_minibatches=1,
                    ppo_epochs=1, save_every_episodes=10 ** 9, replay_every_episodes=10 ** 9)

    class Harness(Phase1Trainer):
        def __init__(self):
            super().__init__(cfg)
            self.log_dir = str(tmp_path / "runs")

        def build_envs(self):
            make = env_factory or (lambda: gym_wrapper.MicroRoyaleEnv(
                {"opponent": PHASE1_OPPONENT, "teacher_stage": 0}))
            return gym.vector.SyncVectorEnv([make])

    t = Harness()
    t.setup()
    return t


@pytest.mark.parametrize("mode", ["on", "zero"])
def test_a_tracker_trainer_runs_a_rollout_and_an_update_at_the_right_width(tmp_path, monkeypatch, mode):
    t = build(tmp_path, monkeypatch, mode)
    try:
        w = tcfg.tail_width(mode)
        assert t.net.tracker_size == w > 0
        assert t.net.obs_size == t.envs.single_observation_space.shape[0] == t._obs.shape[-1]
        t.collect_rollout()
        stats = t.run_update()
        assert stats is not None
        t.buffer.clear()
        assert "tracker_branch.0.weight" in t.net.state_dict()
    finally:
        t.envs.close()


def test_the_rollout_asserts_the_width(tmp_path, monkeypatch):
    """A tracker net fed engine-width observations (the env's tail missing) must
    raise, not silently skip the branch."""
    t = build(tmp_path, monkeypatch, "on")
    try:
        t._obs = t._obs[:, :t.net.tracker_start]
        with pytest.raises(ValueError, match="tracker"):
            t.collect_rollout()
    finally:
        t.envs.close()


def test_off_builds_exactly_todays_net(tmp_path, monkeypatch):
    t = build(tmp_path, monkeypatch, None)
    try:
        ref = MicroRoyaleNet(num_ability_slots=t.num_ability_slots())
        assert list(t.net.state_dict().keys()) == list(ref.state_dict().keys())
        assert t.net.tracker_size == 0 and not hasattr(t.net, "tracker_branch")
        assert t._obs.shape[-1] == CE(gym_wrapper.DEFAULT_DECK, gym_wrapper.DEFAULT_DECK, 100).observation_size()
        t.collect_rollout()
        assert t.run_update() is not None
    finally:
        t.envs.close()


def test_load_net_follows_the_setting(monkeypatch, tmp_path):
    from python_ai.models import policy_io
    monkeypatch.setenv("CLASH_OPP_TRACKER", "on")
    path = str(tmp_path / "old.pth")
    torch.save(MicroRoyaleNet().state_dict(), path)       # an old-layout checkpoint
    net = policy_io.load_net(path, "cpu", verbose=False)
    assert net.tracker_size == tcfg.tail_width("on")


def test_the_exploiter_refuses_a_tracker_run(monkeypatch):
    from python_ai.trainers import exploiter
    monkeypatch.setenv("CLASH_OPP_TRACKER", "on")
    with pytest.raises(NotImplementedError, match="CLASH_OPP_TRACKER"):
        exploiter.run_exploiter_burst(None, "cpu", 0, 0)
