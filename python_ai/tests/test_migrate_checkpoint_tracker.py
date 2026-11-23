"""migrate_checkpoint_tracker: an old (tracker-less) checkpoint becomes a tracker
checkpoint that resumes cleanly, optimizer moments and all.

The main test builds a tiny PHASE-2 trainer (in-process envs, tmp paths,
CLASH_OPP_TRACKER unset), trains one update so Adam has state, migrates, then
resumes a CLASH_OPP_TRACKER=on trainer from the migrated file.
"""
import subprocess
import sys

import gymnasium as gym
import pytest
import torch

import clash_royale_env as E

from python_ai.envs import gym_wrapper, selfplay_env
from python_ai.models.net import MicroRoyaleNet
from python_ai.tools import migrate_checkpoint_tracker as mig
from python_ai.tracking import config as tcfg

CE = E.ClashRoyaleEnv


def _build(tmp_path, monkeypatch, mode, weights):
    from python_ai.rl.config import PPOConfig
    from python_ai.trainers import train_selfplay as ts
    monkeypatch.setenv("CLASH_WEIGHTS", str(tmp_path / "w.pth"))
    monkeypatch.setenv("CLASH_LOGDIR", str(tmp_path / "runs"))
    monkeypatch.setenv("CLASH_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    if mode is None:
        monkeypatch.delenv("CLASH_OPP_TRACKER", raising=False)
    else:
        monkeypatch.setenv("CLASH_OPP_TRACKER", mode)
    if not weights.exists():
        # A from-scratch phase 2 seeds from a phase-1 file; give it a tmp one.
        boot = tmp_path / "bootstrap.pth"
        torch.save(MicroRoyaleNet(num_ability_slots=0).state_dict(), boot)
        monkeypatch.setattr(ts, "BOOTSTRAP_FROM_PATH", str(boot))
    cfg = PPOConfig(num_envs=1, update_timestep=4, bptt_chunk=2, num_minibatches=1,
                    ppo_epochs=1, save_every_episodes=10 ** 9, replay_every_episodes=10 ** 9)

    class Harness(ts.Phase2Trainer):
        weight_path = str(weights)

        def __init__(self):
            super().__init__(cfg)
            self.log_dir = str(tmp_path / ("runs_" + weights.stem))

        def build_envs(self):
            return gym.vector.SyncVectorEnv(
                [selfplay_env.make_env(scenarios_enabled=False)])

        def _refresh_pool(self, initial=False):
            self.historical_pool = []
            self.envs.call("refresh_pfsp_pool", self._pool_broadcast())

    t = Harness()
    clean = []
    orig = t.restore_common
    t.restore_common = lambda ck: clean.append(orig(ck)) or clean[-1]
    t.clean_loads = clean
    t.setup()
    return t


@pytest.fixture
def old_ckpt(tmp_path, monkeypatch):
    path = tmp_path / "old_selfplay.pth"
    t = _build(tmp_path, monkeypatch, None, path)
    try:
        t.collect_rollout()
        assert t.run_update() is not None
        t.buffer.clear()
        t.episodes_completed = 4321
        t.entropy_reboost_episode = 1234
        t.last_stall_reboost_episode = 2345
        t.save_checkpoint(verbose=False)
    finally:
        t.envs.close()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck["pfsp_stats"] = {"opp_a": 0.31, "opp_b": 0.77}
    ck["pfsp_counts"] = {"opp_a": 40, "opp_b": 12}
    torch.save(ck, path)
    return path


def _migrate(old, out, mode="on", seed=0):
    return subprocess.run(
        [sys.executable, "-m", "python_ai.tools.migrate_checkpoint_tracker",
         "--in", str(old), "--out", str(out), "--mode", mode, "--seed", str(seed)],
        capture_output=True, text=True)


def test_a_migrated_phase2_checkpoint_resumes_cleanly(tmp_path, monkeypatch, old_ckpt, capsys):
    new_path = tmp_path / "new_selfplay.pth"
    res = _migrate(old_ckpt, new_path)
    assert res.returncode == 0, res.stdout + res.stderr
    old = torch.load(old_ckpt, map_location="cpu", weights_only=False)

    t = _build(tmp_path, monkeypatch, "on", new_path)
    try:
        out = capsys.readouterr().out
        # (a) a clean load, with the optimizer restored, and no settings warning
        assert t.clean_loads == [True]
        assert "Optimizer state NOT restored" not in out
        assert "DIFFERENT CLASH_* SETTINGS" not in out
        # (c) no entropy re-boost
        assert "forcing a fresh" not in out
        assert t.entropy_reboost_episode == 1234
        assert t.last_stall_reboost_episode == 2345
        # (d) episodes and PFSP estimates
        assert t.episodes_completed == 4321
        assert t._last_pfsp[0] == {"opp_a": 0.31, "opp_b": 0.77}
        assert t._last_pfsp[1] == {"opp_a": 40, "opp_b": 12}
        # (b) optimizer state of every old parameter is bit-equal
        n_old = len(old["optimizer"]["param_groups"][0]["params"])
        new_sd = t.optimizer.state_dict()
        assert len(new_sd["param_groups"][0]["params"]) == n_old + 4
        assert len(old["optimizer"]["state"]) == n_old
        for i in range(n_old):
            a, b = old["optimizer"]["state"][i], new_sd["state"][i]
            assert a.keys() == b.keys()
            for k in a:
                assert torch.equal(torch.as_tensor(a[k]), torch.as_tensor(b[k])), (i, k)
        for i in range(n_old, n_old + 4):
            assert i not in new_sd["state"]
        # (e) outputs equal at a random tail
        ref = MicroRoyaleNet(num_ability_slots=t.num_ability_slots()).eval()
        assert ref.load_state_dict(old["model"]) is not None
        obs = mig._real_observations(6)
        g = torch.Generator().manual_seed(9)
        full = torch.cat([obs, torch.randn(6, tcfg.tail_width("on"), generator=g)], 1)
        t.net.eval()
        with torch.no_grad():
            for fn in (mig._rollout_outputs, mig._update_outputs):
                for x, y in zip(fn(ref, obs), fn(t.net, full)):
                    assert torch.equal(x, y), fn.__name__
        # the trainer can still step on the migrated state
        t.collect_rollout()
        assert t.run_update() is not None
    finally:
        t.envs.close()


def test_the_recorded_settings_are_stamped_with_the_mode(tmp_path, old_ckpt):
    new_path = tmp_path / "n.pth"
    assert _migrate(old_ckpt, new_path, mode="zero").returncode == 0
    ck = torch.load(new_path, map_location="cpu", weights_only=False)
    assert ck["clash_settings"]["CLASH_OPP_TRACKER"] == "zero"


def test_the_input_is_not_modified(tmp_path, old_ckpt):
    before = old_ckpt.read_bytes()
    assert _migrate(old_ckpt, tmp_path / "n.pth").returncode == 0
    assert old_ckpt.read_bytes() == before


def test_refuses_out_equal_to_in(old_ckpt):
    before = old_ckpt.read_bytes()
    res = _migrate(old_ckpt, old_ckpt)
    assert res.returncode != 0 and "differ" in (res.stdout + res.stderr)
    assert old_ckpt.read_bytes() == before


def test_refuses_an_existing_out(tmp_path, old_ckpt):
    out = tmp_path / "exists.pth"
    out.write_bytes(b"keep me")
    res = _migrate(old_ckpt, out)
    assert res.returncode != 0 and "exists" in (res.stdout + res.stderr)
    assert out.read_bytes() == b"keep me"


def test_refuses_a_checkpoint_that_already_has_the_branch(tmp_path, old_ckpt):
    once = tmp_path / "once.pth"
    assert _migrate(old_ckpt, once).returncode == 0
    res = _migrate(once, tmp_path / "twice.pth")
    assert res.returncode != 0 and "already" in (res.stdout + res.stderr)
    assert not (tmp_path / "twice.pth").exists()


def test_refuses_an_unknown_mode(tmp_path, old_ckpt):
    assert _migrate(old_ckpt, tmp_path / "n.pth", mode="off").returncode != 0
