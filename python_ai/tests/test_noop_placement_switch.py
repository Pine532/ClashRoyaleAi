"""The no-op placement switch (CLASH_NOOP_PLACEMENT_IN_RATIO, docs/TODO.md L18).

On a no-op row the engine ignores the sampled cell, yet its log-prob is part of
the PPO ratio. The switch, off, leaves it out. The rollout's stored log-prob
and the update's recomputed one must leave it out together, or every no-op
decision row's ratio is off by exp(placement log-prob). On, nothing changes.
"""
import torch

from python_ai.rl import ppo as ppo_mod
from python_ai.rl.ppo import ratio_placement_logprob
from python_ai.tests.test_rl_base_trainer import (  # noqa: F401  (fixtures)
    _never_touch_the_live_checkpoint, _phase1, workdir)

HAND = 4


def test_on_by_default_the_placement_part_is_untouched():
    assert ppo_mod.NOOP_PLACEMENT_IN_RATIO
    lp = torch.tensor([-1.0, -2.0, float("-inf")])
    assert ratio_placement_logprob(lp, torch.tensor([HAND, 0, HAND]), HAND) is lp


def test_off_a_noop_rows_placement_is_left_out(monkeypatch):
    monkeypatch.setattr(ppo_mod, "NOOP_PLACEMENT_IN_RATIO", False)
    lp = torch.tensor([-1.0, -2.0, -3.0, float("-inf")])
    got = ratio_placement_logprob(lp, torch.tensor([HAND, 0, HAND, 3]), HAND)
    assert got.tolist() == [0.0, -2.0, 0.0, float("-inf")]


def test_off_the_rollout_and_the_update_agree(workdir, monkeypatch):
    monkeypatch.setattr(ppo_mod, "NOOP_PLACEMENT_IN_RATIO", False)
    seen = {}
    real_update = ppo_mod.PPOUpdater.update

    def update(self, batch, *a, **k):
        stats = real_update(self, batch, *a, **k)
        seen.update(batch=batch, stats=stats)
        return stats
    monkeypatch.setattr(ppo_mod.PPOUpdater, "update", update)
    _phase1()
    b = seen["batch"]
    noop_decisions = int(((b["decision"] > 0) & (b["card_actions"] == HAND)).sum())
    assert noop_decisions > 0, "the rollout drew no no-op on a decision row; nothing was tested"
    # Fresh weights at the first minibatch: rollout and update must agree to
    # float roundoff. One side alone would put exp(placement log-prob), about
    # e^5 for a fresh net, into every no-op decision row's ratio.
    assert seen["stats"].ratio_dev_first < 1e-3
    assert seen["stats"].nonfinite_skips == 0
