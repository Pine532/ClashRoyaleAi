"""A row outside `decision` cannot affect the update, even with a stored log-prob of -inf.

The actor loss excludes non-decision rows. It used to do that only by multiplying
by the 0/1 `decision` mask, after exp(new - old): a stored log-prob of -inf
(an action drawn at probability 0, which MPS's sampler did, docs/TODO.md L11)
made the ratio inf, inf * 0 = NaN, and the update dropped the whole minibatch.
The log-ratio is now masked with torch.where before exp.
"""
import copy
import math

import numpy as np
import torch
import torch.optim as optim

from python_ai.rl.ppo import PPOUpdater
from python_ai.tests.test_rl_ppo import TINY, rollout  # noqa: F401  (fixture)

ROW = (1, 0)   # (t, env)


def _update(net, batch):
    net = copy.deepcopy(net)
    np.random.seed(0)
    torch.manual_seed(0)
    T, N = batch["rewards"].shape
    adv, ret = torch.randn(T, N), torch.zeros(T, N)
    stats = PPOUpdater(net, optim.Adam(net.parameters(), lr=1e-4), TINY).update(
        batch, adv, ret, vf_clip_range=0.2, ent_coef_card=0.05,
        ent_coef_placement=0.06, coverage_coef=0.02)
    return stats, net.state_dict()


def _excluded(batch, stored_logprob):
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    b["decision"][ROW] = 0.0
    b["logprobs"][ROW] = stored_logprob
    return b


def test_a_minus_inf_log_prob_on_an_excluded_row_drops_nothing(rollout):
    net, batch = rollout
    stats, _ = _update(net, _excluded(batch, float("-inf")))
    assert stats.nonfinite_skips == 0
    for name in ("actor_loss", "critic_loss", "total_loss", "clip_frac", "ratio_dev_first"):
        assert math.isfinite(getattr(stats, name)), (name, stats)


def test_an_excluded_rows_stored_log_prob_does_not_matter_at_all(rollout):
    """-inf, a huge finite value and the true value all give the same weights, bit for bit."""
    net, batch = rollout
    _, want = _update(net, _excluded(batch, float(batch["logprobs"][ROW])))
    for stored in (float("-inf"), 50.0, -1e6):
        stats, got = _update(net, _excluded(batch, stored))
        assert stats.nonfinite_skips == 0, stored
        assert all(torch.equal(want[k], got[k]) for k in want), stored
