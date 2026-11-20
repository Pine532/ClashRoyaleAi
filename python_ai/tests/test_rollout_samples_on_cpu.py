"""Rollout actions are drawn on the CPU (rl/sampling.py).

torch 2.13's MPS sampler occasionally draws a category whose probability is
exactly 0 (2026-11-19: about 1 in 10^5 on a masked 612-cell distribution, none
on the CPU). A masked action then enters the buffer with log-prob -inf, the
update's ratio for that row is NaN, and the minibatch is dropped. The live run
lost one minibatch per epoch on about 1 update in 30 that way (docs/TODO.md L11).
"""
import pytest
import torch
from torch.distributions import Categorical

from python_ai.rl import abilities as ability_mod
from python_ai.rl import base_trainer
from python_ai.rl.sampling import sample_categorical
from python_ai.tests.test_rl_base_trainer import (  # noqa: F401  (fixtures)
    _never_touch_the_live_checkpoint, _phase1, workdir)

needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
NEG_INF = float("-inf")


def _placement_like(legal=288, cells=612, seed=0):
    """A masked placement-sized distribution: `legal` finite logits, the rest -inf."""
    logits = torch.randn(cells, generator=torch.Generator().manual_seed(seed))
    logits[legal:] = NEG_INF
    return logits


def test_on_the_cpu_the_draw_is_exactly_categorical_sample():
    """CPU training must not change: same samples, same generator state."""
    heads = [
        _placement_like().expand(64, -1).contiguous(),                  # placement
        torch.tensor([[0.3, NEG_INF, 1.2, NEG_INF, 0.0]] * 64),         # card arms
        torch.tensor([[0.0, NEG_INF], [0.5, -0.5]] * 32),               # ability
    ]
    for logits in heads:
        torch.manual_seed(123)
        want = Categorical(logits=logits).sample()
        want_state = torch.get_rng_state()
        torch.manual_seed(123)
        got = sample_categorical(logits)
        assert torch.equal(got, want)
        assert torch.equal(torch.get_rng_state(), want_state)


def test_the_rollout_draws_card_and_placement_through_the_cpu_sampler(workdir, monkeypatch):
    sizes = []

    def spy(logits):
        sizes.append(logits.shape[-1])
        return sample_categorical(logits)
    monkeypatch.setattr(base_trainer, "sample_categorical", spy)
    trainer = _phase1()
    steps = trainer.cfg.update_timestep
    assert sizes.count(trainer.net.hand_size + 1) == steps, "card draws"
    assert sizes.count(trainer.net.placement_cells) == steps, "placement draws"


def test_ability_draws_go_through_the_cpu_sampler(monkeypatch):
    shapes = []

    def spy(logits):
        shapes.append(tuple(logits.shape))
        return sample_categorical(logits)
    monkeypatch.setattr(ability_mod, "sample_categorical", spy)
    logits = [torch.zeros(3, 2), torch.zeros(3, 2)]
    ready = torch.tensor([[True, False]] * 3)
    actions, logprob, _ = ability_mod.sample(logits, ready)
    assert shapes == [(3, 2), (3, 2)]
    assert (actions[:, 1] == 0).all(), "a head that is not ready can only decline"
    assert torch.isfinite(logprob).all()


@needs_mps
def test_on_mps_no_draw_lands_on_a_zero_probability_cell():
    logits = _placement_like()
    zero = ~(torch.softmax(logits, -1) > 0)
    rows = logits.to("mps").expand(100_000, -1).contiguous()

    def zero_probability_draws(draw):
        torch.manual_seed(0)
        return sum(int(zero[draw(rows).cpu()].sum()) for _ in range(20))   # 2,000,000 draws

    # The control must fire: torch's own MPS draw lands on p = 0 about 17 times
    # in 2,000,000 here (torch 2.13). If it no longer does, this test cannot
    # see the fix either way.
    naive = zero_probability_draws(lambda x: Categorical(logits=x).sample())
    if naive == 0:
        pytest.skip(f"torch {torch.__version__}'s MPS sampler drew no zero-probability "
                    "cell in 2,000,000 draws; this test can no longer see the fix")
    assert zero_probability_draws(sample_categorical) == 0, (
        f"zero-probability draws through sample_categorical (torch's own MPS draw: {naive})")
    assert sample_categorical(rows[:4]).device.type == "mps"
