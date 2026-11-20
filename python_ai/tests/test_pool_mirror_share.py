"""With the deck pool on, a share of episodes is a mirror: the trainee's own deck.

The user's decision (2026-11-20): keep about 25% mirror games once the pool
comes in, so the bot keeps practising the mirror. A deck with a Champion or Hero
cannot be a pool entry (deck_pool refuses them), so for such a deck this is the
only way it meets itself while the pool is on.
"""
import random

from python_ai.envs import gym_wrapper
from python_ai.opponents import deck_pool
from python_ai.trainers import train


def _env(**cfg):
    base = {"opponent": "teacher", "teacher_stage": 0, "deck_pool": True, "scenario_seed": 11}
    base.update(cfg)
    return gym_wrapper.MicroRoyaleEnv(base)


def test_about_a_quarter_of_pool_episodes_are_mirrors():
    env = _env(pool_mirror_share=0.25)
    n, mirrors = 400, 0
    for _ in range(n):
        env.reset()
        if env.current_deck_name == gym_wrapper.MIRROR_DECK_NAME:
            mirrors += 1
            assert list(env.current_opp_deck) == list(env.ai_deck)
            assert env.teacher.deck == list(env.ai_deck)
        else:
            assert env.current_deck_name in env.deck_pool_stats
    # 0.25 +/- about 3 standard deviations at n = 400.
    assert 0.18 < mirrors / n < 0.32, mirrors


def test_share_zero_leaves_the_pool_deck_sequence_exactly_as_before():
    env = _env(pool_mirror_share=0.0)
    got = []
    for _ in range(60):
        env.reset()
        got.append(env.current_deck_name)
    # The pre-change draw: sample_deck alone on the worker's seeded stream,
    # with the priors (no outcome is recorded by resets alone).
    pool = deck_pool.load_pool()
    weights = deck_pool.pfsp_weights({d.name: d.prior_win_rate for d in pool})
    rng = random.Random(11)
    want = [deck_pool.sample_deck(pool, weights, rng).name for _ in range(60)]
    assert got == want


def test_a_mirror_result_never_moves_a_pool_decks_estimate():
    env = _env(pool_mirror_share=1.0)
    env.reset()
    assert env.current_deck_name == gym_wrapper.MIRROR_DECK_NAME
    before = dict(env.deck_pool_stats)
    env._record_deck_outcome(True)
    assert env.deck_pool_stats == before


def test_the_trainer_asks_for_a_quarter_with_the_pool_and_none_in_mirror_mode():
    assert train.PHASE1_POOL_MIRROR_SHARE == 0.25
    env = train.make_env(seed=3)()
    assert env.pool_mirror_share == (0.25 if train.PHASE1_DECK_POOL else 0.0)
