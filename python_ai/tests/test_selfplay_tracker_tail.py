"""CLASH_OPP_TRACKER on the pipeline-2 env (selfplay_env): the trainee's tail."""
import numpy as np
import pytest

import clash_royale_env as E

from python_ai.envs import selfplay_env
from python_ai.tracking import config as tcfg
from python_ai.tracking import engine_feed as ef
from python_ai.tracking import opponent_tracker as ot

CE = E.ClashRoyaleEnv
NOOP = {"card_index": CE.HAND_SIZE, "target_x": 0.0, "target_y": 0.0}
HERO_DECK = [129, 184, 126, 183, 0, 31, 24, 101]


def make(monkeypatch, mode, **cfg):
    if mode is None:
        monkeypatch.delenv("CLASH_OPP_TRACKER", raising=False)
    else:
        monkeypatch.setenv("CLASH_OPP_TRACKER", mode)
    return selfplay_env.MicroRoyaleSelfPlayEnv({"scenarios_enabled": False, **cfg})


def test_off_has_no_tail_and_the_engine_observation(monkeypatch):
    env = make(monkeypatch, None)
    assert env._tail is None
    n = env.game.observation_size()
    assert env.observation_space.shape == (n,)
    obs, _ = env.reset(seed=1)
    assert np.array_equal(obs, np.array(env.game.get_observation_for_team(0), dtype=np.float32))
    for _ in range(10):
        obs, *_ = env.step(dict(NOOP))
        assert obs.shape == (n,)
        assert np.array_equal(obs, np.array(env.game.get_observation_for_team(0), dtype=np.float32))


def test_on_appends_the_tail_and_it_equals_the_trackers_features(monkeypatch):
    env = make(monkeypatch, "on")
    n, w = env.game.observation_size(), tcfg.tail_width("on")
    assert env.observation_space.shape == (n + w,)
    obs, _ = env.reset()
    assert obs.shape == (n + w,)
    for _ in range(40):
        obs, *_ = env.step(dict(NOOP))
        assert obs.shape == (n + w,)
        assert np.array_equal(obs[:n], np.array(env.game.get_observation_for_team(0), dtype=np.float32))
        assert np.array_equal(obs[n:], env._tail.tracker.features(env.game.get_current_tick()))


def test_zero_gives_an_all_zero_tail(monkeypatch):
    env = make(monkeypatch, "zero")
    n, w = env.game.observation_size(), tcfg.tail_width("zero")
    obs, _ = env.reset()
    assert obs.shape == (n + w,) and not obs[n:].any()
    for _ in range(20):
        obs, *_ = env.step(dict(NOOP))
        assert obs.shape == (n + w,) and not obs[n:].any()
    assert env._tail.feed is None


@pytest.mark.parametrize("kind", ["neural", "scripted:Cycler", "builtin"])
def test_the_opponents_plays_reach_the_tail(monkeypatch, kind):
    env = make(monkeypatch, "on")
    if kind == "builtin":
        env.set_builtin_opponent("builtin:heuristic@1.50")
    elif kind != "neural":
        env.set_scripted_opponent(kind.split(":")[1])
    env.reset()
    for _ in range(120):
        obs, *_ = env.step(dict(NOOP))
        played = [c for c in range(int(CE.NUM_CARD_IDS)) if env.game.get_last_played_tick(1, c) >= 0]
        if played:
            assert set(env._tail.tracker.seen_order()) == set(played)
            off = ot.feature_offsets(int(CE.NUM_CARD_IDS))
            n = env.game.observation_size()
            assert obs[n + off["scalars"] + ot.SCALAR_INDEX["opp_distinct_seen"]] > 0
            return
    pytest.fail(f"{kind} never played in 120 decisions")


def test_reset_clears_the_tracker(monkeypatch):
    env = make(monkeypatch, "on")
    env.reset()
    for _ in range(80):
        env.step(dict(NOOP))
    assert env._tail.tracker.distinct_seen() > 0
    obs, _ = env.reset()
    assert env._tail.tracker.distinct_seen() == 0
    off = ot.feature_offsets(int(CE.NUM_CARD_IDS))
    n = env.game.observation_size()
    assert not obs[n + off["opp_in_hand"]:n + off["opp_in_hand"] + int(CE.NUM_CARD_IDS)].any()


def test_scenario_resets_carry_a_consistent_tail(monkeypatch):
    env = make(monkeypatch, "on", scenarios_enabled=True)
    env.scenario_rng = env.rng = np.random.default_rng(0)
    monkeypatch.setattr(selfplay_env.scenarios, "SCENARIO_INJECTION_PROB", 1.0)
    obs, _ = env.reset()
    n = env.game.observation_size()
    assert env.scenario_active is not None
    assert np.array_equal(obs[n:], env._tail.tracker.features(env.game.get_current_tick()))


def test_the_opponents_ability_press_reaches_the_tail(monkeypatch):
    env = make(monkeypatch, "on", deck=HERO_DECK)
    env.reset()
    real = env._opponent_action
    env._opponent_action = lambda: (*real()[:3], True, False)
    for _ in range(150):
        env.step(dict(NOOP))
        if env._tail.tracker.opp_ability_spend() > 0:
            return
    pytest.fail("team 1's ability press never reached the tracker")


def test_the_trainees_own_next_card_is_the_probes(monkeypatch):
    env = make(monkeypatch, "on")
    env.reset()
    for _ in range(5):
        obs, *_ = env.step(dict(NOOP))
    n = env.game.observation_size()
    off = ot.feature_offsets(int(CE.NUM_CARD_IDS))
    nxt = ef.probe_next_card(env.game, 0)
    assert nxt is not None and obs[n + off["own_next"] + nxt] == 1.0
