"""CLASH_OPP_TRACKER on the pipeline-1 env (gym_wrapper): the tail on the
trainee's observation.

off  : bit-identical to the engine's observation, no feed objects.
on   : engine width + tracker tail, the tail equal to the env's tracker features.
zero : the same width, an all-zero tail (the A/B control).
Tests may read the opponent's hidden state only to compare.
"""
import gymnasium as gym
import numpy as np
import pytest

import clash_royale_env as E

from python_ai.envs import gym_wrapper
from python_ai.tracking import config as tcfg
from python_ai.tracking import engine_feed as ef
from python_ai.tracking import opponent_tracker as ot

CE = E.ClashRoyaleEnv
HOG = [15, 6, 25, 40, 24, 72, 33, 7]
HERO_DECK = [129, 184, 126, 183, 0, 31, 24, 101]        # Hero Ice Wizard at slot 1
NOOP = {"card_index": CE.HAND_SIZE, "target_x": 0.0, "target_y": 0.0}


def make(monkeypatch, mode, **cfg):
    if mode is None:
        monkeypatch.delenv("CLASH_OPP_TRACKER", raising=False)
    else:
        monkeypatch.setenv("CLASH_OPP_TRACKER", mode)
    return gym_wrapper.MicroRoyaleEnv({"ai_deck": HOG, "opp_deck": HOG, **cfg})


def play_action(env, prefer=None):
    """A legal play of the first affordable hand card (`prefer` first), or a
    no-op."""
    g = env.game
    hand = list(g.get_hand())
    order = sorted(range(len(hand)), key=lambda i: hand[i] != prefer)
    for i in order:
        c = hand[i]
        if g.get_elixir() >= E.get_card_info(c)["cost"]:
            for y in range(17):
                if g.is_valid_placement(c, 9, y, 0):
                    return {"card_index": i, "target_x": 9.0, "target_y": float(y)}
    return dict(NOOP)


def tail_of(env, obs):
    return obs[env.game.observation_size():]


def test_off_is_bit_identical_to_the_engine(monkeypatch):
    env = make(monkeypatch, None)
    assert env._tail is None
    n = env.game.observation_size()
    assert env.observation_space.shape == (n,)
    obs, _ = env.reset(seed=5)
    ref = E.ClashRoyaleEnv(HOG, HOG, 3600, E.TowerTroopType.NONE, E.TowerTroopType.NONE)
    ref.seed(5)
    ref.reset()
    assert np.array_equal(obs, np.array(ref.get_observation_for_team(0), dtype=np.float32))
    for _ in range(25):
        a = play_action(env)
        obs, *_ = env.step(a)
        r = ref.step(a["card_index"], a["target_x"], a["target_y"], 10, False, False)
        assert obs.shape == (n,)
        assert np.array_equal(obs, np.array(r.observation, dtype=np.float32))


def test_unset_and_off_are_the_same(monkeypatch):
    assert tcfg.mode({}) == "off" and tcfg.mode({"CLASH_OPP_TRACKER": "off"}) == "off"
    assert tcfg.tail_width("off") == 0
    with pytest.raises(ValueError):
        tcfg.mode({"CLASH_OPP_TRACKER": "yes"})


def test_on_appends_the_tracker_tail(monkeypatch):
    env = make(monkeypatch, "on")
    n = env.game.observation_size()
    w = ot.feature_width(int(CE.NUM_CARD_IDS))
    assert tcfg.tail_width("on") == w
    assert env.observation_space.shape == (n + w,)
    obs, _ = env.reset(seed=5)
    assert obs.shape == (n + w,) and obs.dtype == np.float32
    assert np.array_equal(obs[:n], np.array(env.game.get_observation_for_team(0), dtype=np.float32))
    for _ in range(30):
        obs, *_ = env.step(play_action(env))
        tick = env.game.get_current_tick()
        assert obs.shape == (n + w,)
        assert np.array_equal(tail_of(env, obs), env._tail.tracker.features(tick))
    # the own next card is the probe's
    off = ot.feature_offsets(int(CE.NUM_CARD_IDS))
    nxt = ef.probe_next_card(env.game, 0)
    tail = tail_of(env, obs)
    assert nxt is not None and tail[off["own_next"] + nxt] == 1.0


def test_zero_has_the_same_layout_and_no_information(monkeypatch):
    env = make(monkeypatch, "zero")
    n = env.game.observation_size()
    w = tcfg.tail_width("zero")
    assert w == ot.feature_width(int(CE.NUM_CARD_IDS))
    assert env.observation_space.shape == (n + w,)
    obs, _ = env.reset(seed=5)
    assert obs.shape == (n + w,) and not tail_of(env, obs).any()
    for _ in range(30):
        obs, *_ = env.step(play_action(env))
        assert not tail_of(env, obs).any()
    assert env._tail.feed is None            # nothing is polled in the control


@pytest.mark.parametrize("opponent", ["builtin", "teacher"])
def test_the_opponents_plays_reach_the_tail(monkeypatch, opponent):
    env = make(monkeypatch, "on", opponent=opponent)
    env.reset(seed=3)
    off = ot.feature_offsets(int(CE.NUM_CARD_IDS))
    seen = False
    for _ in range(80):
        obs, *_ = env.step(dict(NOOP))
        played = [c for c in set(HOG) if env.game.get_last_played_tick(1, c) >= 0]
        if played:
            seen = True
            assert set(env._tail.tracker.seen_order()) == set(played)
            tail = tail_of(env, obs)
            assert tail[off["scalars"] + ot.SCALAR_INDEX["opp_distinct_seen"]] > 0
            break
    assert seen, "the opponent never played in 80 decisions"


def test_reset_and_autoreset_clear_the_tracker(monkeypatch):
    env = make(monkeypatch, "on", opponent="teacher")
    env.reset(seed=3)
    for _ in range(60):
        env.step(dict(NOOP))
    assert env._tail.tracker.distinct_seen() > 0
    obs, _ = env.reset()
    assert env._tail.tracker.distinct_seen() == 0 and env._tail.tracker.seen_order() == []
    off = ot.feature_offsets(int(CE.NUM_CARD_IDS))
    tail = tail_of(env, obs)
    assert not tail[off["opp_in_hand"]:off["opp_in_hand"] + int(CE.NUM_CARD_IDS)].any()

    # Autoreset: a short match in a vector env, stepped through its end.
    monkeypatch.setenv("CLASH_OPP_TRACKER", "on")
    venv = gym.vector.SyncVectorEnv([lambda: gym_wrapper.MicroRoyaleEnv(
        {"ai_deck": HOG, "opp_deck": HOG, "max_ticks": 400, "opponent": "teacher"})])
    venv.reset(seed=[3])
    act = {"card_index": np.array([CE.HAND_SIZE]), "target_x": np.zeros((1, 1), np.float32),
           "target_y": np.zeros((1, 1), np.float32),
           "activate_ability_slot1": np.array([0]), "activate_ability_slot2": np.array([0])}
    inner = venv.envs[0]
    done_at = None
    for i in range(60):
        _, _, term, trunc, _ = venv.step(act)
        if term[0] or trunc[0]:
            done_at = i
            assert inner._tail.tracker.distinct_seen() > 0     # the terminal obs kept its tail
            break
    assert done_at is not None
    obs, *_ = venv.step(act)                                    # the autoreset step
    assert inner._tail.tracker.distinct_seen() == 0
    assert not obs[0][inner.game.observation_size() + off["opp_in_hand"]:
                      inner.game.observation_size() + off["opp_in_hand"] + int(CE.NUM_CARD_IDS)].any()


def test_scenario_resets_carry_a_consistent_tail(monkeypatch):
    env = make(monkeypatch, "on", defensive_scenario_prob=1.0)
    obs, _ = env.reset(seed=2)
    n = env.game.observation_size()
    assert env.last_scenario is not None
    assert obs.shape == (n + tcfg.tail_width("on"),)
    assert np.array_equal(obs[:n], np.array(env.game.get_observation_for_team(0), dtype=np.float32))
    assert np.array_equal(tail_of(env, obs),
                          env._tail.tracker.features(env.game.get_current_tick()))


def test_the_trainees_ability_press_reaches_the_tail(monkeypatch):
    env = make(monkeypatch, "on", ai_deck=HERO_DECK, opp_deck=HERO_DECK, opponent="builtin")
    env.reset(seed=4)
    pressed = False
    for _ in range(300):
        ready = env.game.is_champion_ability_ready(0, 1)
        # A press step plays no card, so the elixir is there for the ability.
        a = dict(NOOP) if ready else play_action(env, prefer=HERO_DECK[1])
        a["activate_ability_slot1"] = 1 if ready else 0
        obs, *_ = env.step(a)
        if ready:
            pressed = True
            break
    assert pressed
    cost = env._tail.tracker.own_ability_spend()
    assert cost > 0
    off = ot.feature_offsets(int(CE.NUM_CARD_IDS))
    assert tail_of(env, obs)[off["scalars"] + ot.SCALAR_INDEX["own_ability_spend"]] > 0


def test_the_teachers_ability_press_reaches_the_tail(monkeypatch):
    env = make(monkeypatch, "on", ai_deck=HERO_DECK, opp_deck=HERO_DECK, opponent="teacher")
    env.reset(seed=4)
    env.teacher.ability_flags = lambda game, obs: (True, False)
    for _ in range(120):
        env.step(dict(NOOP))
        if env._tail.tracker.opp_ability_spend() > 0:
            return
    pytest.fail("the opponent's ability press never reached the tracker")


def test_the_setting_is_recorded_and_a_change_on_resume_is_a_training_change():
    """Resuming under another CLASH_OPP_TRACKER changes what the policy sees: it
    must warn as a CHANGED setting, not pass as operational (a pin: this holds
    because the variable is a CLASH_* one and is not in OPERATIONAL_SETTINGS)."""
    from python_ai.rl import checkpointing as ck
    assert "CLASH_OPP_TRACKER" in ck.clash_settings({"CLASH_OPP_TRACKER": "on", "HOME": "x"})
    for saved, current in (({}, {"CLASH_OPP_TRACKER": "on"}),
                           ({"CLASH_OPP_TRACKER": "on"}, {"CLASH_OPP_TRACKER": "zero"}),
                           ({"CLASH_OPP_TRACKER": "zero"}, {})):
        changed, operational = ck.settings_drift(saved, current)
        assert [c[0] for c in changed] == ["CLASH_OPP_TRACKER"] and not operational
