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

    def scripted():
        """Team 1 deploys the Hero as soon as it can, then presses its ability."""
        g = env.game
        hand = list(g.get_hand_for_team(1))
        if HERO_DECK[1] in hand and g.get_elixir_for_team(1) >= E.get_card_info(HERO_DECK[1])["cost"]:
            cell = next(((x, y) for y in range(34) for x in range(18)
                         if g.is_valid_placement(HERO_DECK[1], x, y, 1)), None)
            if cell:
                return hand.index(HERO_DECK[1]), float(cell[0]), float(33 - cell[1]), False, False
        ready = bool(g.is_champion_ability_ready(1, 1))
        if not ready and HERO_DECK[1] not in hand:
            # Cycle until the Hero comes round.
            for i, c in enumerate(hand):
                if g.get_elixir_for_team(1) >= E.get_card_info(c)["cost"]:
                    cell = next(((x, y) for y in range(34) for x in range(18)
                                 if g.is_valid_placement(c, x, y, 1)), None)
                    if cell:
                        return i, float(cell[0]), float(33 - cell[1]), False, False
        return CE.HAND_SIZE, 0.0, 0.0, ready, False

    env._opponent_action = scripted
    for _ in range(300):
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


# ---------------------------------------------------------------- neural opponents

import torch                                                        # noqa: E402

from python_ai.envs.gym_wrapper import DEFAULT_DECK_ABILITY_SLOTS    # noqa: E402
from python_ai.models.net import MicroRoyaleNet                      # noqa: E402

W = None


def _width():
    return ot.feature_width(int(CE.NUM_CARD_IDS))


def _snapshot(tmp_path, name, tracker, seed):
    """A bare state_dict file. tracker=False: the old layout (no branch keys).
    tracker=True: the new layout with a NONZERO branch (as a trained one is)."""
    torch.manual_seed(seed)
    net = MicroRoyaleNet(num_ability_slots=DEFAULT_DECK_ABILITY_SLOTS,
                         tracker_size=_width() if tracker else 0)
    if tracker:
        with torch.no_grad():
            for p in net.tracker_branch.parameters():
                p.copy_(torch.randn_like(p) * 0.5)
    path = str(tmp_path / f"{name}.pth")
    torch.save(net.state_dict(), path)
    return path


def _action_stream(env, steps, seed0):
    out = []
    for i in range(steps):
        torch.manual_seed(seed0 + i)
        out.append(env._opponent_action())
        env.step(dict(NOOP))
    return out


def test_an_old_snapshot_plays_exactly_as_before(monkeypatch, tmp_path):
    """Same weights, same engine seed: the actions of team 1 with the tracker on
    (a zero tail) equal those with it off. Also after a NEW-layout snapshot was
    loaded into the same env first (the reused opponent net must not keep its
    trained tracker weights)."""
    old = _snapshot(tmp_path, "old", False, seed=1)
    new = _snapshot(tmp_path, "new", True, seed=2)

    off = make(monkeypatch, None)
    off.set_historical_opponent(old)
    off.game.seed(11)
    off.reset()
    ref = _action_stream(off, 40, 100)

    on = make(monkeypatch, "on")
    assert on.opponent_net.expects_tail
    on.set_historical_opponent(new)          # leaves trained tracker weights behind
    on.set_historical_opponent(old)
    on.game.seed(11)
    on.reset()
    assert not on._opponent_tracker_active()
    got = _action_stream(on, 40, 100)
    assert got == ref
    assert any(a[0] != CE.HAND_SIZE for a in ref), "team 1 never played: the control is empty"


def test_a_new_layout_snapshot_receives_its_own_nonzero_tail(monkeypatch, tmp_path):
    new = _snapshot(tmp_path, "new", True, seed=2)
    env = make(monkeypatch, "on")
    env.set_historical_opponent(new)
    env.reset()
    assert env._opponent_tracker_active()
    seen = []
    real = env.opponent_net.extract_features_hires
    env.opponent_net.extract_features_hires = lambda obs: (
        seen.append((obs.clone(), env._tail1.tracker.features(int(env.game.get_current_tick())))),
        real(obs))[1]
    n = env.game.observation_size()
    played = None
    for _ in range(40):
        torch.manual_seed(0)
        env._opponent_action()
        a = {"card_index": CE.HAND_SIZE, "target_x": 0.0, "target_y": 0.0}
        hand = list(env.game.get_hand())
        for i, c in enumerate(hand):
            if played is None and env.game.get_elixir() >= E.get_card_info(c)["cost"]:
                cell = next(((x, y) for y in range(17) for x in range(18)
                             if env.game.is_valid_placement(c, x, y, 0)), None)
                if cell:
                    a = {"card_index": i, "target_x": float(cell[0]), "target_y": float(cell[1])}
                    played = c
                    break
        env.step(a)
    assert seen and all(o.shape[-1] == n + _width() for o, _ in seen)
    tail = seen[-1][0][0, n:].numpy()
    assert tail.any()
    assert np.array_equal(tail, seen[-1][1])      # exactly team 1's tracker features
    # it is team 1's tracker: observer 1, watching TEAM 0's plays
    assert env._tail1.observer == 1
    assert played is not None and played in env._tail1.tracker.seen_order()
    off = ot.feature_offsets(int(CE.NUM_CARD_IDS))
    assert tail[off["own_next"]:off["own_next"] + int(CE.NUM_CARD_IDS)].sum() == 1.0
    # the trainee's own tail still follows team 1 (a separate tracker)
    assert env._tail.observer == 0 and env._tail is not env._tail1
