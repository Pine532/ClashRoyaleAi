"""The shared opponent tracker, fed from perception events.

CONFORMANCE (the requirement): one event stream recorded from a real simulator
game through EngineFeed, replayed through the perception adapter with the ENGINE
rules, gives the same features as the simulator's at every decision. With the
REAL_GAME rules only the elixir scalar differs.

Tests read the opponent's hand/elixir nowhere: the recorded stream is what a
viewer sees (plays with ticks, ability presses, the viewer's own next card).
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from contracts import UNKNOWN_CARD_SIM_ID, PlacementEvent

MINION_GIANT = [129, 184, 126, 183, 0, 31, 24, 101]     # Hero Ice Wizard at index 1
HOG = [15, 6, 25, 40, 24, 72, 33, 7]
MIRROR_DECK = [15, 6, 25, 40, 24, 72, 164, 7]


@pytest.fixture(scope="module")
def sim(engine):
    from python_ai.tracking import engine_feed as ef
    from python_ai.tracking import opponent_tracker as ot
    return engine, ef, ot


def _legal_play(E, env, team, rng, p):
    if rng.random() > p:
        return (-1, 0, 0)
    hand = env.get_hand_for_team(team)
    cooldown = env.get_hand_cooldown_ticks(team)
    elixir = env.get_elixir_for_team(team)
    order = [i for i in range(4) if cooldown[i] == 0
             and elixir >= E.get_card_info(hand[i])["cost"]]
    rng.shuffle(order)
    for i in order:
        cells = [(x, y) for x in range(18) for y in range(34)
                 if env.is_valid_placement(hand[i], x, y, team)]
        if cells:
            x, y = rng.choice(cells)
            return (i, x, y if team == 0 else 33 - y)
    return (-1, 0, 0)


def record_game(sim, deck, seed, p=0.3, decisions=300, presses=False):
    """Play a seeded game with the opponent (team 1) observed by team 0 through
    EngineFeed. Returns [(decision_tick, [(name, args...)], features)]: the
    tracker calls EngineFeed made at each decision, in order, and the features
    the engine-side tracker read."""
    E, ef, ot = sim
    CE = E.ClashRoyaleEnv
    none = E.TowerTroopType.NONE
    env = CE(list(deck), list(deck), 3600, none, none)
    env.seed(seed)

    log = []

    class Recording(ot.OpponentTracker):
        def on_play(self, tick, card_id):
            log.append(("opp_play", int(tick), int(card_id)))
            super().on_play(tick, card_id)

        def on_ability(self, tick, cost):
            log.append(("opp_ability", int(tick), float(cost)))
            super().on_ability(tick, cost)

        def on_own_ability(self, cost):
            log.append(("own_ability", float(cost)))
            super().on_own_ability(cost)

        def set_own_next(self, card_id):
            log.append(("own_next", None if card_id is None else int(card_id)))
            super().set_own_next(card_id)

    tr = Recording(ef.rules_from_engine(), ef.card_facts_from_engine(),
                   int(CE.NUM_CARD_IDS), float(CE.MAX_MATCH_ELIXIR))
    feed = ef.EngineFeed(env, 0, tr, decks=(deck, deck))
    rng = random.Random(seed)
    slots = [s for s in (1, 2) if E.get_card_info(deck[s])["is_hero"]
             or E.get_card_info(deck[s])["is_champion"]]
    out = []
    for _ in range(decisions):
        if env.is_game_over():
            break
        a0 = _legal_play(E, env, 0, rng, p)
        a1 = _legal_play(E, env, 1, rng, p)
        flags = [False, False, False, False]
        if presses:
            for s in slots:
                if env.is_champion_ability_ready(1, s):
                    feed.note_activation(1, s, True)
                    flags[2] = True
                    break
            for s in slots:
                if env.is_champion_ability_ready(0, s):
                    feed.note_activation(0, s, True)
                    flags[0] = True
                    break
        env.step_self_play(*a0, *a1, 10, *flags)
        n0 = len(log)
        feed.poll()
        tick = int(env.get_current_tick())
        feats = tr.features(tick)
        out.append((tick, list(log[n0:]), feats))
    return out


def replay(adapter, recording):
    feats = []
    for tick, events, _ in recording:
        for ev in events:
            kind = ev[0]
            if kind == "opp_play":
                adapter.on_opponent_play(ev[1], ev[2])
            elif kind == "opp_ability":
                adapter.on_opponent_ability(ev[1], ev[2])
            elif kind == "own_ability":
                adapter.on_own_ability(tick, ev[1])
            elif kind == "own_next":
                adapter.set_own_next(ev[1])
        feats.append(adapter.features(tick))
    return feats


@pytest.mark.parametrize("deck,seed,presses", [(HOG, 0, False), (HOG, 3, False),
                                               (MIRROR_DECK, 1, False),
                                               (MINION_GIANT, 0, True)])
def test_engine_rules_replay_is_identical_to_the_simulator(sim, deck, seed, presses):
    from track.tracker_adapter import TrackerAdapter
    rec = record_game(sim, deck, seed, presses=presses)
    assert any(ev[0] == "opp_play" for _, evs, _ in rec for ev in evs)
    feats = replay(TrackerAdapter("engine"), rec)
    assert len(feats) == len(rec)
    for (tick, _, truth), got in zip(rec, feats):
        assert np.array_equal(got, truth), tick
    if presses:
        kinds = {ev[0] for _, evs, _ in rec for ev in evs}
        assert "opp_ability" in kinds, "no ability press: the hook was not exercised"


def test_real_game_rules_differ_only_in_the_elixir_scalar(sim):
    _, _, ot = sim
    from track.tracker_adapter import TrackerAdapter
    rec = record_game(sim, HOG, 0, decisions=360)
    eng = replay(TrackerAdapter("engine"), rec)
    real = replay(TrackerAdapter("real"), rec)
    n = ot.feature_width(int(sim[0].ClashRoyaleEnv.NUM_CARD_IDS))
    elixir = (n - ot.NUM_SCALARS) + ot.SCALAR_INDEX["opp_elixir"]
    other = np.ones(n, bool)
    other[elixir] = False
    diff_ticks = 0
    for a, b in zip(eng, real):
        assert np.array_equal(a[other], b[other])
        diff_ticks += a[elixir] != b[elixir]
    assert diff_ticks > 0, "control: the real rules must change the elixir"
    # faster regeneration: never below the engine's
    assert all(b[elixir] >= a[elixir] - 1e-6 for a, b in zip(eng, real))


def test_real_preset_is_the_shared_real_game_rules(sim):
    _, ef, ot = sim
    from track.tracker_adapter import TrackerAdapter
    CE = sim[0].ClashRoyaleEnv
    a = TrackerAdapter("real")
    assert a.tracker.rules == ot.real_game_rules(CE.DOUBLE_ELIXIR_TICK, CE.TRIPLE_ELIXIR_TICK)
    assert TrackerAdapter("engine").tracker.rules == ef.rules_from_engine()
    with pytest.raises(ValueError):
        TrackerAdapter("bogus")


def test_placement_event_feeds_only_opponent_plays(sim):
    from track.tracker_adapter import TrackerAdapter
    a = TrackerAdapter("real")

    def ev(team, card, tick):
        return PlacementEvent(tick=tick, wall_time_ms=0.0, card_sim_id=card,
                              card_real_name="x", team=team, tile_x=1, tile_y=1,
                              confidence=1.0)
    a.on_placement(ev(0, 15, 10))
    assert a.tracker.seen_order() == []
    a.on_placement(ev(1, 15, 20))
    assert a.tracker.seen_order() == [15]


def test_unmapped_opponent_play_is_counted_and_flags_a_degraded_cycle(sim):
    from track.tracker_adapter import TrackerAdapter
    a = TrackerAdapter("real")
    a.on_opponent_play(10, UNKNOWN_CARD_SIM_ID)
    assert a.tracker.seen_order() == []           # nothing to debit or cycle
    assert a.unmapped_plays == 1 and a.degraded


def test_own_next_from_cycle_state_and_unknown_is_none(sim):
    from contracts import CycleState
    from track.tracker_adapter import TrackerAdapter
    a = TrackerAdapter("real")
    a.set_own_next_from_cycle(CycleState(hand=(1, 2, 3, 4), next_card=15, queue=()))
    assert a.tracker.own_next() == 15
    a.set_own_next_from_cycle(CycleState(hand=(1, 2, 3, 4), next_card=UNKNOWN_CARD_SIM_ID, queue=()))
    assert a.tracker.own_next() is None


def test_own_play_makes_the_own_next_stale(sim):
    from track.tracker_adapter import TrackerAdapter
    a = TrackerAdapter("real")
    a.set_own_next(15)
    a.on_own_play(30, 6)
    assert a.tracker.own_next() is None            # until the next reading


def test_reset_starts_a_new_match(sim):
    from track.tracker_adapter import TrackerAdapter
    a = TrackerAdapter("real")
    a.on_opponent_play(10, 15)
    a.on_opponent_play(20, UNKNOWN_CARD_SIM_ID)
    a.reset()
    assert a.tracker.seen_order() == [] and a.unmapped_plays == 0 and not a.degraded
