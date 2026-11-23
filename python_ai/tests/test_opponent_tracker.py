"""The opponent tracker core on synthetic event streams (no engine).

The core is pure Python so perception runs the same code on vision events. These
tests pin its rules: the cycle, event-timed elixir, the floor correction,
Mirror, Spirit Empress, the Collector schedule, ability spend and the feature
layout. Conformance against the real engine is test_tracker_conformance.py.
"""
import numpy as np
import pytest

from python_ai.tracking import opponent_tracker as ot
from python_ai.tracking.opponent_tracker import (CardFacts, OpponentTracker,
                                                 TrackerRules)

N = 16
MAX_MATCH = 280.0
MIRROR, EMPRESS, COLLECTOR = 10, 11, 12

# ids 0..7 are an eight-card deck; costs 1..8 -> cost = id + 1 capped to keep
# plays affordable in tests that start from 5 elixir.
COSTS = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 1.0, 5: 2.0, 6: 3.0, 7: 4.0,
         8: 5.0, 9: 5.0, MIRROR: 3.0, EMPRESS: 3.0, COLLECTOR: 6.0}
KINDS = {MIRROR: "mirror", EMPRESS: "spirit_empress", COLLECTOR: "elixir_collector"}


def engine_like_rules(**kw):
    """The engine's numbers, spelled out here only for the synthetic tests; the
    real ENGINE preset is built from the bindings (engine_feed.rules_from_engine)."""
    base = dict(regen_per_tick=0.035,
                phase_schedule=((0, 1.0), (1200, 2.0), (1800, 3.0)),
                start_elixir=5.0, cap=10.0, hand_size=4,
                collector_first_delay=160, collector_period=150,
                collector_lifetime=300, collector_amount=1.0)
    base.update(kw)
    return TrackerRules(**base)


def make(rules=None, **kw):
    return OpponentTracker(rules or engine_like_rules(), CardFacts(COSTS, KINDS), N,
                           max_match_elixir=MAX_MATCH, **kw)


def play_all(tr, cards, start=0, gap=40):
    t = start
    for c in cards:
        tr.on_play(t, c)
        t += gap
    return t


# ---------------------------------------------------------------- cycle

def test_cycle_before_four_plays_cards_are_out_and_next_unknown():
    tr = make()
    tr.on_play(10, 0)
    tr.on_play(50, 1)
    assert tr.hand_known() == set()
    assert tr.next_card() is None
    assert tr.unknown_in_hand() == 4
    assert tr.distinct_seen() == 2


def test_cycle_after_four_plays_next_is_first_of_last_four():
    tr = make()
    for i, c in enumerate([0, 1, 2, 3, 4]):
        tr.on_play(10 + 40 * i, c)
    # plays p1..p5 = 0,1,2,3,4; next = p(n-3) = p2 = 1; queue = 1,2,3,4
    assert tr.next_card() == 1
    # seen {0,1,2,3,4} minus last four plays {1,2,3,4} = {0} in hand
    assert tr.hand_known() == {0}
    assert tr.unknown_in_hand() == 3


def test_cycle_exactly_four_plays_next_is_first_play():
    tr = make()
    play_all(tr, [0, 1, 2, 3], gap=30)
    assert tr.next_card() == 0
    assert tr.hand_known() == set()
    assert tr.unknown_in_hand() == 4


def test_cycle_after_all_eight_seen_hand_is_the_rest_of_the_deck():
    tr = make()
    seq = [0, 1, 2, 3, 4, 5, 6, 7]
    t = 0
    for c in seq:
        tr.advance(t + 100)       # plenty of elixir: no corrections wanted
        tr.on_play(t + 100, c)
        t += 100
    assert tr.distinct_seen() == 8
    assert tr.unknown_in_hand() == 0
    assert tr.hand_known() == {0, 1, 2, 3}      # deck minus last four plays 4..7
    assert tr.next_card() == 4
    # the cycle turns: card 4 is played again, the hand is now deck minus 5,6,7,4
    tr.on_play(t + 100, 4)
    assert tr.next_card() == 5
    assert tr.hand_known() == {0, 1, 2, 3}


def test_hand_has_exactly_hand_size_members_once_eight_seen():
    tr = make()
    t = 0
    for c in [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]:
        t += 120
        tr.on_play(t, c)
    assert len(tr.hand_known()) + tr.unknown_in_hand() == 4
    assert tr.unknown_in_hand() == 0


# ---------------------------------------------------------------- elixir

def test_regen_is_event_timed_across_phases():
    tr = make()
    tr.advance(100)
    assert tr.elixir_estimate() == pytest.approx(5.0 + 100 * 0.035, abs=1e-4)
    tr2 = make(engine_like_rules(start_elixir=0.0))
    tr2.advance(1300)
    expect = 0.0
    for t in range(1, 1301):
        expect = min(expect + 0.035 * (1.0 if t < 1200 else 2.0), 10.0)
    assert tr2.elixir_estimate() == pytest.approx(expect, abs=1e-4)
    tr3 = make(engine_like_rules(start_elixir=0.0))
    tr3.advance(1900)
    # capped by 3x phase: 0.105/tick, reaches the cap well before 1900
    assert tr3.elixir_estimate() == pytest.approx(10.0)


def test_phase_multiplier_boundaries_follow_the_engine_rule():
    r = engine_like_rules()
    assert r.multiplier_at(1) == 1.0
    assert r.multiplier_at(1199) == 1.0
    assert r.multiplier_at(1200) == 2.0
    assert r.multiplier_at(1799) == 2.0
    assert r.multiplier_at(1800) == 3.0


def test_cap_holds_and_play_debits_from_the_capped_value():
    tr = make()
    tr.advance(500)                 # would be 22 elixir uncapped
    assert tr.elixir_estimate() == pytest.approx(10.0)
    tr.on_play(500, 3)              # cost 4
    assert tr.elixir_estimate() == pytest.approx(6.0)
    assert tr.corrections() == 0


def test_debit_at_the_plays_own_tick_not_the_decision_boundary():
    a = make()
    a.on_play(3, 3)      # 5 + 3*0.035 - 4
    a.advance(10)
    b = make()
    b.advance(10)
    b.on_play(10, 3)     # debited at the boundary: same total only if no cap
    assert a.elixir_estimate() == pytest.approx(b.elixir_estimate(), abs=1e-4)
    # but a cap between the two makes the order matter, which is the point
    c = make(engine_like_rules(start_elixir=9.99))
    c.on_play(1, 0)      # regen to 10 (cap), then -1 = 9
    c.advance(10)
    d = make(engine_like_rules(start_elixir=9.99))
    d.advance(10)        # capped at 10
    d.on_play(10, 0)     # 9
    assert c.elixir_estimate() > d.elixir_estimate() + 0.2


def test_floor_correction_raises_to_cost_then_debits():
    tr = make(engine_like_rules(start_elixir=0.0))
    tr.on_play(1, 3)                 # cost 4 with ~0.035 elixir: impossible
    assert tr.corrections() == 1
    assert tr.elixir_estimate() == pytest.approx(0.0, abs=1e-6)
    tr.on_play(2, 0)                 # cost 1, estimate ~0.035: another correction
    assert tr.corrections() == 2


def test_no_correction_when_affordable():
    tr = make()
    tr.on_play(1, 2)
    assert tr.corrections() == 0


def test_mirror_costs_last_non_mirror_plus_one_and_records_its_own_id():
    tr = make(engine_like_rules(start_elixir=10.0))
    tr.on_play(0, 3)                  # cost 4 -> 6
    before = tr.elixir_estimate()
    tr.on_play(0, MIRROR)             # 4 + 1 = 5
    assert tr.elixir_estimate() == pytest.approx(before - 5.0, abs=1e-4)
    assert tr.seen_order()[-1] == MIRROR
    # a second Mirror still copies the last NON-mirror play
    tr.advance(200)                   # back to the cap
    e = tr.elixir_estimate()
    tr.on_play(200, MIRROR)
    assert tr.elixir_estimate() == pytest.approx(e - 5.0, abs=1e-4)
    assert tr.corrections() == 0


def test_mirror_with_nothing_to_copy_is_an_error_not_a_free_play():
    tr = make()
    with pytest.raises(ValueError):
        tr.on_play(5, MIRROR)


def test_spirit_empress_cost_depends_on_estimate():
    hi = make(engine_like_rules(start_elixir=8.0))
    hi.on_play(0, EMPRESS)
    assert hi.elixir_estimate() == pytest.approx(2.0, abs=1e-4)    # paid 6
    lo = make(engine_like_rules(start_elixir=5.0))
    lo.on_play(0, EMPRESS)
    assert lo.elixir_estimate() == pytest.approx(2.0, abs=1e-4)    # paid 3
    edge = make(engine_like_rules(start_elixir=6.0))
    edge.on_play(0, EMPRESS)
    assert edge.elixir_estimate() == pytest.approx(0.0, abs=1e-4)  # exactly 6 -> 6


# ---------------------------------------------------------------- collector

def _collector_gain(rules, horizon):
    """Ticks at which a Collector played at tick 0 credits elixir. The estimate
    is zeroed before each tick (test-only poke) so the cap cannot hide a pulse."""
    tr = make(rules)
    tr.on_play(0, COLLECTOR)
    pulses = []
    for t in range(1, horizon + 1):
        tr._elixir = np.float32(0.0)
        tr.advance(t)
        if tr.elixir_estimate() > 0.5:
            pulses.append(t)
    return pulses


def test_collector_schedule_engine_preset_is_a_single_pulse():
    r = engine_like_rules(start_elixir=10.0)
    assert _collector_gain(r, 1300) == [160]


def test_collector_schedule_real_game_preset_pulses_until_lifetime():
    r = ot.real_game_rules(double_elixir_tick=1200, triple_elixir_tick=1800)
    r = TrackerRules(**{**r.__dict__, "start_elixir": 10.0})
    assert r.collector_lifetime == 1100
    assert _collector_gain(r, 1300) == [160, 310, 460, 610, 760, 910, 1060]


def test_collector_pulse_lands_before_a_same_tick_play():
    r = engine_like_rules(start_elixir=0.0)
    tr = make(r.replace(start_elixir=10.0))
    tr.on_play(0, COLLECTOR)               # 4
    tr._elixir = np.float32(0.0)           # test-only poke: stay clear of the cap
    tr.advance(159)
    e159 = tr.elixir_estimate()
    tr.advance(160)
    assert tr.elixir_estimate() == pytest.approx(e159 + 0.035 + 1.0, abs=1e-4)


def test_destroyed_collector_stops_its_pulses():
    r = ot.real_game_rules(1200, 1800).replace(start_elixir=10.0)
    tr = make(r)
    tr.on_play(0, COLLECTOR)
    tr.on_collector_destroyed(200)            # after the first pulse at 160
    pulses = []
    for t in range(201, 1300):
        tr._elixir = np.float32(0.0)
        tr.advance(t)
        if tr.elixir_estimate() > 0.5:
            pulses.append(t)
    assert pulses == []
    tr.on_collector_destroyed(1300)           # none live: a no-op, not an error


def test_real_game_preset_values():
    r = ot.real_game_rules(double_elixir_tick=1200, triple_elixir_tick=1800)
    assert r.regen_per_tick == pytest.approx(1.0 / 28.0)       # 2.8 s per elixir
    assert r.collector_lifetime == 1100
    assert r.start_elixir == 5.0 and r.cap == 10.0 and r.hand_size == 4


# ---------------------------------------------------------------- abilities

def test_opponent_ability_debits_and_is_counted():
    tr = make(engine_like_rules(start_elixir=8.0))
    tr.on_ability(0, 2.0)
    assert tr.elixir_estimate() == pytest.approx(6.0, abs=1e-4)
    assert tr.opp_ability_spend() == pytest.approx(2.0)
    tr.on_ability(10, 3.0)
    assert tr.opp_ability_spend() == pytest.approx(5.0)


def test_ability_floor_correction():
    tr = make(engine_like_rules(start_elixir=0.0))
    tr.on_ability(1, 2.0)
    assert tr.corrections() == 1


def test_grant_credits_and_caps():
    tr = make(engine_like_rules(start_elixir=9.5))
    tr.on_elixir_grant(0, 2.0)
    assert tr.elixir_estimate() == pytest.approx(10.0)
    tr2 = make(engine_like_rules(start_elixir=4.0))
    tr2.on_elixir_grant(0, 2.0)
    assert tr2.elixir_estimate() == pytest.approx(6.0, abs=1e-4)


def test_own_ability_spend_and_own_next():
    tr = make()
    assert tr.own_next() is None
    tr.set_own_next(3)
    tr.on_own_ability(2.0)
    f = tr.features(0)
    off = ot.feature_offsets(N)
    assert f[off["own_next"] + 3] == 1.0
    assert f[off["scalars"] + ot.SCALAR_INDEX["own_next_known"]] == 1.0
    assert f[off["scalars"] + ot.SCALAR_INDEX["own_ability_spend"]] == pytest.approx(2.0 / MAX_MATCH)
    tr.set_own_next(None)
    assert tr.features(0)[off["own_next"] + 3] == 0.0


def test_time_must_not_run_backwards():
    tr = make()
    tr.advance(50)
    with pytest.raises(ValueError):
        tr.on_play(10, 0)


def test_unknown_card_id_is_an_error():
    tr = make()
    with pytest.raises(KeyError):
        tr.on_play(1, 99)


# ---------------------------------------------------------------- features

def test_feature_width_and_offsets_are_forward_and_documented():
    assert ot.feature_width(256) == 776
    off = ot.feature_offsets(256)
    assert off["opp_in_hand"] == 0
    assert off["opp_next"] == 256
    assert off["own_next"] == 512
    assert off["scalars"] == 768
    assert ot.NUM_SCALARS == 8
    assert off["scalars"] + ot.NUM_SCALARS == 776
    tr = OpponentTracker(engine_like_rules(), CardFacts(COSTS, KINDS), 256,
                         max_match_elixir=MAX_MATCH)
    f = tr.features(0)
    assert f.shape == (776,) and f.dtype == np.float32


def test_feature_values_after_five_plays():
    tr = make()
    t = 0
    for c in [0, 1, 2, 3, 4]:
        t += 60
        tr.on_play(t, c)
    f = tr.features(t)
    off = ot.feature_offsets(N)
    S = off["scalars"]
    ix = ot.SCALAR_INDEX
    assert f[off["opp_in_hand"] + 0] == 1.0
    assert f[off["opp_in_hand"]:off["opp_in_hand"] + N].sum() == 1.0
    assert f[off["opp_next"] + 1] == 1.0 and f[off["opp_next"]:off["opp_next"] + N].sum() == 1.0
    assert f[S + ix["opp_unknown_in_hand"]] == pytest.approx(3 / 4)
    assert f[S + ix["opp_next_known"]] == 1.0
    assert f[S + ix["opp_distinct_seen"]] == pytest.approx(5 / 8)
    assert f[S + ix["opp_elixir"]] == pytest.approx(tr.elixir_estimate() / 10.0, abs=1e-6)
    assert f[S + ix["corrections"]] == pytest.approx(min(1.0, tr.corrections() / 10))
    assert f[S + ix["own_next_known"]] == 0.0
    assert f[S + ix["opp_ability_spend"]] == 0.0


def test_features_before_any_play():
    tr = make()
    f = tr.features(0)
    off = ot.feature_offsets(N)
    S = off["scalars"]
    assert f[:off["scalars"]].sum() == 0.0
    assert f[S + ot.SCALAR_INDEX["opp_unknown_in_hand"]] == 1.0
    assert f[S + ot.SCALAR_INDEX["opp_elixir"]] == pytest.approx(0.5)


def test_correction_scalar_saturates_at_one():
    tr = make(engine_like_rules(start_elixir=0.0))
    for i in range(12):
        tr.on_play(i + 1, 0)
    f = tr.features(12)
    assert f[ot.feature_offsets(N)["scalars"] + ot.SCALAR_INDEX["corrections"]] == 1.0


def test_features_do_not_mutate_between_calls_at_the_same_tick():
    tr = make()
    tr.on_play(5, 0)
    a = tr.features(20).copy()
    b = tr.features(20)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------- reset

def test_reset_returns_to_the_opening_state():
    tr = make()
    play_all(tr, [0, 1, 2, 3, 4], gap=60)
    tr.on_ability(300, 1.0)
    tr.on_own_ability(2.0)
    tr.set_own_next(2)
    tr.reset()
    fresh = make()
    assert np.array_equal(tr.features(0), fresh.features(0))
    assert tr.corrections() == 0 and tr.seen_order() == []
    tr.on_play(1, 0)        # time restarts: no "backwards" error after reset


def test_core_imports_neither_torch_nor_the_engine():
    """Perception runs this module on vision events: no torch, no engine."""
    import ast
    src = open(ot.__file__).read()
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert not mods & {"torch", "clash_royale_env", "python_ai"}, mods
