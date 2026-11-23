"""The opponent tracker against engine truth, in real simulator games.

Tests may read the opponent's hand and elixir, but only to COMPARE: the tracker
and the feed see plays and ability presses only. Measured on 2026-11-22 (see the
numbers in each docstring); every game is seeded and deterministic per build.
"""
import random

import numpy as np
import pytest

import clash_royale_env as E

from python_ai.tracking import engine_feed as ef
from python_ai.tracking import opponent_tracker as ot

CE = E.ClashRoyaleEnv
NONE = E.TowerTroopType.NONE

MINION_GIANT = [129, 184, 126, 183, 0, 31, 24, 101]     # Hero Ice Wizard at index 1
HOG = [15, 6, 25, 40, 24, 72, 33, 7]
MIRROR_DECK = [15, 6, 25, 40, 24, 72, 164, 7]
COLLECTOR_DECK = [15, 6, 25, 40, 24, 72, 99, 7]
TOL = 1e-4


def new_env(deck0, deck1, seed):
    env = CE(list(deck0), list(deck1), 3600, NONE, NONE)
    env.seed(seed)
    return env


def legal_play(env, team, rng, p):
    """A random legal play (card index, x, y in the step_self_play frame) with
    probability p; (-1, 0, 0) otherwise. Team 1's y is mirrored back."""
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


def ability_slots(deck):
    return [s for s in (1, 2) if len(deck) > s
            and (E.get_card_info(deck[s])["is_hero"] or E.get_card_info(deck[s])["is_champion"])]


def play_match(deck0, deck1, seed, p=0.3, team1_activates=True, multiplier=None,
               max_decisions=400, observer=0, p_observer=None):
    """Random legal plays by both sides; team 1 (the opponent of `observer`=0)
    presses its ability whenever ready. Returns a dict of per-decision truth
    comparisons and the feed."""
    env = new_env(deck0, deck1, seed)
    if multiplier is not None:
        env.set_opponent_elixir_multiplier(multiplier)
    tr = ef.make_engine_tracker()
    feed = ef.EngineFeed(env, observer, tr, decks=(deck0, deck1))
    rng = random.Random(seed)
    opp = 1 - observer
    errs, hand_viol, next_viol, claims, unknown_viol = [], 0, 0, 0, 0
    opp_deck = (deck0, deck1)[opp]
    for _ in range(max_decisions):
        if env.is_game_over():
            break
        a = [None, None]
        a[observer] = legal_play(env, observer, rng, p if p_observer is None else p_observer)
        a[opp] = legal_play(env, opp, rng, p)
        press = False
        if team1_activates:
            for s in ability_slots(opp_deck):
                if env.is_champion_ability_ready(opp, s):
                    feed.note_activation(opp, s, True)
                    press = True
                    break
        flags = [False, False, False, False]
        if press:
            flags[2 + 0 if opp == 1 else 0] = True
        env.step_self_play(*a[0], *a[1], 10, *flags)
        feed.poll()
        tr.advance(env.get_current_tick())
        # ---- comparison (the only place truth is read for the opponent)
        errs.append(tr.elixir_estimate() - env.get_elixir_for_team(opp))
        true_hand = set(env.get_hand_for_team(opp))
        known = tr.hand_known()
        if not known <= true_hand:
            hand_viol += 1
        if len(true_hand - known) != tr.unknown_in_hand():
            unknown_viol += 1
        nxt = tr.next_card()
        if nxt is not None and not env.is_game_over():   # a finished match has no next play
            claims += 1
            if nxt != ef.probe_next_card(env, opp):
                next_viol += 1
    return {"env": env, "tracker": tr, "feed": feed, "errs": errs, "hand_viol": hand_viol,
            "unknown_viol": unknown_viol, "next_viol": next_viol, "next_claims": claims}


def assert_exact(r):
    assert r["errs"], "the match produced no decisions"
    assert max(abs(e) for e in r["errs"]) < TOL
    assert r["tracker"].corrections() == 0
    assert r["hand_viol"] == 0
    assert r["unknown_viol"] == 0
    assert r["next_viol"] == 0


# ------------------------------------------------------------ rules / facts

def test_engine_rules_match_the_live_engine():
    rules = ef.rules_from_engine()
    for t in (1, 1199, 1200, 1799, 1800, 3000):
        assert rules.multiplier_at(t) == E.elixir_multiplier_at_tick(t), t
    assert rules.hand_size == CE.HAND_SIZE
    env = new_env(HOG, HOG, 1)
    assert env.get_elixir_for_team(1) == rules.start_elixir
    # regeneration, measured over 200 idle 1x ticks from a known level
    env.set_elixir_for_team(1, 0.0)
    for _ in range(200):
        env.step_self_play(-1, 0, 0, -1, 0, 0, 1)
    assert env.get_elixir_for_team(1) == pytest.approx(200 * rules.regen_per_tick, abs=1e-3)
    env.set_elixir_for_team(1, rules.cap)
    env.step_self_play(-1, 0, 0, -1, 0, 0, 1)
    assert env.get_elixir_for_team(1) == rules.cap


def test_card_facts_identify_special_cards_by_name():
    facts = ef.card_facts_from_engine()
    assert facts.kind(164) == ot.MIRROR and facts.kind(188) == ot.MIRROR
    assert facts.kind(165) == ot.SPIRIT_EMPRESS
    assert facts.kind(99) == ot.ELIXIR_COLLECTOR
    assert facts.kind(15) == ot.NORMAL
    assert facts.cost(15) == E.get_card_info(15)["cost"]
    with pytest.raises(KeyError):
        facts.cost(255)         # an unregistered id is an error, not a free card
    # every id the engine knows is covered; nothing is keyed to an id here
    for cid in range(CE.NUM_CARD_IDS):
        try:
            info = E.get_card_info(cid)
        except ValueError:
            continue
        assert facts.cost(cid) == info["cost"]


def test_engine_tracker_feature_width():
    tr = ef.make_engine_tracker()
    assert tr.features(0).shape == (ot.feature_width(CE.NUM_CARD_IDS),)
    assert ot.feature_width(CE.NUM_CARD_IDS) == 776


# -------------------------------------------------------------- the feed

def test_first_poll_after_reset_does_not_swallow_a_tick_zero_play():
    env = new_env(HOG, HOG, 2)
    tr = ef.make_engine_tracker()
    feed = ef.EngineFeed(env, 0, tr, decks=(HOG, HOG))
    rng = random.Random(0)
    env.set_elixir_for_team(1, 10.0)
    a1 = legal_play(env, 1, rng, 1.0)
    assert a1[0] >= 0
    card = env.get_hand_for_team(1)[a1[0]]
    env.step_self_play(-1, 0, 0, *a1, 10)
    feed.poll()
    assert tr.seen_order() == [card]
    assert tr.distinct_seen() == 1
    feed.poll()                       # polling again emits nothing new
    assert tr.seen_order() == [card]


def test_plays_are_emitted_with_their_own_tick():
    env = new_env(HOG, HOG, 3)
    tr = ef.make_engine_tracker()
    feed = ef.EngineFeed(env, 0, tr, decks=(HOG, HOG))
    rng = random.Random(1)
    for _ in range(20):
        env.step_self_play(-1, 0, 0, *legal_play(env, 1, rng, 0.6), 10)
        feed.poll()
    assert feed.stats["plays"] == len(tr.seen_order()) > 3
    for c in tr.seen_order():
        assert env.get_last_played_tick(1, c) >= 0


# ------------------------------------------------------------ conformance

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_conformance_hog_deck(seed):
    """Exact elixir (|err| < 1e-4 every decision), hand and next card."""
    assert_exact(play_match(HOG, HOG, seed, p=0.25))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_conformance_minion_giant_mirror_with_hero_ability(seed):
    """Team 1 presses Hero Ice Wizard's 2.0 ability whenever ready.

    Measured 2026-11-22 over seeds 0-5: the press is counted at its own tick and
    the elixir is exact (max |err| 0.0000) EXCEPT when the Hero dies armed (its
    first hit has not landed) and the engine refunds 2.0
    (GameManager.h:124-136, refundArmedAbilities). That refund is not a visible
    event, so the estimate then sits 2.0 BELOW the engine until the 10 cap
    absorbs it (seed 5: -2.0 at tick 460, 0.0 again by 1560). The error is
    therefore one-sided and bounded by 2.0 per press; the cycle stays exact.
    """
    r = play_match(MINION_GIANT, MINION_GIANT, seed, p=0.3)
    presses = r["feed"].stats["abilities_counted"]
    assert r["errs"]
    assert max(r["errs"]) < TOL                          # never above the engine
    assert min(r["errs"]) > -2.0 * presses - TOL         # at most one refund per press
    assert r["hand_viol"] == 0 and r["unknown_viol"] == 0 and r["next_viol"] == 0
    if seed == 0:
        # the ability really was pressed and counted (a control for 'no presses')
        assert presses >= 1
        assert r["tracker"].opp_ability_spend() == pytest.approx(2.0 * presses)


def test_conformance_covers_the_triple_elixir_phase():
    seen_late = 0
    for seed in range(4):
        r = play_match(HOG, HOG, seed, p=0.12)
        assert_exact(r)
        seen_late += r["env"].get_current_tick() > 1800
    assert seen_late >= 1, "no game reached 3x elixir: the phase was not exercised"


def test_conformance_deck_with_mirror():
    plays = 0
    for seed in range(8):
        r = play_match(MIRROR_DECK, MIRROR_DECK, seed, p=0.3)
        assert_exact(r)
        plays += r["tracker"].seen_order().count(164)
    assert plays >= 1, "Mirror was never played: the rule was not exercised"


def test_conformance_deck_with_elixir_collector():
    """Exact while the Collector survives (the observer stays passive; a
    destroyed Collector is a separate, pinned case below)."""
    plays = 0
    for seed in range(8):
        r = play_match(COLLECTOR_DECK, COLLECTOR_DECK, seed, p=0.3, p_observer=0.0)
        assert_exact(r)
        plays += r["tracker"].seen_order().count(99)
    assert plays >= 1, "the Collector was never played: the rule was not exercised"


def test_destroyed_collector_error_profile_and_its_hook():
    """A Collector destroyed before its pulse is visible to a viewer (the
    building is gone) but not to the feed: the tracker then credits a pulse the
    engine never pays, +1.0 above the engine. on_collector_destroyed repairs
    it exactly."""
    def run(report):
        env = new_env(COLLECTOR_DECK, COLLECTOR_DECK, 5)
        tr = ef.make_engine_tracker()
        feed = ef.EngineFeed(env, 0, tr, decks=(COLLECTOR_DECK, COLLECTOR_DECK))
        env.set_hand_for_team(1, [99, 15, 6, 25])
        env.set_elixir_for_team(1, 10.0)
        tr._elixir = np.float32(10.0)           # test-only poke: the tracker sees the same bar
        x, y = next((x, y) for x in range(18) for y in range(34)
                    if env.is_valid_placement(99, x, y, 1) and y > 24)
        env.step_self_play(-1, 0, 0, 0, x, 33 - y, 1)
        feed.poll()
        for _ in range(2):                      # team 0 knights on top of it
            env.inject(0, x, y, 0, -1, 0)
        for _ in range(100):
            env.step_self_play(-1, 0, 0, -1, 0, 0, 1)
            feed.poll()
        if report:
            tr.on_collector_destroyed(env.get_current_tick())
        # keep both bars at zero through tick 160 so the cap cannot hide a pulse
        tr.advance(env.get_current_tick())
        env.set_elixir_for_team(1, 0.0)
        tr._elixir = np.float32(0.0)
        worst = 0.0
        for _ in range(80):
            env.step_self_play(-1, 0, 0, -1, 0, 0, 1)
            feed.poll()
            tr.advance(env.get_current_tick())
            err = tr.elixir_estimate() - env.get_elixir_for_team(1)
            worst = max(worst, abs(err))
            env.set_elixir_for_team(1, 0.0)
            tr._elixir = np.float32(0.0)        # test-only poke
        assert env.get_current_tick() > 170 and not env.is_game_over()
        return worst
    assert run(report=False) == pytest.approx(1.0, abs=0.05)
    assert run(report=True) < TOL


def test_engine_collector_is_a_single_pulse_at_play_plus_160():
    env = new_env(COLLECTOR_DECK, COLLECTOR_DECK, 5)
    env.set_hand_for_team(1, [99, 15, 6, 25])
    env.set_elixir_for_team(1, 10.0)
    cell = next((x, y) for x in range(18) for y in range(34)
                if env.is_valid_placement(99, x, y, 1) and y > 24)
    env.step_self_play(-1, 0, 0, 0, cell[0], 33 - cell[1], 1)
    assert env.get_last_played_tick(1, 99) == 0
    pulses = []
    for _ in range(700):
        env.set_elixir_for_team(1, 0.0)
        env.step_self_play(-1, 0, 0, -1, 0, 0, 1)
        if env.get_elixir_for_team(1) > 0.5:
            pulses.append(env.get_current_tick())
    assert pulses == [160]


def test_a_hidden_multiplier_makes_the_estimate_diverge():
    """CONTROL THAT MUST FIRE: a 1.5x opponent regenerates faster than the
    tracker believes, so the comparison is live: errors appear and the floor
    correction triggers."""
    r = play_match(HOG, HOG, 1, p=0.25, multiplier=1.5)
    assert max(abs(e) for e in r["errs"]) > 0.3
    assert r["tracker"].corrections() > 0
    # the cycle does not depend on elixir, so it stays exact
    assert r["hand_viol"] == 0 and r["next_viol"] == 0


# --------------------------------------------------------------- abilities

def _hero_on_board(seed=5):
    env = new_env(MINION_GIANT, MINION_GIANT, seed)
    tr = ef.make_engine_tracker()
    feed = ef.EngineFeed(env, 0, tr, decks=(MINION_GIANT, MINION_GIANT))
    env.set_hand_for_team(1, [184, 126, 129, 183])
    env.set_elixir_for_team(1, 10.0)
    tr._elixir = np.float32(10.0)           # test-only poke: the tracker sees the same bar
    env.step_self_play(-1, 0, 0, 0, 6, 11, 10)
    feed.poll()
    env.step_self_play(-1, 0, 0, -1, 0, 0, 10)
    feed.poll()
    assert env.is_champion_ability_ready(1, 1)
    return env, tr, feed


def test_ability_cost_is_measured_on_a_fork_and_cached():
    env, tr, feed = _hero_on_board()
    before = env.get_elixir_for_team(1)
    assert ef.measure_ability_cost(env, 1, 1) == 2.0
    assert env.get_elixir_for_team(1) == before          # the live match is untouched
    assert env.is_champion_ability_ready(1, 1)
    feed.note_activation(1, 1, True)
    assert feed.ability_costs == {184: 2.0}


def test_ability_refused_for_elixir_after_a_same_tick_card_is_not_counted():
    """The engine plays the card first (GameManager.h runSelfPlayTicks), then
    refuses an ability it can no longer pay for. Hero + Minion Giant (4) with
    5 elixir: ready before the step, refused during it. The spend must not be
    counted, and the estimate must still match the engine."""
    env, tr, feed = _hero_on_board()
    tr.advance(env.get_current_tick())
    env.set_elixir_for_team(1, 5.0)
    tr._elixir = np.float32(5.0)          # test-only poke: the tracker sees the same bar
    assert env.is_champion_ability_ready(1, 1)           # 5 >= 2 before the step
    hand = env.get_hand_for_team(1)
    giant = hand.index(183)
    cooldown = env.get_hand_cooldown_ticks(1)[giant]
    assert cooldown == 0
    feed.note_activation(1, 1, True)
    spent_before = tr.opp_ability_spend()
    env.step_self_play(-1, 0, 0, giant, 6, 33 - 20, 1, False, False, True, False)
    feed.poll()
    tr.advance(env.get_current_tick())
    assert 183 in tr.seen_order()
    assert tr.opp_ability_spend() == spent_before        # refused: not counted
    assert feed.stats["abilities_refused"] == 1
    assert tr.elixir_estimate() == pytest.approx(env.get_elixir_for_team(1), abs=TOL)


def test_ability_accepted_alongside_a_same_tick_card_is_counted():
    env, tr, feed = _hero_on_board()
    tr.advance(env.get_current_tick())
    env.set_elixir_for_team(1, 7.0)
    tr._elixir = np.float32(7.0)
    giant = env.get_hand_for_team(1).index(183)
    feed.note_activation(1, 1, True)
    env.step_self_play(-1, 0, 0, giant, 6, 33 - 20, 1, False, False, True, False)
    feed.poll()
    tr.advance(env.get_current_tick())
    assert tr.opp_ability_spend() == 2.0
    assert tr.elixir_estimate() == pytest.approx(env.get_elixir_for_team(1), abs=TOL)


def test_an_unready_press_spends_nothing():
    env, tr, feed = _hero_on_board()
    feed.note_activation(1, 1, False)
    assert feed.stats["abilities_counted"] == 0 and tr.opp_ability_spend() == 0.0


def test_unseen_grant_error_profile_and_the_hook_that_repairs_it():
    """A refund (Hero Ice Wizard dying armed, GameManager.h:124-136) is not
    visible to the feed. It could not be provoked from Python in this build
    (the Hero fires on its first landed hit), so the error profile of ANY unseen
    grant is pinned instead: the estimate sits below the engine by the amount
    until the cap absorbs it, and on_elixir_grant repairs it exactly."""
    def run(report):
        env = new_env(HOG, HOG, 4)
        tr = ef.make_engine_tracker()
        ef.EngineFeed(env, 0, tr, decks=(HOG, HOG))
        for _ in range(5):
            env.step_self_play(-1, 0, 0, -1, 0, 0, 10)
        env.set_elixir_for_team(1, env.get_elixir_for_team(1) + 2.0)    # the unseen +2
        if report:
            tr.on_elixir_grant(env.get_current_tick(), 2.0)
        env.step_self_play(-1, 0, 0, -1, 0, 0, 10)
        tr.advance(env.get_current_tick())
        return tr.elixir_estimate() - env.get_elixir_for_team(1)
    assert run(report=False) == pytest.approx(-2.0, abs=1e-3)
    assert abs(run(report=True)) < TOL


def test_own_ability_spend_counts_only_accepted_presses():
    env, tr, feed = _hero_on_board_own()
    feed.note_activation(0, 1, True)
    env.step_self_play(-1, 0, 0, -1, 0, 0, 10, True, False, False, False)
    feed.poll()
    assert tr.own_ability_spend() == 2.0
    assert feed.stats["own_abilities_counted"] == 1


def _hero_on_board_own(seed=6):
    env = new_env(MINION_GIANT, MINION_GIANT, seed)
    tr = ef.make_engine_tracker()
    feed = ef.EngineFeed(env, 0, tr, decks=(MINION_GIANT, MINION_GIANT))
    env.set_hand_for_team(0, [184, 126, 129, 183])
    env.set_elixir_for_team(0, 10.0)
    env.step_self_play(0, 6, 11, -1, 0, 0, 10)
    env.step_self_play(-1, 0, 0, -1, 0, 0, 10)
    assert env.is_champion_ability_ready(0, 1)
    return env, tr, feed


def test_own_ability_refused_by_a_same_tick_card_is_not_counted():
    env, tr, feed = _hero_on_board_own()
    env.set_elixir_for_team(0, 5.0)
    giant = env.get_hand_for_team(0).index(183)
    feed.note_activation(0, 1, True)
    env.step_self_play(giant, 6, 11, -1, 0, 0, 1, True, False, False, False)
    feed.poll()
    assert tr.own_ability_spend() == 0.0
    assert feed.stats["own_abilities_refused"] == 1


# ------------------------------------------------------------ own next card

@pytest.mark.parametrize("team", [0, 1])
def test_probe_next_card_equals_the_card_that_enters_the_hand(team):
    env = new_env(HOG, HOG, 11 + team)
    rng = random.Random(team)
    checked = 0
    for _ in range(120):
        if env.is_game_over():
            break
        env.set_elixir_for_team(team, 10.0)
        play = legal_play(env, team, rng, 1.0)
        if play[0] < 0:
            env.step_self_play(-1, 0, 0, -1, 0, 0, 10)
            continue
        hand_before = list(env.get_hand_for_team(team))
        predicted = ef.probe_next_card(env, team)
        assert predicted is not None
        if team == 0:
            env.step_self_play(*play, -1, 0, 0, 10)
        else:
            env.step_self_play(-1, 0, 0, *play, 10)
        hand_after = list(env.get_hand_for_team(team))
        entered = [c for c in hand_after if c not in hand_before]
        # probing a DIFFERENT slot than the one played still names the same
        # next card: the queue front does not depend on the slot
        assert entered == [predicted]
        checked += 1
    assert checked >= 20


def test_probe_does_not_change_the_live_match():
    env = new_env(HOG, HOG, 9)
    env.set_elixir_for_team(0, 3.0)
    snap = (env.get_elixir_for_team(0), list(env.get_hand_for_team(0)),
            env.get_current_tick(), list(env.get_observation_for_team(0)))
    ef.probe_next_card(env, 0)
    assert (env.get_elixir_for_team(0), list(env.get_hand_for_team(0)),
            env.get_current_tick(), list(env.get_observation_for_team(0))) == snap


def test_probe_skips_a_mirror_with_nothing_to_copy():
    env = new_env(MIRROR_DECK, MIRROR_DECK, 7)
    assert env.set_hand_for_team(0, [164, 15, 6, 25])
    nxt = ef.probe_next_card(env, 0)
    assert nxt is not None and nxt not in (164, 15, 6, 25)


def test_probe_is_none_when_no_slot_can_be_played():
    env = new_env(HOG, HOG, 8)
    env.set_hand_for_team(0, [15, 6, 25, 40])
    env.set_elixir_for_team(0, 10.0)
    rng = random.Random(2)
    # play, then immediately all four slots: the new card is locked, the other
    # three are not, so build the all-locked state by playing four in a row
    for _ in range(4):
        env.set_elixir_for_team(0, 10.0)
        a = legal_play(env, 0, rng, 1.0)
        env.step_self_play(*a, -1, 0, 0, 1)
    cooldown = env.get_hand_cooldown_ticks(0)
    if all(c > 0 for c in cooldown):
        assert ef.probe_next_card(env, 0) is None
    else:
        pytest.skip("the engine unlocks slots independently here")


def test_cycle_rule_agrees_with_the_probe_after_four_plays():
    """Cross-check: a tracker following team 0's plays names the card the
    probe says will enter team 0's hand, from the fourth play on."""
    env = new_env(HOG, HOG, 13)
    tr = ef.make_engine_tracker()
    feed = ef.EngineFeed(env, 1, tr, decks=(HOG, HOG), track_own_next=False)
    rng = random.Random(3)
    agree = 0
    for _ in range(80):
        env.set_elixir_for_team(0, 10.0)
        play = legal_play(env, 0, rng, 0.8)
        env.step_self_play(*play, -1, 0, 0, 10)
        feed.poll()
        if env.is_game_over():
            break
        nxt = tr.next_card()
        if nxt is not None:
            assert nxt == ef.probe_next_card(env, 0)
            agree += 1
    assert agree >= 20


def test_feed_publishes_the_own_next_card():
    env = new_env(HOG, HOG, 14)
    tr = ef.make_engine_tracker()
    feed = ef.EngineFeed(env, 0, tr, decks=(HOG, HOG))
    feed.poll()
    assert tr.own_next() == ef.probe_next_card(env, 0)
    probes = feed.stats["probes"]
    feed.poll()
    assert feed.stats["probes"] == probes          # cached until the hand changes
    env.set_elixir_for_team(0, 10.0)
    env.step_self_play(*legal_play(env, 0, random.Random(0), 1.0), -1, 0, 0, 10)
    feed.poll()
    assert feed.stats["probes"] == probes + 1
    f = tr.features(env.get_current_tick())
    off = ot.feature_offsets(CE.NUM_CARD_IDS)
    assert f[off["own_next"] + tr.own_next()] == 1.0


# ------------------------------------------------------- visible-only guard

def test_tracker_code_never_reads_hidden_opponent_state():
    """The tracker and feed may read only visible things. The feed's opponent
    side calls get_last_played_tick(opponent, .); nothing reads the opponent's
    elixir, hand, queue or the engine's spend counters."""
    import ast
    import inspect
    banned = {"get_elixir_for_team", "get_hand_for_team", "get_elixir_spent",
              "get_elixir_spent_on_card", "get_hand", "get_elixir"}
    core = inspect.getsource(ot)
    assert not any(b in core for b in banned)
    feed_src = inspect.getsource(ef)
    tree = ast.parse(feed_src)
    used = {}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and node.attr in banned:
                used.setdefault(fn.name, set()).add(node.attr)
    # Allowed, own-side or probe-side only: the viewer's own bar and spend, the
    # fork probes (which act on the viewer's own team or a measurement fork).
    assert set(used) <= {"note_activation", "_emit_own_press", "_refresh_own_next",
                         "probe_next_card", "measure_ability_cost"}, used


# ------------------------------------------------- review follow-up (F1)

MIRROR_COLLECTOR_DECK = [15, 6, 25, 40, 99, 72, 164, 7]


def test_mirror_of_an_elixir_collector_pulses_like_the_engine():
    """A Mirror copies the Collector's spawn, so the engine pulses for it too.
    Both bars are zeroed every tick so only the pulses move them."""
    D = MIRROR_COLLECTOR_DECK
    env = new_env(D, D, 5)
    tr = ef.make_engine_tracker()
    feed = ef.EngineFeed(env, 0, tr, decks=(D, D), track_own_next=False)
    assert env.set_hand_for_team(1, [99, 164, 15, 6])
    env.set_elixir_for_team(1, 10.0)
    tr._elixir = np.float32(10.0)             # test-only poke: the same bar
    cell = next((x, y) for x in range(18) for y in range(34)
                if env.is_valid_placement(99, x, y, 1) and y > 24)
    env.step_self_play(-1, 0, 0, 0, cell[0], 33 - cell[1], 1)
    feed.poll()
    for _ in range(25):
        env.step_self_play(-1, 0, 0, -1, 0, 0, 1)
    feed.poll()
    env.set_elixir_for_team(1, 10.0)
    tr.advance(env.get_current_tick())
    tr._elixir = np.float32(10.0)
    i = env.get_hand_for_team(1).index(164)
    cell2 = next((x, y) for x in range(18) for y in range(34)
                 if env.is_valid_placement(164, x, y, 1) and y > 24
                 and 3 <= x <= 14 and abs(x - cell[0]) > 3)
    env.step_self_play(-1, 0, 0, i, cell2[0], 33 - cell2[1], 1)
    feed.poll()
    assert tr.seen_order()[:2] == [99, 164]
    tr.advance(env.get_current_tick())
    worst, pulses = 0.0, 0
    for _ in range(400):
        env.set_elixir_for_team(1, 0.0)
        tr._elixir = np.float32(0.0)
        env.step_self_play(-1, 0, 0, -1, 0, 0, 1)
        feed.poll()
        tr.advance(env.get_current_tick())
        e = tr.elixir_estimate()
        pulses += e > 0.5
        worst = max(worst, abs(e - env.get_elixir_for_team(1)))
    assert pulses >= 2, "both Collectors should have pulsed in the window"
    assert worst < 1e-3


def test_mirror_registers_a_collector_only_when_the_copied_card_is_one():
    facts = ot.CardFacts({1: 2.0, 2: 3.0, 3: 2.0},
                         {1: ot.ELIXIR_COLLECTOR, 2: ot.MIRROR})
    rules = ef.rules_from_engine()
    plain = ot.OpponentTracker(rules, facts, 8, 280.0)
    plain.on_play(10, 3)
    plain.on_play(20, 2)
    assert plain._collectors == []
    tr = ot.OpponentTracker(rules, facts, 8, 280.0)
    tr.on_play(10, 1)
    tr.on_play(20, 2)
    assert tr._collectors == [10, 20]


def test_a_press_with_no_same_tick_card_is_counted_even_when_the_estimate_is_low():
    """Readiness includes elixir, and nothing else was played that tick, so the
    engine accepted the press. A drifted estimate must be corrected, not obeyed."""
    env, tr, feed = _hero_on_board()
    tr.advance(env.get_current_tick())
    env.set_elixir_for_team(1, 6.0)
    tr._elixir = np.float32(1.0)             # test-only poke: an estimate that drifted low
    feed.note_activation(1, 1, True)
    env.step_self_play(-1, 0, 0, -1, 0, 0, 10, False, False, True, False)
    feed.poll()
    tr.advance(env.get_current_tick())
    assert feed.stats["abilities_counted"] == 1
    assert feed.stats["abilities_refused"] == 0
    assert tr.opp_ability_spend() == 2.0
    assert tr.corrections() == 1             # the floor correction is visible
