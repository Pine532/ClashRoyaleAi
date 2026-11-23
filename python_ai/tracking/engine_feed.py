"""The simulator feed for the opponent tracker, and the ENGINE rules preset.

`EngineFeed` turns what a viewer of the simulated match could see (the
opponent's plays, with their ticks; the viewer's own ability presses) into
tracker events. It never reads the opponent's elixir, hand, queue or the
engine's spend counters. (The viewer's OWN hidden-to-nobody state, its own
elixir and spend, is used to judge its own ability presses.)

Usage, per decision step::

    feed = EngineFeed(env, observer_team=0, tracker=tracker, decks=(deck0, deck1))
    feed.reset()                       # after env.reset()/env.seed()
    ...
    feed.note_activation(team, slot, was_ready)   # BEFORE the step, per flag
    env.step_self_play(...)
    feed.poll()                        # AFTER the step
    features = tracker.features(env.get_current_tick())
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import clash_royale_env as E

from python_ai.tracking.opponent_tracker import (ELIXIR_COLLECTOR, MIRROR, NORMAL,
                                                 SPIRIT_EMPRESS, CardFacts,
                                                 OpponentTracker, TrackerRules)

CE = E.ClashRoyaleEnv

#: Kind of each special card, matched by NAME (an Evolution reuses its base
#: card's name verbatim, so Mirror 164 and Mirror Evolution 188 both match; no
#: id appears here).
_KIND_BY_NAME = {
    "mirror": MIRROR,
    "spirit empress": SPIRIT_EMPRESS,
    "elixir collector": ELIXIR_COLLECTOR,
}


def rules_from_engine() -> TrackerRules:
    """The ENGINE preset: reproduces the simulator exactly.

    From the bindings: the phase ticks (DOUBLE_ELIXIR_TICK, TRIPLE_ELIXIR_TICK;
    the multipliers 2 and 3 are `elixir_multiplier_at_tick`'s) and the hand
    size (HAND_SIZE). Not bound, hardcoded and asserted against live engines in
    tests/test_tracker_conformance.py:

    * regen 0.035 per tick: GameManager.h ELIXIR_REGEN_RATE (also
      perception/timebase.py SIM_ELIXIR_REGEN_PER_TICK).
    * start 5.0: PlayerState.h initializeDeck (`elixir = 5.0f`).
    * cap 10.0: GameManager::step's std::min(.., 10.0f).
    * Elixir Collector: first pulse at play + 160 (CardFactories.h
      periodicTicksUntilNext = interval 150, counting after the 10-tick deploy,
      DEPLOY_TIME_TICKS in CardStats.h), period 150 (CardRegistry.h id 99,
      withPeriodicEffect(150, ElixirGrantEffect(1.0))), lifetime 300
      (CardStats.h lifetimeTicks default, Building.h: expiry at ticksAlive >=
      lifetime, after that tick's update, so a pulse at play + 300 still
      lands; the 310 one does not: the engine's default gives a SINGLE pulse).
    * Mirror +1 and Spirit Empress 3 / 6 at 6 elixir: GameManager::playCard.
    """
    double_tick, triple_tick = int(CE.DOUBLE_ELIXIR_TICK), int(CE.TRIPLE_ELIXIR_TICK)
    return TrackerRules(
        regen_per_tick=0.035,
        phase_schedule=((0, 1.0), (double_tick, 2.0), (triple_tick, 3.0)),
        start_elixir=5.0, cap=10.0, hand_size=int(CE.HAND_SIZE),
        collector_first_delay=160, collector_period=150, collector_lifetime=300,
        collector_amount=1.0,
    )


def card_facts_from_engine() -> CardFacts:
    """Cost and kind of every registered card id, from get_card_info. Kinds are
    matched by name, never by id."""
    costs: Dict[int, float] = {}
    kinds: Dict[int, str] = {}
    for cid in range(int(CE.NUM_CARD_IDS)):
        try:
            info = E.get_card_info(cid)
        except ValueError:          # an id with no registered card
            continue
        costs[cid] = float(info["cost"])
        kind = _KIND_BY_NAME.get(str(info["name"]).strip().lower())
        if kind is not None:
            kinds[cid] = kind
    return CardFacts(costs, kinds)


def make_engine_tracker() -> OpponentTracker:
    return OpponentTracker(rules_from_engine(), card_facts_from_engine(),
                           int(CE.NUM_CARD_IDS), max_match_elixir=float(CE.MAX_MATCH_ELIXIR))


# ------------------------------------------------------------------ probes

def measure_ability_cost(game, team: int, slot: int) -> float:
    """The elixir a ready Hero/Champion ability costs, by measurement.

    get_card_info does not expose abilityElixirCost. On a snapshot fork the
    team is topped up to 10 and the ability activated; the elixir difference is
    the cost. The live game is untouched. Raises if the fork refuses (the
    caller must pass a slot that is ready apart from elixir).
    """
    fork = game.snapshot()
    fork.set_elixir_for_team(team, 10.0)
    before = fork.get_elixir_for_team(team)
    if not fork.activate_champion_ability(team, slot):
        raise RuntimeError(f"ability probe: team {team} slot {slot} refused on the fork")
    return round(float(before - fork.get_elixir_for_team(team)), 4)


def _legal_cell(game, card_id: int, team: int) -> Optional[Tuple[int, int]]:
    """Some legal board cell for the card, or None. Board coordinates."""
    for y in range(int(CE.BOARD_HEIGHT)):
        for x in range(int(CE.BOARD_WIDTH)):
            if game.is_valid_placement(card_id, x, y, team):
                return x, y
    return None


def probe_next_card(game, team: int) -> Optional[int]:
    """The card that would enter the hand on `team`'s next play, or None.

    On a `snapshot()` fork (never clone(), which is the Clone card): the team is
    topped up to 10 elixir, an unlocked hand slot is played at a legal cell for
    one tick, and the card that now sits in that slot is the next card. A slot
    whose play fails (Mirror with nothing to copy, an illegal cell) is skipped;
    None when no slot can be played.
    """
    hand = list(game.get_hand_for_team(team))
    cooldown = list(game.get_hand_cooldown_ticks(team))
    for i, card in enumerate(hand):
        if cooldown[i] > 0:
            continue
        cell = _legal_cell(game, card, team)
        if cell is None:
            continue
        x, y = cell
        fork = game.snapshot()
        fork.set_elixir_for_team(team, 10.0)
        # step_self_play takes team 1's cell in team 0's frame (y mirrored).
        ym = y if team == 0 else int(CE.BOARD_HEIGHT) - 1 - y
        noop = (-1, 0, 0)
        if team == 0:
            fork.step_self_play_fast(i, x, ym, *noop, 1)
        else:
            fork.step_self_play_fast(*noop, i, x, ym, 1)
        after = list(fork.get_hand_for_team(team))
        if after[i] != card:
            return int(after[i])
    return None


# ------------------------------------------------------------------- feed

class EngineFeed:
    """Feeds `tracker` (which follows the opponent of `observer_team`) from a
    simulated match. See the module docstring for the call order.

    `decks`: (deck of team 0, deck of team 1), used only to key the ability
    cost cache by card id; without it the key is (team, slot).
    """

    def __init__(self, game, observer_team: int, tracker: OpponentTracker,
                 decks: Optional[Sequence[Sequence[int]]] = None,
                 ability_costs: Optional[Dict] = None, track_own_next: bool = True):
        if observer_team not in (0, 1):
            raise ValueError("observer_team must be 0 or 1")
        self.game = game
        self.observer = observer_team
        self.opponent = 1 - observer_team
        self.tracker = tracker
        self.decks = decks
        self.ability_costs: Dict = {} if ability_costs is None else ability_costs
        self.track_own_next = track_own_next
        self._n = int(CE.NUM_CARD_IDS)
        self.stats = {"plays": 0, "abilities_counted": 0, "abilities_refused": 0,
                      "own_abilities_counted": 0, "own_abilities_refused": 0,
                      "probes": 0}
        self.reset()

    def reset(self) -> None:
        """Call right after env.reset()/seed(): the match is at tick 0 with no
        plays, so the baseline is 'never played' and nothing is swallowed."""
        self.tracker.reset()
        self._last = np.full(self._n, -1, dtype=np.int64)
        self._pending: List[dict] = []
        self._own_next_key: Optional[Tuple[int, ...]] = None
        self._own_next: Optional[int] = None
        self.tracker.set_own_next(None)

    # ------------------------------------------------------------ abilities
    def _ability_key(self, team: int, slot: int):
        if self.decks is not None:
            return int(self.decks[team][slot])
        return (team, slot)

    def _ability_cost(self, team: int, slot: int) -> float:
        key = self._ability_key(team, slot)
        if key not in self.ability_costs:
            self.ability_costs[key] = measure_ability_cost(self.game, team, slot)
        return self.ability_costs[key]

    def note_activation(self, team: int, slot: int, was_ready: bool) -> None:
        """The env asked `team` to activate `slot`'s ability on the coming step.
        Call BEFORE the step, with the readiness it used to decide. A press that
        was not ready is a no-op and spends nothing."""
        if not was_ready:
            return
        rec = {"team": int(team), "slot": int(slot), "tick": int(self.game.get_current_tick()),
               "cost": self._ability_cost(team, slot)}
        if team == self.observer:
            # The viewer's own bar is visible to it: used to judge its own press.
            rec["elixir_before"] = float(self.game.get_elixir_for_team(team))
            rec["spent_before"] = float(self.game.get_elixir_spent(team))
        self._pending.append(rec)

    # ----------------------------------------------------------------- poll
    def poll(self) -> None:
        """After a step: emit new opponent plays in tick order, then the
        pending ability presses (a press at tick t comes after every play at
        t: the engine plays cards first), then refresh the own next card."""
        g = self.game
        new: List[Tuple[int, int]] = []
        for c in range(self._n):
            t = g.get_last_played_tick(self.opponent, c)
            if t != self._last[c]:
                self._last[c] = t
                if t >= 0:
                    new.append((int(t), c))
        new.sort()
        # A press noted at the current tick has not executed yet (the env notes
        # it before the step): an observer that polls mid-step, such as a
        # neural opponent deciding before team 0's press runs, must not see it.
        now = int(g.get_current_tick())
        pending = sorted((p for p in self._pending if p["tick"] < now),
                         key=lambda p: p["tick"])
        self._pending = [p for p in self._pending if p["tick"] >= now]
        opp_press = [p for p in pending if p["team"] == self.opponent]
        own_press = [p for p in pending if p["team"] == self.observer]

        pi = 0
        play_ticks = {t for t, _ in new}
        for t, c in new:
            while pi < len(opp_press) and opp_press[pi]["tick"] < t:
                self._emit_opp_press(opp_press[pi], opp_press[pi]["tick"] in play_ticks); pi += 1
            self.tracker.on_play(t, c)
            self.stats["plays"] += 1
        while pi < len(opp_press):
            self._emit_opp_press(opp_press[pi], opp_press[pi]["tick"] in play_ticks); pi += 1

        for p in own_press:
            self._emit_own_press(p)
        if self.track_own_next:
            self._refresh_own_next()

    def _emit_opp_press(self, p: dict, same_tick_play: bool) -> None:
        tr = self.tracker
        tr.advance(p["tick"])
        # The press was queued only if the engine called the ability ready, and
        # readiness includes the elixir cost. With no card played at that tick
        # the engine therefore accepted it, whatever the estimate says: it is
        # counted, and a low estimate is floor-corrected (corrections() shows
        # it). Only a same-tick card can make the engine refuse (it plays the
        # card first, then refuses an ability it can no longer pay for:
        # GameManager::activateChampionAbility), and there the estimate after
        # the plays is the discriminator.
        if not same_tick_play or tr.elixir_estimate() >= p["cost"] - 1e-6:
            tr.on_ability(p["tick"], p["cost"])
            self.stats["abilities_counted"] += 1
        else:
            self.stats["abilities_refused"] += 1

    def _emit_own_press(self, p: dict) -> None:
        g = self.game
        # Own spend this step on cards (and abilities, if the counter includes
        # them) from the viewer's own counter.
        card_spent = float(g.get_elixir_spent(self.observer)) - p["spent_before"]
        if p["elixir_before"] - card_spent >= p["cost"] - 1e-4:
            self.tracker.on_own_ability(p["cost"])
            self.stats["own_abilities_counted"] += 1
        else:
            self.stats["own_abilities_refused"] += 1

    def _refresh_own_next(self) -> None:
        key = tuple(int(c) for c in self.game.get_hand_for_team(self.observer))
        if key == self._own_next_key:
            return
        self._own_next_key = key
        self._own_next = probe_next_card(self.game, self.observer)
        self.stats["probes"] += 1
        self.tracker.set_own_next(self._own_next)
