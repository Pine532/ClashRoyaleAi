"""Perception events -> the shared opponent tracker.

The training env and the live loop must give the policy the SAME tracker
features, so the logic lives once, in `python_ai.tracking.opponent_tracker`
(pure Python/numpy, event-fed). This module is a thin adapter: it owns one
`OpponentTracker` with a rules preset and maps what vision produces onto the
tracker's calls. It holds no rule of its own.

Presets
-------
  "engine"  `engine_feed.rules_from_engine()`: reproduces the simulator, for the
            conformance test (one recorded event stream gives identical features).
  "real"    `opponent_tracker.real_game_rules(...)`: the real game's economy
            (2.8 s per elixir). The default for live use. Replaying a stream
            under it changes only the elixir scalar.

Event interface (what the detectors will emit)
----------------------------------------------
  on_placement(PlacementEvent)          team 1 only; the placement detector's
                                        output (detect/placements.py, a stub today)
  on_opponent_play(tick, card_sim_id)   the same without the event wrapper
  on_opponent_ability(tick, cost)       HOOK ONLY: no detector exists; nothing in
                                        the live loop calls it yet
  on_own_play(tick, card_sim_id)        our press (mvp_loop knows it)
  on_own_ability(tick, cost)            our own ability press
  set_own_next(card | None) / set_own_next_from_cycle(CycleState)
                                        the "next" slot reading
  on_collector_destroyed(tick)          a visible opponent Elixir Collector died

`tick` is the match clock in ENGINE ticks (timebase.py converts seconds).

Not duplicated
--------------
track/cycle.py, opp_deck.py and opp_elixir.py are left alone: they answer other
questions (own-hand FIFO with permutation recovery; deck-completion and
missed-placement alarms; a clamped balance with a negative-balance alarm) and
they emit no policy features. Their rules overlap the core's, but folding them
in would change what the alarms report; the tracker feeds the policy, they
diagnose the detector.

Degradation
-----------
An opponent play whose card has no simulator id cannot be debited or cycled.
It is counted (`unmapped_plays`) and sets `degraded`; it is never fed as a free
card. Callers should log `degraded`: from the first such play the cycle and
the elixir are unsound.
"""
from __future__ import annotations

from typing import Optional

from contracts import UNKNOWN_CARD_SIM_ID, CycleState, PlacementEvent

ENGINE = "engine"
REAL = "real"
PRESETS = (ENGINE, REAL)


class TrackerAdapter:
    def __init__(self, preset: str = REAL):
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {PRESETS}, got {preset!r}")
        import clash_royale_env as E  # noqa: PLC0415

        from python_ai.tracking import engine_feed  # noqa: PLC0415
        from python_ai.tracking import opponent_tracker as ot  # noqa: PLC0415
        CE = E.ClashRoyaleEnv
        if preset == ENGINE:
            rules = engine_feed.rules_from_engine()
        else:
            rules = ot.real_game_rules(int(CE.DOUBLE_ELIXIR_TICK),
                                       int(CE.TRIPLE_ELIXIR_TICK))
        self.preset = preset
        self.tracker = ot.OpponentTracker(
            rules, engine_feed.card_facts_from_engine(), int(CE.NUM_CARD_IDS),
            max_match_elixir=float(CE.MAX_MATCH_ELIXIR))
        self.width = ot.feature_width(int(CE.NUM_CARD_IDS))
        self.reset()

    def reset(self) -> None:
        """A new match."""
        self.tracker.reset()
        self.unmapped_plays = 0
        self.own_plays = 0

    @property
    def degraded(self) -> bool:
        return self.unmapped_plays > 0

    # ------------------------------------------------------------ opponent
    def on_placement(self, event: PlacementEvent) -> None:
        if event.team != 1:
            return
        self.on_opponent_play(event.tick, event.card_sim_id)

    def on_opponent_play(self, tick: int, card_sim_id: int) -> None:
        if card_sim_id == UNKNOWN_CARD_SIM_ID:
            self.unmapped_plays += 1
            return
        self.tracker.on_play(int(tick), int(card_sim_id))

    def on_opponent_ability(self, tick: int, cost: float) -> None:
        """Hook: the opponent paid `cost` for a Hero/Champion ability."""
        self.tracker.on_ability(int(tick), float(cost))

    def on_collector_destroyed(self, tick: int) -> None:
        self.tracker.on_collector_destroyed(int(tick))

    # ----------------------------------------------------------------- own
    def on_own_play(self, tick: int, card_sim_id: int) -> None:
        """Our press. The cycle moved, so the previous "next" reading is stale
        until the next one arrives (never carried over)."""
        self.own_plays += 1
        self.tracker.set_own_next(None)

    def on_own_ability(self, tick: int, cost: float) -> None:
        """Our ability press; spend is cumulative, so the tick is unused."""
        self.tracker.on_own_ability(float(cost))

    def set_own_next(self, card_sim_id: Optional[int]) -> None:
        if card_sim_id is not None and card_sim_id == UNKNOWN_CARD_SIM_ID:
            card_sim_id = None
        self.tracker.set_own_next(card_sim_id)

    def set_own_next_from_cycle(self, cycle: CycleState) -> None:
        self.set_own_next(cycle.next_card)

    # ------------------------------------------------------------ features
    def features(self, tick: int):
        """The tail vector at `tick` (engine ticks)."""
        return self.tracker.features(int(tick))
