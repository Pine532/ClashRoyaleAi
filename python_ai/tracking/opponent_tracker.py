"""An event-fed tracker of the opponent's hidden state, from visible events only.

Inputs are the things a viewer sees: a card was played at a tick, an ability was
paid for, elixir was granted. From them it keeps

  * the card cycle: which cards the opponent has shown, which of those are in
    hand, and which card is next;
  * an elixir estimate, advanced event by event;
  * the viewer's own next card and both sides' ability spend.

It reads no engine state and imports no engine module, so perception can run the
same code on vision events. The simulator feed is `engine_feed.py`.

Event timing
------------
Ticks follow the engine's clock: the regeneration of tick t is applied (and the
tick's Collector grant credited) when the clock reaches t; a card played "at
tick t" is played after that, before tick t + 1 regenerates
(ClashEnv::runSelfPlayTicks plays at the first of a step's ticks, then steps).
`advance(t)` therefore regenerates through t, and an event at t is debited
after it. A play must be debited at its OWN tick, not at the decision boundary:
with the cap in play the order matters (a 0.42 error was measured at the
boundary, exact at the play's tick).

The cycle (PlayerState.h:146-164)
---------------------------------
The deck is 2 * hand_size cards; the queue holds hand_size. A played card goes
to the back of the queue and the front card fills its slot. After plays
p1..pn with n >= hand_size the queue is exactly the last hand_size plays, oldest
first, so the next card is p(n - hand_size + 1) and the hand is the deck minus
those. Before all cards are seen, the hand is the seen cards minus the queue
plus `unknown` cards. With n < hand_size the played cards sit behind the
original unseen queue: they are out of hand and the next card is unknown.

Circularity (Spirit Empress)
----------------------------
Spirit Empress costs 6 with at least 6 elixir and 3 otherwise
(GameManager::playCard). The tracker decides which from its own estimate, which
is circular: a wrong estimate can pick the wrong cost and the error persists.
The live version should read the card's form (ground = 3, flying = 6) instead.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

NORMAL = "normal"
MIRROR = "mirror"
SPIRIT_EMPRESS = "spirit_empress"
ELIXIR_COLLECTOR = "elixir_collector"
KINDS = (NORMAL, MIRROR, SPIRIT_EMPRESS, ELIXIR_COLLECTOR)

#: Scalars after the three per-card-id blocks, in order (the index into the
#: scalar block).
SCALAR_NAMES = ("opp_unknown_in_hand", "opp_next_known", "opp_distinct_seen",
                "opp_elixir", "corrections", "own_next_known",
                "opp_ability_spend", "own_ability_spend")
SCALAR_INDEX = {n: i for i, n in enumerate(SCALAR_NAMES)}
NUM_SCALARS = len(SCALAR_NAMES)

#: A correction count of this saturates the `corrections` feature.
CORRECTIONS_SATURATION = 10


def feature_offsets(num_card_ids: int) -> Dict[str, int]:
    """FORWARD offsets of each feature section (never locate one from the end).

    opp_in_hand[N] | opp_next[N] | own_next[N] | scalars[NUM_SCALARS]
    """
    n = int(num_card_ids)
    return {"opp_in_hand": 0, "opp_next": n, "own_next": 2 * n, "scalars": 3 * n}


def feature_width(num_card_ids: int) -> int:
    return 3 * int(num_card_ids) + NUM_SCALARS


@dataclass(frozen=True)
class TrackerRules:
    """The world's rules, as numbers. Presets: `engine_feed.rules_from_engine`
    (must reproduce the simulator exactly) and `real_game_rules` below."""
    regen_per_tick: float
    #: ((from_tick, multiplier), ...) ascending; the multiplier in force at tick
    #: t is the last entry with from_tick <= t.
    phase_schedule: Tuple[Tuple[int, float], ...]
    start_elixir: float
    cap: float
    hand_size: int
    #: A Collector played at tick T credits at T + first_delay, then every
    #: period ticks, while (pulse tick - T) <= lifetime.
    collector_first_delay: int
    collector_period: int
    collector_lifetime: int
    collector_amount: float = 1.0
    #: Mirror pays the copied card's cost plus this.
    mirror_surcharge: float = 1.0
    #: Spirit Empress pays `empress_high_cost` with at least
    #: `empress_threshold` elixir, else `empress_low_cost`.
    empress_threshold: float = 6.0
    empress_low_cost: float = 3.0
    empress_high_cost: float = 6.0

    @property
    def deck_size(self) -> int:
        return 2 * self.hand_size

    def multiplier_at(self, tick: int) -> float:
        m = self.phase_schedule[0][1]
        for start, mult in self.phase_schedule:
            if tick >= start:
                m = mult
            else:
                break
        return m

    def replace(self, **kw) -> "TrackerRules":
        return dataclasses.replace(self, **kw)


def real_game_rules(double_elixir_tick: int, triple_elixir_tick: int) -> TrackerRules:
    """The real game's economy.

    * Regeneration: one elixir per 2.8 s (perception/timebase.py,
      REAL_SECONDS_PER_ELIXIR_1X) at 10 ticks per second, i.e. 1/28 per tick.
      The engine's 0.035 is about 2% slower.
    * Phase ticks are passed in (the engine binds them; the real game's 2x at
      2:00 and 3x at 3:00 are the same instants).
    * Elixir Collector lifetime 1100 ticks: the 2026-10-06 balance update
      (docs/LIVE_SYNC.md:1223). Its first-pulse delay (160 = deploy 10 + 150)
      and period (150) are the engine's, unverified against the real game.
    * Start 5.0, cap 10.0, hand 4: as the engine.
    """
    return TrackerRules(
        regen_per_tick=1.0 / (2.8 * 10),
        phase_schedule=((0, 1.0), (int(double_elixir_tick), 2.0),
                        (int(triple_elixir_tick), 3.0)),
        start_elixir=5.0, cap=10.0, hand_size=4,
        collector_first_delay=160, collector_period=150, collector_lifetime=1100,
    )


class CardFacts:
    """Per card id: cost and kind, supplied by the caller. The core never
    hardcodes ids. An unknown id raises KeyError (no silent zero cost)."""

    def __init__(self, costs: Mapping[int, float], kinds: Optional[Mapping[int, str]] = None):
        self._costs = {int(k): float(v) for k, v in costs.items()}
        self._kinds = {int(k): v for k, v in (kinds or {}).items()}
        for k, v in self._kinds.items():
            if v not in KINDS:
                raise ValueError(f"card {k}: unknown kind {v!r}")
            if k not in self._costs:
                raise KeyError(f"card {k} has a kind but no cost")

    def cost(self, card_id: int) -> float:
        return self._costs[int(card_id)]

    def kind(self, card_id: int) -> str:
        cid = int(card_id)
        if cid not in self._costs:
            raise KeyError(cid)
        return self._kinds.get(cid, NORMAL)


class OpponentTracker:
    """See the module docstring. All state is plain Python/numpy and cheap to
    copy (`copy.deepcopy`), for search forks."""

    def __init__(self, rules: TrackerRules, facts: CardFacts, num_card_ids: int,
                 max_match_elixir: float):
        if max_match_elixir <= 0:
            raise ValueError("max_match_elixir must be positive")
        self.rules = rules
        self.facts = facts
        self.num_card_ids = int(num_card_ids)
        self.max_match_elixir = float(max_match_elixir)
        self._f32 = np.float32
        self._cap = self._f32(rules.cap)
        self._regen = self._f32(rules.regen_per_tick)
        self.reset()

    # ------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        self._tick = 0
        self._elixir = self._f32(self.rules.start_elixir)
        self._plays: List[int] = []
        self._seen: List[int] = []            # distinct, first-seen order
        self._last_non_mirror: Optional[int] = None
        self._corrections = 0
        self._inconsistent = 0                # distinct seen above the deck size
        self._collectors: List[int] = []      # play ticks of live Collectors
        self._opp_ability = 0.0
        self._own_ability = 0.0
        self._own_next: Optional[int] = None

    # --------------------------------------------------------------- clock
    def _mult(self, tick: int) -> np.float32:
        return self._f32(self.rules.multiplier_at(tick))

    def advance(self, tick: int) -> None:
        """Regenerate (and credit Collector pulses) through `tick`."""
        tick = int(tick)
        if tick < self._tick:
            raise ValueError(f"time ran backwards: {tick} < {self._tick}")
        r = self.rules
        e = self._elixir
        for t in range(self._tick + 1, tick + 1):
            e = min(e + self._regen * self._mult(t), self._cap)
            if self._collectors:
                grant = 0
                for pt in self._collectors:
                    d = t - pt - r.collector_first_delay
                    if d >= 0 and d % r.collector_period == 0 and (t - pt) <= r.collector_lifetime:
                        grant += 1
                if grant:
                    e = min(e + self._f32(r.collector_amount * grant), self._cap)
                self._collectors = [pt for pt in self._collectors
                                    if (t - pt) < r.collector_lifetime]
        self._elixir = self._f32(e)
        self._tick = tick

    # -------------------------------------------------------------- events
    def _debit(self, cost: float) -> None:
        cost32 = self._f32(cost)
        if self._elixir < cost32:
            # A play of cost c proves the opponent had at least c.
            self._elixir = cost32
            self._corrections += 1
        self._elixir = self._f32(self._elixir - cost32)

    def on_play(self, tick: int, card_id: int) -> None:
        card_id = int(card_id)
        kind = self.facts.kind(card_id)           # KeyError on an unknown id
        self.advance(tick)
        r = self.rules
        if kind == MIRROR:
            if self._last_non_mirror is None:
                raise ValueError("Mirror played with no earlier play to copy")
            cost = self.facts.cost(self._last_non_mirror) + r.mirror_surcharge
        elif kind == SPIRIT_EMPRESS:
            cost = (r.empress_high_cost if self._elixir >= self._f32(r.empress_threshold)
                    else r.empress_low_cost)
        else:
            cost = self.facts.cost(card_id)
        self._debit(cost)
        if kind != MIRROR:
            self._last_non_mirror = card_id
        if kind == ELIXIR_COLLECTOR:
            self._collectors.append(int(tick))
        self._plays.append(card_id)
        if card_id not in self._seen:
            self._seen.append(card_id)
            if len(self._seen) > r.deck_size:
                self._inconsistent += 1

    def on_ability(self, tick: int, cost: float) -> None:
        """The opponent paid `cost` elixir for a Hero/Champion ability."""
        self.advance(tick)
        self._debit(cost)
        self._opp_ability += float(cost)

    def on_elixir_grant(self, tick: int, amount: float) -> None:
        """Elixir credited to the opponent outside regeneration: a refund, a
        Golem-family death on the other side. Capped like every grant."""
        self.advance(tick)
        self._elixir = self._f32(min(self._elixir + self._f32(amount), self._cap))

    def on_own_ability(self, cost: float) -> None:
        self._own_ability += float(cost)

    def set_own_next(self, card_id: Optional[int]) -> None:
        """The viewer's own next card (None when unknown)."""
        self._own_next = None if card_id is None else int(card_id)

    # --------------------------------------------------------------- state
    def elixir_estimate(self) -> float:
        return float(self._elixir)

    def corrections(self) -> int:
        return self._corrections

    def inconsistencies(self) -> int:
        return self._inconsistent

    def seen_order(self) -> List[int]:
        """Every play in order (a Mirror under its own id)."""
        return list(self._plays)

    def distinct_seen(self) -> int:
        return len(self._seen)

    def _queue(self) -> List[int]:
        return self._plays[-self.rules.hand_size:] if self._plays else []

    def hand_known(self) -> Set[int]:
        """Seen cards known to be in hand: the seen set minus the queue."""
        q = set(self._queue())
        return {c for c in self._seen if c not in q}

    def unknown_in_hand(self) -> int:
        return max(0, self.rules.hand_size - len(self.hand_known()))

    def next_card(self) -> Optional[int]:
        """p(n - hand_size + 1) once hand_size plays are known, else None."""
        h = self.rules.hand_size
        if len(self._plays) < h:
            return None
        return self._plays[-h]

    def own_next(self) -> Optional[int]:
        return self._own_next

    def opp_ability_spend(self) -> float:
        return self._opp_ability

    def own_ability_spend(self) -> float:
        return self._own_ability

    # ------------------------------------------------------------ features
    def features(self, tick: int) -> np.ndarray:
        """The fixed-width float32 vector; layout in `feature_offsets`."""
        self.advance(tick)
        n = self.num_card_ids
        off = feature_offsets(n)
        out = np.zeros(feature_width(n), dtype=np.float32)
        for c in self.hand_known():
            out[off["opp_in_hand"] + c] = 1.0
        nxt = self.next_card()
        if nxt is not None:
            out[off["opp_next"] + nxt] = 1.0
        if self._own_next is not None:
            out[off["own_next"] + self._own_next] = 1.0
        s = off["scalars"]
        ix = SCALAR_INDEX
        r = self.rules
        out[s + ix["opp_unknown_in_hand"]] = self.unknown_in_hand() / r.hand_size
        out[s + ix["opp_next_known"]] = 0.0 if nxt is None else 1.0
        out[s + ix["opp_distinct_seen"]] = len(self._seen) / r.deck_size
        out[s + ix["opp_elixir"]] = float(self._elixir) / r.cap
        out[s + ix["corrections"]] = min(1.0, self._corrections / CORRECTIONS_SATURATION)
        out[s + ix["own_next_known"]] = 0.0 if self._own_next is None else 1.0
        out[s + ix["opp_ability_spend"]] = self._opp_ability / self.max_match_elixir
        out[s + ix["own_ability_spend"]] = self._own_ability / self.max_match_elixir
        return out
