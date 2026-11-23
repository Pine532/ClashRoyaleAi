"""The tracker tail an env appends to one team's observation.

One `TrackerTail` per (env, observer team). Mode "on" owns an OpponentTracker and
an EngineFeed; mode "zero" owns nothing and returns zeros (the A/B control: same
width, no information, no engine calls). Mode "off" is never constructed: the
envs hold None, so the off path has no feed objects and no extra calls.

Call order per env, matching EngineFeed's::

    tail.reset(decks)                   # right after game.reset()
    tail.note_flags(team, {1: b, 2: b}) # BEFORE the step, for each team's flags
    ... step ...
    obs = tail.append(obs)              # polls the feed, appends the features
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from python_ai.tracking import config as tracker_config


class TrackerTail:
    def __init__(self, game, observer_team: int, mode: str,
                 decks: Optional[Sequence[Sequence[int]]] = None,
                 ability_costs: Optional[Dict] = None):
        if mode not in ("on", "zero"):
            raise ValueError(f"TrackerTail needs mode 'on' or 'zero', got {mode!r}")
        self.game = game
        self.observer = int(observer_team)
        self.mode = mode
        self.width = tracker_config.tail_width(mode)
        self.tracker = None
        self.feed = None
        if mode == "on":
            from python_ai.tracking.engine_feed import EngineFeed, make_engine_tracker
            self.tracker = make_engine_tracker()
            self.feed = EngineFeed(game, self.observer, self.tracker, decks=decks,
                                   ability_costs=ability_costs)
        self._zeros = np.zeros(self.width, dtype=np.float32)

    def reset(self, decks: Optional[Sequence[Sequence[int]]] = None) -> None:
        """Right after game.reset(): the match is at tick 0 with no plays."""
        if self.feed is not None:
            if decks is not None:
                self.feed.decks = decks
            self.feed.reset()

    def note_flags(self, team: int, flags: Dict[int, bool]) -> None:
        """`team` asks for these ability slots on the coming step. Readiness is
        read here, before the step, as the env's own decision used it."""
        if self.feed is None:
            return
        for slot, wanted in flags.items():
            if wanted:
                self.feed.note_activation(
                    team, slot, bool(self.game.is_champion_ability_ready(team, slot)))

    def features(self) -> np.ndarray:
        """Poll the feed and return the tail (a fresh array)."""
        if self.feed is None:
            return self._zeros.copy()
        self.feed.poll()
        return self.tracker.features(int(self.game.get_current_tick()))

    def append(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, self.features()])
