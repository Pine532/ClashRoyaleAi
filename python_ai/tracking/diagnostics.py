"""Per-episode accuracy of the opponent tracker, for TensorBoard.

The envs put `tracker_*` entries in `info` (TrackerTail.diagnostics) only in mode
"on"; the trainer feeds them here each step and logs one set of `Tracker/*`
scalars when an episode finishes. Absent keys (mode off or zero) make every call
a no-op, so those modes log nothing new. Logging only: the elixir error is read
against engine truth and never reaches an observation.
"""
from __future__ import annotations

import numpy as np

KEYS = ("tracker_corrections", "tracker_next_known", "tracker_hand_known",
        "tracker_elixir_abs_err")


class TrackerDiagnostics:
    def __init__(self, num_envs: int):
        self.n = num_envs
        self.reset_all()

    def reset_all(self):
        self.steps = np.zeros(self.n)
        self.next_known = np.zeros(self.n)
        self.hand_known = np.zeros(self.n)
        self.either = np.zeros(self.n)
        self.err_sum = np.zeros(self.n)
        self.err_max = np.zeros(self.n)
        self.corrections = np.zeros(self.n)

    def reset_env(self, i):
        for a in (self.steps, self.next_known, self.hand_known, self.either,
                  self.err_sum, self.err_max, self.corrections):
            a[i] = 0.0

    def step(self, infos, valid):
        """Fold one vector step in. `valid` is 1.0 for a real transition and 0.0
        for the autoreset phantom step."""
        if "tracker_elixir_abs_err" not in infos:
            return
        v = np.asarray(valid, dtype=np.float64)
        nk = np.asarray(infos["tracker_next_known"], dtype=np.float64)
        hk = np.asarray(infos["tracker_hand_known"], dtype=np.float64)
        err = np.asarray(infos["tracker_elixir_abs_err"], dtype=np.float64)
        self.steps += v
        self.next_known += v * nk
        self.hand_known += v * hk
        self.either += v * np.maximum(nk, hk)
        self.err_sum += v * err
        self.err_max = np.maximum(self.err_max, v * err)
        # The last value of a monotone count.
        self.corrections = np.where(
            v > 0, np.asarray(infos["tracker_corrections"], dtype=np.float64),
            self.corrections)

    def finish_episode(self, i):
        """Env i's episode summary (a dict of Tracker/* tags), or None when it
        saw no tracker data. Resets env i."""
        n = self.steps[i]
        if n <= 0:
            return None
        out = {
            "Tracker/Corrections": float(self.corrections[i]),
            "Tracker/NextKnownShare": float(self.next_known[i] / n),
            "Tracker/HandKnownShare": float(self.hand_known[i] / n),
            "Tracker/HandOrNextKnownShare": float(self.either[i] / n),
            "Tracker/ElixirAbsErrMean": float(self.err_sum[i] / n),
            "Tracker/ElixirAbsErrMax": float(self.err_max[i]),
        }
        self.reset_env(i)
        return out
