"""CLASH_OPP_TRACKER: the one place the opponent-tracker setting is read.

    off   (default, unset)  exactly the engine's observation; nets built with
                            tracker_size=0; no feed objects; no extra engine
                            calls on the step path.
    on                      the tracker tail is appended to every observation the
                            trainee sees; nets are built with tracker_size = the
                            tail width.
    zero                    the same layout and net as "on", the tail all zeros:
                            the A/B control (same code, no information).

It is a training setting, not an operational one: resuming under a different
value changes what the policy is shown, so `checkpointing.settings_drift`
reports it as a CHANGED setting (it is recorded with every other CLASH_*
variable and is deliberately not in OPERATIONAL_SETTINGS).

Read at call time (env and net construction), not at import, so a test can
set it without a fresh interpreter; a vector-env worker inherits the
environment of the process that built it.
"""
from __future__ import annotations

import os
from typing import Mapping, Optional

ENV_VAR = "CLASH_OPP_TRACKER"
MODES = ("off", "on", "zero")


def mode(environ: Optional[Mapping[str, str]] = None) -> str:
    """"off", "on" or "zero". An unknown value raises (never a silent off)."""
    env = os.environ if environ is None else environ
    raw = env.get(ENV_VAR)
    if raw is None or raw.strip() == "":
        return "off"
    value = raw.strip().lower()
    if value not in MODES:
        raise ValueError(f"{ENV_VAR}={raw!r}: expected one of {MODES}")
    return value


def tail_width(m: Optional[str] = None) -> int:
    """Floats appended to the engine observation: 0 when off."""
    m = mode() if m is None else m
    if m == "off":
        return 0
    import clash_royale_env as E
    from python_ai.tracking.opponent_tracker import feature_width
    return feature_width(int(E.ClashRoyaleEnv.NUM_CARD_IDS))


def net_kwargs(m: Optional[str] = None) -> dict:
    """Keyword arguments every training-time MicroRoyaleNet is built with."""
    return {"tracker_size": tail_width(m)}
