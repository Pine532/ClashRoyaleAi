"""Hand-slot readiness: the engine's post-cycle lock, which the observation lacks.

`PlayerState::playCard` refuses a hand slot for 20 ticks after a card cycles into
it, exactly as it refuses an unaffordable card. The observation carries elixir and
costs but not this lock, so `MicroRoyaleNet.affordability_mask` alone called those
slots legal. Measured 2026-11-18 on a rung-0 policy, 10-16% of card attempts were
silently refused, on main as on the live-sync branch, and their advantages were
noise.

The engine reports the lock (`ClashRoyaleEnv.get_hand_cooldown_ticks`). An action
is applied before the step's first tick, so a slot reading 0 at a decision is one
`playCard` accepts. The env puts per-slot readiness in its info, the rollout masks
the card head with it, and the buffer stores it so the update applies the same
mask. Readiness is read every decision, so a slot stays masked for as long as the
engine holds it, whatever `skip_frames` is.
"""
import numpy as np
import torch

#: Per-slot info keys, one bool each (gymnasium's vector info merges scalars).
INFO_KEY = "hand_slot{}_ready"


def info_entries(game, team, hand_size):
    """{info key: bool} for one env, read from the engine."""
    ticks = game.get_hand_cooldown_ticks(team)
    return {INFO_KEY.format(i): bool(ticks[i] == 0) for i in range(hand_size)}


def ready_from_engine(game, team, hand_size):
    """(1, hand_size) bool, for a policy that acts inside the env (phase 2's
    opponent)."""
    ticks = game.get_hand_cooldown_ticks(team)
    return torch.tensor([[ticks[i] == 0 for i in range(hand_size)]], dtype=torch.bool)


def ready_from_infos(infos, num_envs, hand_size):
    """(num_envs, hand_size) bool readiness for the next decision.

    A missing key reads READY. The only step without it is the phantom
    post-autoreset step, which returns a fresh match's observation, and the engine
    zeroes the lock when it resets a hand.
    """
    ready = np.ones((num_envs, hand_size), dtype=bool)
    for i in range(hand_size):
        key = INFO_KEY.format(i)
        v = infos.get(key)
        if v is None:
            continue
        present = np.asarray(infos.get("_" + key, np.ones(num_envs, dtype=bool)),
                             dtype=bool).reshape(num_envs)
        vals = np.asarray(v, dtype=bool).reshape(num_envs)
        ready[:, i] = np.where(present, vals, True)
    return torch.as_tensor(ready)
