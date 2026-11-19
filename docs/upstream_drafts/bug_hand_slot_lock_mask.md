# Draft issue (not posted): the card mask allows hand slots the engine has locked

**Title:** Card mask ignores the 20-tick post-cycle hand lock, so 10-16% of card attempts are silently refused

Hi! While investigating a slow start in training, I found that the policy's card mask lets it
pick hand slots the engine will refuse.

**What happens.** When a card is played, `PlayerState::playCard` (`include/core/PlayerState.h`)
cycles the next card into that hand slot and locks the slot for 20 ticks
(`handCooldownTicks[slot] = 20`). During the lock it refuses the slot, the same way it refuses an
unaffordable card. The observation carries elixir and card costs but not the lock, so
`MicroRoyaleNet.affordability_mask` calls the slot legal. The policy samples it, nothing happens,
and PPO learns from an action that had no effect.

**How often.** These were measured by forking real states with `env.snapshot()` and checking
whether elixir was actually spent:
- A rung-0 policy: 15.9% of its card attempts were refused.
- A uniform-random legal policy: 10.7% on my branch and 9.8% on `main` (9ab9b45). It is not a
  regression.
- Every refused attempt was the slot played on the previous decision. With `skip_frames = 10`
  that is one decision; with closer decisions it is up to 19 decisions.

**Fix (on my fork, branch `update/live-sync`):**
- A read-only binding, `ClashRoyaleEnv.get_hand_cooldown_ticks(team)` (no gameplay change).
- The env reports per-slot readiness in its `info` every step, as it already does for Champion
  abilities.
- The rollout passes it to `affordability_mask(obs, hand_ready)`.
- The rollout buffer stores it, so the PPO update applies the same mask.
- Phase 2's frozen opponent reads team 1's readiness from the engine.

The observation layout is unchanged and checkpoints load. Tests check that the env reports the
lock for exactly as long as the engine refuses the slot, at `skip_frames` 10, 5 and 1. They also
check that the update reproduces the rollout's mask; without the stored readiness, the PPO
ratio drifts.

Happy to open a PR if that's useful.
