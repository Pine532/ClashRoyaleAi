# Draft issue (not posted): buildings never fire their deploy bursts

**Title:** Buildings never fire their deploy burst (`spawnDefensiveBuilding` skips `spawnDeployEffect`)

Hi! While syncing card values with the live game I found that a building's one-time burst on
landing never happens.

**What happens.** `CardFactories::spawnDefensiveBuilding` (`include/core/CardFactories.h`)
builds the building and returns. Every other factory (both squad archetypes and both
building-targeter archetypes) also calls `spawnDeployEffect`. So a `withSpawnEffect(...)` on a
`DefensiveBuilding` is silently dead data.

**Cards affected.** No other building sets a burst, and buildings only spawn through this one
factory, so these three are the only ones:

- **Goblin Drill** (98, and both halves of Evolution 153): the 84-damage emergence burst.
- **Cannon Evolution** (129): the Deploy Barrage.
- **Tesla Evolution** (154): the Electro Pulse and its stun.

**Reproduce.** Spawn an evolved Cannon (129) one tile from an enemy unit: the unit takes no
damage. Spawn an Electro Wizard (35) the same way: its zap lands.

**Fix (on my fork, branch `update/live-sync`).**

- `abccf30`: `spawnDefensiveBuilding` calls `spawnDeployEffect`, as the other factories do.
  `tests/core/test_building_deploy_effects.cpp` (`[deploy_effect]`): 4 of its 6 cases fail
  without the fix.
- `b069d65`, a companion change: a burst can carry its own Crown Tower damage
  (`withSpawnEffectCrownTowerDamage`, passed through as `AreaSpell::crownTowerDamage`).
  Otherwise the newly firing Goblin Drill would hit towers for 84; the official value is 20
  (2026-10-06: "Crown Tower Spawn Damage: 0 → 20").

The fix is gameplay-affecting for those three cards only. Checkpoints load.

Happy to open a PR with just these two commits if that's useful.
