#pragma once
#include "CardStats.h"

// Spirit Empress's ground and flying forms; GameManager::playCard picks one by
// the elixir held at play time (the real switching rule is unsourced). Both
// share id 165 and the same placement footprint, so only cost and stats differ.
//
// 2026-05-04 balance update: both forms 1121 hp; 2026-09-08 balance update:
// both forms 320 damage (docs/LIVE_SYNC.md). Ground: melee 1.2, 1.2 s,
// fast, ground-only, 3 elixir. Flying: range 5, 1.4 s, medium, air and ground,
// 6 elixir.
inline CardStats spiritEmpressGroundStats() {
    CardStats stats;
    stats.id = 165;
    stats.name = "Spirit Empress";
    stats.cost = 3.0f;
    stats.archetype = Archetype::MeleeSquad;
    stats.hp = 1121; // 2026-05-04 balance update (supercell.com May balance changes 2026)
    // 2026-03-02 balance update: ground form Medium -> Fast (docs/LIVE_SYNC.md),
    // replacing the earlier off-tier 0.85 placeholder. Scaled here because
    // direct assignment bypasses CardStats::troop().
    stats.speed = SPEED_FAST * MOVEMENT_SPEED_SCALE;
    stats.attackRange = 1.2f;
    stats.damage = 320; // 2026-09-08 balance update (docs/LIVE_SYNC.md)
    stats.attackCooldown = 12;
    stats.symbol = '<';
    return stats;
}

inline CardStats spiritEmpressFlyingStats() {
    CardStats stats;
    stats.id = 165;
    stats.name = "Spirit Empress";
    stats.cost = 6.0f;
    stats.archetype = Archetype::RangedSquad;
    stats.hp = 1121; // 2026-05-04 balance update (supercell.com May balance changes 2026)
    // Scaled for the same reason as the ground form.
    stats.speed = 0.5f * MOVEMENT_SPEED_SCALE;
    stats.attackRange = 5.0f;
    stats.damage = 320; // 2026-09-08 balance update (docs/LIVE_SYNC.md)
    stats.attackCooldown = 14;
    stats.symbol = '<';
    stats.isFlying = true;
    stats.targetsAir = true;
    return stats;
}
