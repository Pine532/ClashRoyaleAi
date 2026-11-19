#pragma once
#include <memory>
#include "PeriodicEffect.h"
#include "CombatEntity.h"
#include "Board.h"

// Elite Barbarians Evolution's spear: on each periodic tick, damages the
// nearest enemy ground target whose surface-to-surface distance lies in
// [minRange, maxRange]. Sources: official 8 Sep 2026 (range 3-4.5), 26 Aug 2026
// (damage 220); ground-only and the minimum range are from the single-source
// trophycoach guide. Surface-to-surface follows the engine's attack-range
// convention; the thrower's own radius is the implicit troop radius. No flight
// time and no Rage path (unsourced). No-op when nothing is in the band; the
// timer still restarts, so a spear is not banked.
class EliteBarbarianSpearEffect : public IPeriodicEffect {
    float minRange;
    float maxRange;
    int damage;

public:
    EliteBarbarianSpearEffect(float minRange, float maxRange, int damage)
        : minRange(minRange), maxRange(maxRange), damage(damage) {}

    void apply(Board& board, const Vector2D& position, int team) const override {
        std::shared_ptr<Entity> best;
        float bestGap = 0.0f;
        for (const auto& e : board.getEntities()) {
            if (!e->isAlive() || e->team == team || !e->isTargetable()) continue;
            if (e->isFlying) continue; // spear hits ground only
            if (!std::dynamic_pointer_cast<CombatEntity>(e)) continue; // troops, buildings, towers
            const float gap = position.distanceTo(e->position)
                            - Entity::IMPLICIT_TROOP_RADIUS
                            - CombatEntity::effectiveRadiusOf(*e);
            if (gap < minRange || gap > maxRange) continue;
            if (!best || gap < bestGap) { best = e; bestGap = gap; }
        }
        if (best) best->takeDamage(damage);
    }
};
