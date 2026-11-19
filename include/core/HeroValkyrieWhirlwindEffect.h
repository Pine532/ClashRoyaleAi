#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Board.h"
#include <memory>

// Hero Valkyrie's "Wild Whirlwind". On activation she (1) closes to melee
// adjacency with the nearest enemy non-tower GROUND unit within dashRange
// (towers are excluded: official 2026-09-01 fix "no longer goes directly to
// towers"); with none in range she spins where she stands, and (2) starts a
// spin of spinTicks in which CombatEntity::update fires, every 2.5 ticks
// (hitIntervalHalfTicks = 5 half-ticks), an area hit of `damage` on every enemy
// ground entity within `radius`, INCLUDING invisible ones (official 2026-09-03
// fix). Crown Towers take towerDamageMultiplier of it. While spinning she takes
// selfDamageTakenMultiplier of incoming damage (the existing curse multiplier
// applied to herself) and makes no normal attacks. The once-per-deployment
// limit comes from the registration (usesLimit 1).
// Numbers: single source, https://timesaver.gg/blog/clash-royale-hero-valkyrie
// (wiki-derived); not contradicted. The dash duration is unpublished, so the
// dash is instant and the spin window starts at activation.
class HeroValkyrieWhirlwindEffect : public IAbilityEffect {
    float dashRange;
    int spinTicks;
    float radius;
    int hitIntervalHalfTicks;
    int damage;
    float selfDamageTakenMultiplier;
    float towerDamageMultiplier;

public:
    HeroValkyrieWhirlwindEffect(float dashRange, int spinTicks, float radius, int hitIntervalHalfTicks,
            int damage, float selfDamageTakenMultiplier, float towerDamageMultiplier)
        : dashRange(dashRange), spinTicks(spinTicks), radius(radius),
          hitIntervalHalfTicks(hitIntervalHalfTicks), damage(damage),
          selfDamageTakenMultiplier(selfDamageTakenMultiplier),
          towerDamageMultiplier(towerDamageMultiplier) {}

    void apply(Board& board, CombatEntity& self) const override {
        std::shared_ptr<Entity> nearest;
        float best = dashRange;
        for (const auto& e : board.getEntities()) {
            if (e->team == self.team || !e->isAlive() || !e->isTargetable()) continue;
            if (e->isFlying || e->isTower()) continue;
            float d = self.position.distanceTo(e->position);
            if (d <= best) { best = d; nearest = e; }
        }
        if (nearest) {
            // Melee adjacency (1.0 tile), as GoldenKnightDashEffect; never overshoots.
            pullToward(self, nearest->position, self.position.distanceTo(nearest->position) - 1.0f);
        }
        self.whirlTicksRemaining = spinTicks;
        self.whirlElapsedTicks = 0;
        self.whirlHitsDone = 0;
        self.whirlHitIntervalHalfTicks = hitIntervalHalfTicks;
        self.whirlRadius = radius;
        self.whirlDamage = damage;
        self.whirlTowerDamage = static_cast<int>(damage * towerDamageMultiplier); // 97 * 0.5 = 48 (truncation, engine convention)
        self.applyCurse(selfDamageTakenMultiplier, spinTicks);
    }
};
