#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Board.h"

// Hero Berserker's "Savage Survival". For `durationTicks` the Hero swings every
// `hitCooldownTicks`, her hp cannot fall below 1, and the damage she deals to a
// Crown Tower is scaled by `towerDamageMultiplier`. Once per deploy (usesLimit 1).
// Not modelled: the "Ultra Fast" movement boost, for which no numeric speed is
// published, and the elixir refund when she is defeated before activating.
class HeroBerserkerSavageSurvivalEffect : public IAbilityEffect {
    int durationTicks;
    int hitCooldownTicks;
    float towerDamageMultiplier;

public:
    HeroBerserkerSavageSurvivalEffect(int durationTicks, int hitCooldownTicks, float towerDamageMultiplier)
        : durationTicks(durationTicks), hitCooldownTicks(hitCooldownTicks),
          towerDamageMultiplier(towerDamageMultiplier) {}

    void apply(Board&, CombatEntity& self) const override {
        self.abilityHpFloorTicksRemaining = durationTicks;
        self.abilityHitCooldownTicks = hitCooldownTicks;
        self.abilityHitCooldownTicksRemaining = durationTicks;
        self.towerDamageDealtMultiplier = towerDamageMultiplier;
        self.towerDamageDealtTicksRemaining = durationTicks;
    }
};
