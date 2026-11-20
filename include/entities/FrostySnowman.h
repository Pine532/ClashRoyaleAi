#pragma once
#include "MeleeTroop.h"
#include "CardStats.h"
#include "Board.h"
#include <memory>
#include <vector>

// Hero Ice Wizard's Snowman (Frosty Fella; HeroIceWizardFrostyFellaEffect
// spawns it and records what it froze). A stationary, non-attacking troop:
// - its hp drains linearly to 0 over lifetimeTicks, written to hp directly (as
//   Building's expiry is), so no curse, shield or parry changes the drain;
//   damage from enemies comes on top, so it can die sooner;
// - while it lives, and for at most freezeCapTicks, it holds a full freeze on
//   every unit it froze at spawn, wherever that unit has since been moved;
// - once every unit it froze has died, it destroys itself selfDestructTicks
//   later.
// It is an ordinary troop otherwise: damageable, pushed by knockback and
// Tornado, picked up by Evo Giant Snowball. It is transform-immune (set from
// its CardStats), so Mother Witch's curse never turns it into a hog.
class FrostySnowman : public MeleeTroop {
public:
    // The hold is re-applied every tick for two ticks, so there is no gap
    // whichever of the Snowman and a frozen unit updates first within a tick;
    // the freeze ends at most one tick after the Snowman does.
    static constexpr int HOLD_TICKS = 2;

    int lifetimeTicks;
    int freezeCapTicks;
    int selfDestructTicks;
    std::vector<int> frozenIds;
    int ticksAlive = 0;
    int ticksSinceAllFrozenDied = 0;

    FrostySnowman(int id, float x, float y, int team, const CardStats& stats,
                  int lifetimeTicks, int freezeCapTicks, int selfDestructTicks)
        : MeleeTroop(id, x, y, stats.hp, team, stats.speed, stats.attackRange, stats.damage,
                     stats.attackCooldown, stats.symbol),
          lifetimeTicks(lifetimeTicks), freezeCapTicks(freezeCapTicks),
          selfDestructTicks(selfDestructTicks), spawnHp(stats.hp) {}

    // For Board::deepCopy: everything, including the frozen set (ids survive
    // the copy).
    std::shared_ptr<Entity> snapshot() const override {
        return std::make_shared<FrostySnowman>(*this);
    }

    void update(Board& board) override {
        MeleeTroop::update(board);
        if (!isAlive()) return;
        ++ticksAlive;

        // Linear decay: spawnHp over lifetimeTicks, the fraction carried over.
        decayCarry += static_cast<float>(spawnHp) / static_cast<float>(lifetimeTicks);
        const int drain = static_cast<int>(decayCarry);
        if (drain > 0) {
            hp -= drain;
            decayCarry -= static_cast<float>(drain);
            board.statsEvents.notifyAttributionCleared({ id });
        }
        if (ticksAlive >= lifetimeTicks) hp = 0;

        // The freeze, held while it lives, up to the cap.
        bool anyFrozenAlive = false;
        for (int frozenId : frozenIds) {
            auto unit = std::dynamic_pointer_cast<CombatEntity>(board.findEntity(frozenId));
            if (!unit || !unit->isAlive()) continue;
            anyFrozenAlive = true;
            if (isAlive() && ticksAlive < freezeCapTicks) unit->applyFreeze(HOLD_TICKS, 0.0f);
        }

        // Nothing left to hold: gone selfDestructTicks after the last one died.
        if (anyFrozenAlive) {
            ticksSinceAllFrozenDied = 0;
        } else if (++ticksSinceAllFrozenDied >= selfDestructTicks) {
            hp = 0;
        }
    }

private:
    int spawnHp;
    float decayCarry = 0.0f;
};
