#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "Building.h"
#include "AreaSpell.h"
#include "FreezeOnHit.h"
#include "Board.h"

// Hero Ice Wizard's "Frosty Fella". Official: a snowman appears behind the
// nearest slowed troop while the troops in the area are frozen in place
// (Supercell 2026-09-07). Modelled: a one-shot area centred on the nearest
// slowed enemy troop (no invented "behind" offset), dealing `damage` to every
// enemy in `radius` and freezing it (factor 0) for `freezeTicks`. The Snowman
// itself is NOT spawned (its hit points are contradicted between sources), so
// the freeze is a fixed length rather than ending when the Snowman dies. Does
// nothing if no enemy troop is slowed; the engine's activation path has
// already charged the elixir and the single use by then.
class HeroIceWizardFrostyFellaEffect : public IAbilityEffect {
    float radius;
    int damage;
    int freezeTicks;

public:
    HeroIceWizardFrostyFellaEffect(float radius, int damage, int freezeTicks)
        : radius(radius), damage(damage), freezeTicks(freezeTicks) {}

    // Nearest enemy troop (neither a Tower nor a Building) with an active slow
    // (freezeTicks > 0 and freezeSlow < 1; a fully frozen troop counts),
    // measured centre to centre from `self`. Ties go to the lowest entity id.
    // Cast to CombatEntity, not Troop, so any combatant is eligible.
    std::shared_ptr<CombatEntity> findTarget(const Board& board, const CombatEntity& self) const {
        std::shared_ptr<CombatEntity> best;
        float bestD2 = 0.0f;
        for (const auto& e : board.getEntities()) {
            if (!e->isAlive() || e->team == self.team) continue;
            auto troop = std::dynamic_pointer_cast<CombatEntity>(e);
            if (!troop || troop->isTower() || std::dynamic_pointer_cast<Building>(e)) continue;
            if (troop->freezeTicks <= 0 || troop->freezeSlow >= 1.0f) continue;
            float dx = troop->position.x - self.position.x;
            float dy = troop->position.y - self.position.y;
            float d2 = dx * dx + dy * dy;
            if (!best || d2 < bestD2 || (d2 == bestD2 && troop->id < best->id)) {
                best = troop;
                bestD2 = d2;
            }
        }
        return best;
    }

    void apply(Board& board, CombatEntity& self) const override {
        auto target = findTarget(board, self);
        if (!target) return;
        auto blast = std::make_shared<AreaSpell>(
            board.allocateId(), target->position.x, target->position.y, self.team,
            radius, damage, /*delayTicks=*/0, /*symbol=*/'*',
            std::make_shared<FreezeOnHit>(freezeTicks, 0.0f));
        board.addEntity(blast);
    }
};
