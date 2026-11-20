#pragma once
#include "AbilityEffect.h"
#include "CombatEntity.h"
#include "AreaSpell.h"
#include "Board.h"
#include "CardStats.h"
#include "CardFactories.h"
#include "FrostySnowman.h"
#include <cmath>
#include <memory>

// Hero Ice Wizard's "Frosty Fella" (2 elixir, single use per deployment),
// built from RoyaleAPI's developer-build write-up
// (royaleapi.com/blog/hero-ice-wizard-september-2026, level 11, relayed by the
// user 2026-11-20) plus the later official changes; docs/LIVE_SYNC.md.
//
// Pressing ARMS it (the elixir and the single use are spent then). It fires on
// an enemy that this Hero's own hit has slowed: at once if one already is when
// pressed, else on the next hit of his that lands his slow. A target his hit
// kills (no slow lands) leaves it armed for the next.
//
// Refund, official 2026-10-27 (supersedes the write-up's launch-era "no refund
// while armed"): "Hero Ice Wizard now refunds Elixir if defeated before his
// projectile hits the target". Armed means the projectile that fires it has
// not hit yet, so he refunds exactly when he dies armed, whether no shot was
// in flight, a kill left it armed, or a shot was still flying (a dead Hero's
// shot fires nothing). The amount is the ability's 2 elixir: a reading of
// "refunds Elixir", which states none. GameManager::refundArmedAbilities pays
// it. Only his own slow counts (CombatEntity::slowMarkOwnerId,
// set here from his hits), so a slow from anyone else, or one Monk reflects
// back at his side, cannot trigger it.
//
// Firing: a Snowman (FrostySnowman) appears `behindTiles` behind the target, on
// the far side from the Hero. That point is the centre of a `radius` blast that
// deals `damage` (`crownTowerDamage` to Crown Towers) to enemy air and ground,
// and of a freeze: every enemy inside the area at that moment is frozen, like
// the Freeze spell, for as long as the Snowman lives, capped. Units arriving
// later are not caught; frozen units moved out stay frozen.
//
// Targets: any enemy he slows, the King Tower included. RoyaleAPI's wording ("a
// troop, a building or even a Princess Tower") reads as any target; the King
// Tower is an INFERENCE from it (user decision, 2026-11-20).
class HeroIceWizardFrostyFellaEffect : public IAbilityEffect {
    float radius;
    int damage;
    int crownTowerDamage;
    float behindTiles;
    float slowFactor;   // the Hero's own on-hit slow, to recognise it
    int slowTicks;
    CardStats snowmanStats;
    int lifetimeTicks;
    int freezeCapTicks;
    int selfDestructTicks;

public:
    HeroIceWizardFrostyFellaEffect(float radius, int damage, int crownTowerDamage, float behindTiles,
                                   float slowFactor, int slowTicks, CardStats snowmanStats,
                                   int lifetimeTicks, int freezeCapTicks, int selfDestructTicks)
        : radius(radius), damage(damage), crownTowerDamage(crownTowerDamage), behindTiles(behindTiles),
          slowFactor(slowFactor), slowTicks(slowTicks), snowmanStats(std::move(snowmanStats)),
          lifetimeTicks(lifetimeTicks), freezeCapTicks(freezeCapTicks), selfDestructTicks(selfDestructTicks) {}

    bool wantsOwnerHits() const override { return true; }
    bool refundsIfOwnerDiesArmed() const override { return true; }

    // Press: arm, and fire at once on an eligible target already under his
    // slow (his current target first, else the nearest; ties: lowest id).
    void apply(Board& board, CombatEntity& self) const override {
        self.abilityArmed = true;
        std::shared_ptr<CombatEntity> best;
        float bestD2 = 0.0f;
        for (const auto& e : board.getEntities()) {
            auto c = std::dynamic_pointer_cast<CombatEntity>(e);
            if (!c || !eligible(self, *c)) continue;
            if (c->id == self.getCurrentTargetId()) { best = c; break; }
            const float dx = c->position.x - self.position.x;
            const float dy = c->position.y - self.position.y;
            const float d2 = dx * dx + dy * dy;
            if (!best || d2 < bestD2 || (d2 == bestD2 && c->id < best->id)) {
                best = c;
                bestD2 = d2;
            }
        }
        if (best) fire(board, self, *best);
    }

    // One of his hits landed (after its on-hit slow). Mark the slow as his,
    // then fire if armed and the target qualifies.
    void onOwnerHitLanded(Board& board, CombatEntity& owner,
                          const std::shared_ptr<CombatEntity>& target) const override {
        if (target->team == owner.team) return;
        if (target->isAlive() && target->hasFreezeOfFactor(slowFactor)) {
            target->slowMarkOwnerId = owner.id;
            target->slowMarkTicks = slowTicks;
        }
        if (owner.abilityArmed && eligible(owner, *target)) fire(board, owner, *target);
    }

    bool eligible(const CombatEntity& owner, const CombatEntity& unit) const {
        return unit.isAlive() && unit.isTargetable() && unit.team != owner.team
            && unit.slowMarkOwnerId == owner.id && unit.slowMarkTicks > 0
            && unit.hasFreezeOfFactor(slowFactor);
    }

private:
    void fire(Board& board, CombatEntity& owner, const CombatEntity& target) const {
        owner.abilityArmed = false;

        // Behind the target, seen from the Hero; straight ahead if they coincide.
        float dx = target.position.x - owner.position.x;
        float dy = target.position.y - owner.position.y;
        const float d = std::sqrt(dx * dx + dy * dy);
        if (d > 1e-6f) {
            dx /= d;
            dy /= d;
        } else {
            dx = 0.0f;
            dy = (owner.team == 0) ? 1.0f : -1.0f;
        }
        const Vector2D spot = board.clampToBoard(
            Vector2D{ target.position.x + dx * behindTiles, target.position.y + dy * behindTiles }, false);

        auto snowman = std::make_shared<FrostySnowman>(board.allocateId(), spot.x, spot.y, owner.team,
                                                       snowmanStats, lifetimeTicks, freezeCapTicks,
                                                       selfDestructTicks);
        CardFactories::applyCardMetadata(snowman, snowmanStats);
        snowman->deployTicksRemaining = 0;

        // The freeze: every enemy inside the area now, by AreaSpell's surface
        // test (centre distance up to radius plus the unit's own radius).
        for (const auto& e : board.getEntities()) {
            auto c = std::dynamic_pointer_cast<CombatEntity>(e);
            if (!c || !c->isAlive() || !c->isTargetable() || c->team == owner.team) continue;
            if (spot.distanceTo(c->position) > radius + CombatEntity::effectiveRadiusOf(*c)) continue;
            c->applyFreeze(FrostySnowman::HOLD_TICKS, 0.0f);
            snowman->frozenIds.push_back(c->id);
        }
        board.addEntity(snowman);

        // The spawn damage, same centre and radius, air and ground.
        auto blast = std::make_shared<AreaSpell>(board.allocateId(), spot.x, spot.y, owner.team,
                                                 radius, damage, /*delayTicks=*/0, /*symbol=*/'*');
        blast->crownTowerDamage = crownTowerDamage;
        board.addEntity(blast);
    }
};
