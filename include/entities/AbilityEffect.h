#pragma once
#include <memory>
#include "Entity.h"

class CombatEntity;

// Behaviour fired by a Champion's activated ability
// (GameManager::activateChampionAbility). Concrete effects live in core/. Takes
// the acting entity, since an ability mutates its own state (e.g. Mighty
// Miner's teleport). Effects are shared by every instance of the card, so any
// per-unit state lives on the entity, not here.
class IAbilityEffect {
public:
    virtual ~IAbilityEffect() = default;
    virtual void apply(Board& board, CombatEntity& self) const = 0;

    // Optional: an ability that reacts to its owner's own hits (Hero Ice
    // Wizard's Frosty Fella, which waits for his slow to land). When true, the
    // owner's projectiles call onOwnerHitLanded after their on-hit effects ran,
    // provided the owner is still alive. Default: not interested, so no other
    // projectile pays for the owner lookup.
    virtual bool wantsOwnerHits() const { return false; }
    virtual void onOwnerHitLanded(Board& /*board*/, CombatEntity& /*owner*/,
                                  const std::shared_ptr<CombatEntity>& /*target*/) const {}
};
