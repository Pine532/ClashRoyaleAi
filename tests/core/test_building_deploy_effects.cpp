// A card's one-time deploy burst (CardStats::withSpawnEffect) and the Crown Tower
// damage it carries. Buildings never fired theirs before 2026-11-18
// (CardFactories::spawnDefensiveBuilding skipped spawnDeployEffect), which left
// the Goblin Drill's emergence, the Cannon Evolution's Deploy Barrage and the
// Tesla Evolution's Electro Pulse as dead data.
#include "live_sync_test_helpers.h"
#include "CardFactories.h"

using namespace live_sync;

namespace {

// The enemy (team 1) left Princess Tower and its hp.
std::shared_ptr<Entity> enemyLeftPrincess(Board& board) {
    std::shared_ptr<Entity> best;
    for (const auto& e : board.getEntities())
        if (e->isTower() && e->team == 1 && e->position.x < 9.0f && e->position.y < 30.0f)
            if (!best || e->position.y < best->position.y) best = e;
    return best;
}

} // namespace

TEST_CASE("A deploy burst's Crown Tower damage is its own, not the troop damage",
          "[deploy_effect][crown_tower]") {
    Match m;
    auto tower = enemyLeftPrincess(m.board());
    REQUIRE(tower != nullptr);
    const int before = tower->hp;

    CardStats burst;
    burst.name = "Test Burst";
    burst.archetype = Archetype::MeleeSquad;
    burst.hp = 100; burst.damage = 0; burst.attackCooldown = 10;
    burst.withSpawnEffect(2.0f, 84).withSpawnEffectCrownTowerDamage(20);
    CardFactories::spawnMeleeSquad(burst, tower->position.x, tower->position.y - 1.5f, 0, m.board());
    m.board().commitPendingEntities();
    m.step(2);

    CHECK(before - tower->hp == 20);
}
