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

namespace {

// Spawns `cardId` (evolved form if asked) 1 tile from a stationary enemy and runs
// the board for a few ticks; returns the enemy.
std::shared_ptr<StationaryCombatant> burstOnNeighbour(int cardId, bool evolved, float distance = 1.0f) {
    auto board = std::make_shared<Board>();
    auto enemy = std::make_shared<StationaryCombatant>(900, 9.0f, 10.0f + distance, 100000, 1, 0.5f, 0, 1000);
    spawn(*board, enemy);
    spawnBare(*board, cardId, 9.0f, 10.0f, 0, evolved);
    // Run each burst to completion, as GameManager does (a spent spell is
    // removed from the board, so it never updates again).
    std::vector<std::shared_ptr<Entity>> bursts;
    for (const auto& e : board->getEntities())
        if (std::dynamic_pointer_cast<AreaSpell>(e)) bursts.push_back(e);
    for (const auto& b : bursts)
        for (int t = 0; t < 30 && b->isAlive(); ++t) b->update(*board);
    return enemy;
}

} // namespace

TEST_CASE("Goblin Drill's emergence burst hits a troop beside it for 84",
          "[deploy_effect][building]") {
    auto enemy = burstOnNeighbour(98, false);
    CHECK(100000 - enemy->hp == 84);
}

TEST_CASE("Goblin Drill's emergence burst deals 20 to a Crown Tower",
          "[deploy_effect][building][crown_tower]") {
    Match m;
    auto tower = enemyLeftPrincess(m.board());
    REQUIRE(tower != nullptr);
    const int before = tower->hp;
    m.spawnCard(98, tower->position.x, tower->position.y - 1.5f, 0);
    m.step(2);  // the first Goblin pops at 3 s, so only the burst can reach the tower
    CHECK(before - tower->hp == 20);
}

TEST_CASE("Cannon Evolution's Deploy Barrage fires when the evolved Cannon lands",
          "[deploy_effect][building]") {
    CHECK(burstOnNeighbour(129, true)->hp < 100000);
    // The un-evolved half has no barrage.
    CHECK(burstOnNeighbour(129, false)->hp == 100000);
}

TEST_CASE("Tesla Evolution's Electro Pulse damages and stuns when the evolved Tesla lands",
          "[deploy_effect][building]") {
    auto enemy = burstOnNeighbour(154, true, 4.0f);  // inside the 6-tile pulse, outside the Tesla's reach
    CHECK(enemy->hp < 100000);
    CHECK(enemy->freezeSlow == Catch::Approx(0.0f));
    CHECK(enemy->freezeTicks > 0);
}

TEST_CASE("A building without a deploy burst still does nothing on landing (plain Cannon)",
          "[deploy_effect][building]") {
    CHECK(burstOnNeighbour(25, false)->hp == 100000);
}

TEST_CASE("Cannon Evolution's Deploy Barrage deals 261 in a 2.0-tile radius",
          "[deploy_effect][building][live_sync_values]") {
    CHECK(100000 - burstOnNeighbour(129, true, 1.0f)->hp == 261);  // 2026-10-06
    // 2.6 tiles: inside the old 2.5 radius plus the 0.4 hitbox, outside the new 2.0.
    CHECK(burstOnNeighbour(129, true, 2.6f)->hp == 100000);        // 2025-05-05
}

TEST_CASE("Tesla Evolution's Electro Pulse deals 148", "[deploy_effect][building][live_sync_values]") {
    CHECK(100000 - burstOnNeighbour(154, true, 4.0f)->hp == 148);  // 2026-05-04
}
