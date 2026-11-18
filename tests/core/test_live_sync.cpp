// The 2026-11 live-game sync (docs/LIVE_SYNC.md): one block of cases per
// balance patch, oldest first, each tagged [live_sync][patch-YYYY-MM-DD] and
// written to fail on the engine before that patch's commit. The source for every
// value is the ledger row it cites.
#include "live_sync_test_helpers.h"

using namespace live_sync;

// ---- 2024-08-06 balance update (P01) --------------------------------------

TEST_CASE("Hunter hitpoints are 884 on the base card and its Evolution",
          "[live_sync][patch-2024-08-06]") {
    for (int id : { 62, 146 }) {
        Board board;
        auto body = mainBody(board, id);
        REQUIRE(body != nullptr);
        REQUIRE(body->hp == 884);
    }
    Board evoBoard;
    auto evolved = mainBody(evoBoard, 146, true);
    REQUIRE(evolved != nullptr);
    REQUIRE(evolved->hp == 884);
}

TEST_CASE("Wizard hitpoints are 754 on the base card, its Evolution and the Hero",
          "[live_sync][patch-2024-08-06]") {
    for (int id : { 11, 141, 167 }) {
        Board board;
        auto body = mainBody(board, id);
        REQUIRE(body != nullptr);
        REQUIRE(body->hp == 754);
    }
    Board evoBoard;
    auto evolved = mainBody(evoBoard, 141, true);
    REQUIRE(evolved != nullptr);
    REQUIRE(evolved->hp == 754);
}

TEST_CASE("Ice Wizard splash hits a unit 1.2 tiles from the target (radius 1.5)",
          "[live_sync][patch-2024-08-06]") {
    Board board;
    auto target = std::make_shared<StationaryCombatant>(900, 5.0f, 6.0f, 1000, 1, 5.0f, 10, 10);
    auto bystander = std::make_shared<StationaryCombatant>(901, 5.0f, 7.2f, 1000, 1, 5.0f, 10, 10);
    spawn(board, target);
    spawn(board, bystander);

    auto bodies = spawnBare(board, 34, 5.0f, 5.0f, 0);
    auto iceWizard = firstCombat(bodies);
    REQUIRE(iceWizard != nullptr);
    REQUIRE(iceWizard->splashRadius == Catch::Approx(1.5f));
    advancePastDeploy(iceWizard, board);
    iceWizard->update(board);
    board.commitPendingEntities();
    for (int i = 0; i < 5; ++i) {
        for (const auto& e : board.getEntities())
            if (std::dynamic_pointer_cast<Projectile>(e)) e->update(board);
        board.commitPendingEntities();
    }
    REQUIRE(target->hp < 1000);
    REQUIRE(bystander->hp < 1000);
}

// ---- 2024-10-08 balance update (October balance changes) ----

TEST_CASE("Magic Archer hp is 531 (base and Hero)", "[live_sync][patch-2024-10-08]") {
    Board board;
    auto base = mainBody(board, 63);
    REQUIRE(base != nullptr);
    CHECK(base->hp == 531);
    Board board2;
    auto hero = mainBody(board2, 171);
    REQUIRE(hero != nullptr);
    CHECK(hero->hp == 531);
}

TEST_CASE("Suspicious Bush moves at Medium speed", "[live_sync][patch-2024-10-08]") {
    Board board;
    auto bodies = spawnBare(board, 85);
    std::shared_ptr<Troop> bush;
    for (auto& e : bodies) if (auto t = std::dynamic_pointer_cast<Troop>(e)) { bush = t; break; }
    REQUIRE(bush != nullptr);
    CHECK(bush->getSpeed() == Catch::Approx(SPEED_MEDIUM * MOVEMENT_SPEED_SCALE).epsilon(0.001));
}

TEST_CASE("Evolved Zap lands two zaps, each at full damage", "[live_sync][patch-2024-10-08]") {
    Match m;
    auto giant = firstCombat(m.spawnCard(2, 9.0f, 10.0f, 1));
    REQUIRE(giant != nullptr);
    m.rec->damage.clear();
    m.spawnCard(124, 9.0f, 10.0f, 0, true);
    m.step(30);
    int total = m.rec->totalTo(giant->id);
    int first = 0;
    for (const auto& e : m.rec->damage)
        if (e.targetId == giant->id && e.amount > 0) { first = e.amount; break; }
    REQUIRE(first > 0);
    CHECK(total == 2 * first);
}
