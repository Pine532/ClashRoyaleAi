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

// ---- patch 2024-11-14 (november-balance-changes) ----

// Musketeer Evolution "Sniper Damage 326 -> 392": an empowered shot is
// base 217 x burst multiplier, truncated, as CombatEntity::getEffectiveDamage does.
TEST_CASE("Musketeer Evolution empowered shot deals 392",
          "[live_sync][patch-2024-11-14]") {
    Board board;
    auto m = mainBody(board, 140, true);
    REQUIRE(m != nullptr);
    CHECK(m->getBaseDamage() == 217);
    CHECK(static_cast<int>(m->getBaseDamage() * m->burstDamageMultiplier) == 392);
}

// ---- patch 2025-01-07: Attack Splash Radius 1.5 tiles (Baby Dragon Evolution, Witch + Evolution)
namespace p04_splash {
    inline std::shared_ptr<CombatEntity> spawnFirst(int id, bool evolved) {
        Board board;
        const CardDefinition* def = CardRegistry::getInstance().getCard(id);
        REQUIRE(def != nullptr);
        if (evolved) def->spawnEvolvedEntity(5.0f, 5.0f, 0, board);
        else         def->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto e = std::dynamic_pointer_cast<CombatEntity>(board.getEntities()[0]);
        REQUIRE(e != nullptr);
        return e;
    }
}

TEST_CASE("Baby Dragon Evolution (137) splashes 1.5 tiles in both forms", "[live_sync][patch-2025-01-07]") {
    REQUIRE(p04_splash::spawnFirst(137, false)->splashRadius == Catch::Approx(1.5f));
    REQUIRE(p04_splash::spawnFirst(137, true)->splashRadius == Catch::Approx(1.5f));
    REQUIRE(p04_splash::spawnFirst(44, false)->splashRadius == Catch::Approx(1.5f));
}

TEST_CASE("Witch (71) splashes 1.5 tiles", "[live_sync][patch-2025-01-07]") {
    REQUIRE(p04_splash::spawnFirst(71, false)->splashRadius == Catch::Approx(1.5f));
}

TEST_CASE("Witch Evolution (142) splashes 1.5 tiles in both forms", "[live_sync][patch-2025-01-07]") {
    REQUIRE(p04_splash::spawnFirst(142, false)->splashRadius == Catch::Approx(1.5f));
    REQUIRE(p04_splash::spawnFirst(142, true)->splashRadius == Catch::Approx(1.5f));
}

// --- 2025-02-04 balance update ---

TEST_CASE("Lava Hound's Lava Pups deal 84 damage", "[live_sync][patch-2025-02-04]") {
    Board board;
    auto hound = firstCombat(spawnBare(board, 91));
    REQUIRE(hound != nullptr);
    hound->takeDamage(hound->hp);
    board.cleanDeadEntities();
    board.commitPendingEntities();
    int pups = 0;
    for (const auto& e : board.getEntities()) {
        if (e->name != "Lava Pups") continue;
        auto c = std::dynamic_pointer_cast<CombatEntity>(e);
        REQUIRE(c != nullptr);
        CHECK(c->getBaseDamage() == 84);
        ++pups;
    }
    REQUIRE(pups == 6);
}

// --- 2025-03-04 balance update (supercell.com march-balance-changes) ---

TEST_CASE("Electro Wizard attack damage is 116 and his deploy zap stays 118",
          "[live_sync][patch-2025-03-04]") {
    Board board;
    auto bodies = spawnBare(board, 35);
    std::shared_ptr<MeleeTroop> wiz;
    int zaps = 0;
    for (const auto& e : bodies) {
        if (auto t = std::dynamic_pointer_cast<MeleeTroop>(e)) wiz = t;
        if (std::dynamic_pointer_cast<AreaSpell>(e)) ++zaps;
    }
    REQUIRE(wiz != nullptr);
    CHECK(wiz->getBaseDamage() == 116);
    // The post names only the attack; the deploy zap's damage is unsourced and unchanged.
    REQUIRE(zaps == 1);
    Match m;
    auto victim = std::make_shared<StationaryCombatant>(9001, 9.0f, 11.0f, 1000, 1, 5.0f, 10, 10);
    m.board().addEntity(victim);
    m.board().commitPendingEntities();
    m.spawnCard(35, 9.0f, 10.0f);
    m.step(1);
    CHECK(victim->hp == 1000 - 118);
}

TEST_CASE("Princess deploys in 1.2 s, as the card and as its Evolution (both halves)",
          "[live_sync][patch-2025-03-04]") {
    Board board0;
    auto base = mainBody(board0, 61);
    REQUIRE(base != nullptr);
    CHECK(base->deployTicksRemaining == 12);
    Board board;
    auto evoCard = mainBody(board, 145, false);
    REQUIRE(evoCard != nullptr);
    CHECK(evoCard->deployTicksRemaining == 12);
    auto evoForm = mainBody(board, 145, true);
    REQUIRE(evoForm != nullptr);
    CHECK(evoForm->deployTicksRemaining == 12);
}

// ---------------- 2025-04-09 (April balance changes) ----------------

// Tornado "Damage: 84x2 -> 84 (-50%)": the whole cast now deals 84 to a troop
// in its radius (the engine's Tornado is a single hit).
TEST_CASE("Tornado deals 84 damage in total to a troop in its radius", "[live_sync][patch-2025-04-09]") {
    Match m;
    auto bodies = m.spawnCard(0, 9.0f, 20.0f, 1); // enemy Knight, 1766 hp
    auto knight = firstCombat(bodies);
    REQUIRE(knight != nullptr);
    m.spawnCard(106, 9.0f, 20.0f, 0);
    m.step(30);
    REQUIRE(m.rec->totalTo(knight->id) == 84);
}

// ---- 2025-06-02 balance update (June 2025) ---------------------------------

TEST_CASE("Goblin Demolisher kamikaze form expires 10 s after it spawns", "[live_sync][patch-2025-06-02]") {
    Board board;
    auto demolisher = mainBody(board, 66);
    REQUIRE(demolisher != nullptr);
    demolisher->takeDamage(demolisher->hp / 2 + 1); // past 50%: transforms
    demolisher->update(board);
    REQUIRE_FALSE(demolisher->isAlive());
    board.cleanDeadEntities();
    board.commitPendingEntities();

    std::shared_ptr<CombatEntity> kamikaze;
    for (const auto& e : board.getEntities())
        if (e->name == "Kamikaze Goblin Demolisher") kamikaze = std::dynamic_pointer_cast<CombatEntity>(e);
    REQUIRE(kamikaze != nullptr);

    // No building anywhere, so only the lifetime cap can end it.
    for (int i = 0; i < 99; ++i) kamikaze->update(board);
    REQUIRE(kamikaze->isAlive());
    kamikaze->update(board);
    REQUIRE_FALSE(kamikaze->isAlive()); // 100 ticks = 10 s
}
