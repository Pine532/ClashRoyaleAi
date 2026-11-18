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

// ---------------------------------------------------------------------------
// 2025-07-07 balance update (ledger item 67): Ice Spirit Freeze Duration
// 1.2 s -> 1.1 s, i.e. an 11-tick stun, on the base card and the Evolution.
// ---------------------------------------------------------------------------
namespace {

// Ticks a freshly stunned Knight stays frozen after one Ice Spirit hit. The Knight
// is a stationary-enough, high-hp target; a control requires the hit to land.
int iceSpiritStunTicks(int cardId, bool evolved) {
    Match m;
    auto victims = m.spawnCard(0 /* Knight */, 9.0f, 12.0f, 1);
    auto victim = firstCombat(victims);
    REQUIRE(victim);
    m.spawnCard(cardId, 9.0f, 11.0f, 0, evolved);
    int frozen = 0;
    bool seen = false;
    for (int t = 0; t < 120; ++t) {
        m.step();
        if (victim->freezeTicks > 0) { seen = true; ++frozen; }
        else if (seen) break;
    }
    REQUIRE(seen);  // control: the hit landed
    return frozen;
}

} // namespace

TEST_CASE("Ice Spirit stuns for 1.1 s (11 ticks)", "[live_sync][patch-2025-07-07]") {
    REQUIRE(iceSpiritStunTicks(72, false) == 11);
}

TEST_CASE("Ice Spirit Evolution stuns for 1.1 s (11 ticks)", "[live_sync][patch-2025-07-07]") {
    REQUIRE(iceSpiritStunTicks(144, false) == 11);
}

TEST_CASE("Ice Spirit Evolution evolved form stuns for 1.1 s (11 ticks)", "[live_sync][patch-2025-07-07]") {
    // The evolved stat block (spawnEvolvedEntity) restates the stun separately.
    REQUIRE(iceSpiritStunTicks(144, true) == 11);
}

// ---- patch-2025-08-06 (P11): Dart Goblin Evolution poison cloud, Firecracker sight ----

TEST_CASE("Dart Goblin Evolution poison mark lasts 1 s (2025-08-06 balance update)",
          "[live_sync][patch-2025-08-06]") {
    Match m;
    auto victim = firstCombat(m.spawnCard(19, 9.0f, 12.0f, 1)); // Golem: survives the darts
    REQUIRE(victim);
    m.spawnCard(131, 9.0f, 8.0f, 0, true);
    int markTicks = 0;
    for (int i = 0; i < 100 && markTicks == 0; ++i) {
        m.step();
        markTicks = victim->dotTicksRemaining;
    }
    REQUIRE(markTicks > 0);
    // Official: Poison Cloud Duration 4 s -> 1 s (10 ticks; the mark is applied
    // before that tick's decrement may already have run, so allow <= 10).
    CHECK(markTicks <= 10);
}

TEST_CASE("Firecracker sight range is 8.0 tiles on the base and Evolution ids (2025-08-06)",
          "[live_sync][patch-2025-08-06]") {
    Board b1;
    auto base = mainBody(b1, 64);
    REQUIRE(base);
    CHECK(base->sightRange == 8.0f);
    Board b2;
    auto evo = mainBody(b2, 130, true);
    REQUIRE(evo);
    CHECK(evo->sightRange == 8.0f);
    Board b3;
    auto evoBase = mainBody(b3, 130, false);
    REQUIRE(evoBase);
    CHECK(evoBase->sightRange == 8.0f);
}

// --- patch 2025-09-01 ---------------------------------------------------------

TEST_CASE("Goblin Curse deals 35 per second for 6 seconds (210 total)",
          "[live_sync][patch-2025-09-01]") {
    // Source: 2025-09-01 September Balance Changes, "Damage/Sec: 30 -> 35".
    Match m;
    auto target = firstCombat(m.spawnCard(25, 9.0f, 26.0f, 1));  // Cannon: stationary, survives 210
    REQUIRE(target);
    m.spawnCard(102, 9.0f, 26.0f, 0);
    m.step(120);
    CHECK(m.rec->totalTo(target->id) == 210);
}

TEST_CASE("The Log deals 268 damage to a troop it sweeps",
          "[live_sync][patch-2025-09-01]") {
    // Source: 2025-09-01 "New Season: Back to Drool", "Damage: 266 -> 268".
    Match m;
    auto target = firstCombat(m.spawnCard(25, 9.0f, 26.0f, 1));  // Cannon: stationary
    REQUIRE(target);
    m.spawnCard(33, 9.0f, 24.0f, 0);
    m.step(60);
    CHECK(m.rec->totalTo(target->id) == 268);
}

// ---------------------------------------------------------------------------
// Patch 2025-09-25 (October 2025 update, bug fixes)
// ---------------------------------------------------------------------------

TEST_CASE("Vines top-HP targeting counts shields", "[live_sync][patch-2025-09-25]") {
    Board board;
    // B has the lower bare hp, but the higher hp + shield.
    auto a = firstCombat(spawnBare(board, 0, 9.0f, 10.0f, 1));
    auto b = firstCombat(spawnBare(board, 0, 11.0f, 10.0f, 1));
    REQUIRE(a);
    REQUIRE(b);
    b->hp = 500;
    b->shieldHp = 2000;
    const int aHp = a->hp;

    // targetTopHpCount = 1 (same constructor as the Vines cases in test_area_spell.cpp)
    AreaSpell spell(4, 10.0f, 10.0f, 0, 5.0f, 100, 0, '*', nullptr, false, 1, 0,
        false, 1.0f, 0, 0.0f, nullptr, false, 1);
    spell.update(board);

    REQUIRE(a->hp == aHp);        // full hp, no shield: not the top target
    REQUIRE(b->shieldHp == 1900); // 500 hp + 2000 shield: hit (the shield absorbs)
}

TEST_CASE("Prince charge is cleared by a stun", "[live_sync][patch-2025-09-25]") {
    for (int id : { 14, 46, 87 }) {
        Match m;
        auto troop = firstCombat(m.spawnCard(id, 3.0f, 3.0f, 0));
        REQUIRE(troop);
        int guard = 0;
        while (!troop->isCharging() && guard++ < 400) m.step();
        INFO("card id " << id);
        REQUIRE(troop->isCharging());

        troop->applyFreeze(3, 0.0f); // a Zap-style stun
        m.step();
        REQUIRE_FALSE(troop->isCharging());
        REQUIRE(troop->chargeProgress < troop->chargeThreshold);
    }
}

TEST_CASE("Prince charge is not cleared by a slow", "[live_sync][patch-2025-09-25]") {
    Match m;
    auto troop = firstCombat(m.spawnCard(14, 3.0f, 3.0f, 0));
    REQUIRE(troop);
    int guard = 0;
    while (!troop->isCharging() && guard++ < 400) m.step();
    REQUIRE(troop->isCharging());
    troop->applyFreeze(3, 0.5f); // a slow, not a stun
    m.step();
    REQUIRE(troop->isCharging());
}

TEST_CASE("Prince charge interruption leaves the Bandit dash alone", "[live_sync][patch-2025-09-25]") {
    // Bandit and Boss Bandit model a dash (the 2026-09-07 post: not interruptible).
    for (int id : { 50, 122 }) {
        Match m;
        auto troop = firstCombat(m.spawnCard(id, 3.0f, 3.0f, 0));
        REQUIRE(troop);
        int guard = 0;
        while (!troop->isCharging() && guard++ < 400) m.step();
        INFO("card id " << id);
        REQUIRE(troop->isCharging());
        troop->applyFreeze(3, 0.0f);
        m.step();
        REQUIRE(troop->isCharging());
    }
}

// ---- Patch 2025-10-06 (October balance changes): slows 35% -> 30%, Rage boost
// 35% -> 30%, Vines total damage 335 -> 306. Source: the official post, see
// docs/LIVE_SYNC.md. The engine's freezeSlow is a speed multiplier (30% slow = 0.70).

TEST_CASE("Giant Snowball slows by 30% (speed factor 0.70)", "[live_sync][patch-2025-10-06]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000, 1, 5.0f, 10, 10);
    spawn(board, enemy);
    spawnBare(board, 100, 5.0f, 5.0f, 0);
    auto spell = board.getEntities().back();
    while (spell->isAlive()) spell->update(board);
    REQUIRE(enemy->freezeTicks == 15);
    REQUIRE(enemy->freezeSlow == Catch::Approx(0.70f));
}

TEST_CASE("Ice Golem and Hero Ice Golem death slow is 30% (speed factor 0.70)", "[live_sync][patch-2025-10-06]") {
    for (int id : { 40, 175 }) {
        Board board;
        auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 5.5f, 1000, 1, 5.0f, 10, 10);
        spawn(board, enemy);
        auto golem = mainBody(board, id);
        REQUIRE(golem != nullptr);
        golem->position = { 5.0f, 5.0f };
        golem->hp = 0;
        board.cleanDeadEntities();
        INFO("card id " << id);
        REQUIRE(enemy->freezeTicks == 30);
        REQUIRE(enemy->freezeSlow == Catch::Approx(0.70f));
    }
}

TEST_CASE("Ice Wizard slow is 30% (speed factor 0.70)", "[live_sync][patch-2025-10-06]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000, 1, 5.0f, 10, 10);
    spawn(board, enemy);
    auto wiz = spawnBare(board, 34, 5.0f, 5.0f, 0).front();
    advancePastDeploy(wiz, board);
    wiz->update(board);
    board.commitPendingEntities();
    auto projectile = std::dynamic_pointer_cast<Projectile>(board.getEntities().back());
    REQUIRE(projectile != nullptr);
    projectile->update(board);
    REQUIRE(enemy->freezeTicks == 30);
    REQUIRE(enemy->freezeSlow == Catch::Approx(0.70f));
}

TEST_CASE("Rage boosts allied damage by 30%", "[live_sync][patch-2025-10-06]") {
    Board board;
    auto ally = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000, 0, 5.0f, 10, 10);
    spawn(board, ally);
    spawnBare(board, 108, 5.0f, 5.0f, 0);
    auto spell = board.getEntities().back();
    for (int i = 0; i < 50 && ally->buffTicksRemaining == 0; ++i) spell->update(board);
    REQUIRE(ally->buffTicksRemaining > 0);
    REQUIRE(ally->buffDamageMultiplier == Catch::Approx(1.30f));
}

TEST_CASE("Vines deals 306 total (102 x 3 pulses) to a troop", "[live_sync][patch-2025-10-06]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 5000, 1, 5.0f, 10, 10);
    spawn(board, enemy);
    spawnBare(board, 105, 5.0f, 5.0f, 0);
    auto spell = board.getEntities().back();
    while (spell->isAlive()) spell->update(board);
    REQUIRE(enemy->hp == 5000 - 306);
}

// ---------------------------------------------------------------------------
// Patch 2025-10-24 (official "October balance changes"): Rage boost 35% -> 30%
// (Evolved Barbarians, Lumberjack), Distance Needed for Charge 2 -> 2.5 tiles
// (Prince, Ram Rider; Evolved Royal Recruits was already 2.5).
// ---------------------------------------------------------------------------

TEST_CASE("Evolved Barbarians Blade Rage is +30% attack speed (cooldown x 1/1.30)",
          "[live_sync][patch-2025-10-24]") {
    Board board;
    auto evo = mainBody(board, 155, true);
    REQUIRE(evo != nullptr);
    CHECK(evo->selfHasteDurationTicks == 30);
    CHECK(evo->selfHasteCooldownMultiplier == Catch::Approx(1.0f / 1.30f).margin(0.005f));
}

TEST_CASE("Lumberjack death Rage buffs allies by 30%, base and evolved",
          "[live_sync][patch-2025-10-24]") {
    for (bool evolved : { false, true }) {
        CAPTURE(evolved);
        Board board;
        auto lj = mainBody(board, evolved ? 156 : 21, evolved);
        REQUIRE(lj != nullptr);
        auto ally = firstCombat(spawnBare(board, 24, 9.5f, 10.0f, 0)); // Skeletons, 0.5 tiles away
        REQUIRE(ally != nullptr);
        REQUIRE(ally->buffDamageMultiplier == 1.0f);
        lj->hp = 0;
        lj->onDeath(board);
        CHECK(ally->buffTicksRemaining == 55);
        CHECK(ally->buffDamageMultiplier == Catch::Approx(1.30f));
    }
}

TEST_CASE("Prince needs 2.5 tiles of run-up to charge",
          "[live_sync][patch-2025-10-24]") {
    Board board;
    auto prince = mainBody(board, 14);
    REQUIRE(prince != nullptr);
    CHECK(prince->chargeThreshold == Catch::Approx(2.5f));
    CHECK(prince->chargeMultiplier == Catch::Approx(2.0f));
}

TEST_CASE("Ram Rider's ram needs 2.5 tiles of run-up to charge",
          "[live_sync][patch-2025-10-24]") {
    Board board;
    auto ram = mainBody(board, 87);
    REQUIRE(ram != nullptr);
    CHECK(ram->chargeThreshold == Catch::Approx(2.5f));
    CHECK(ram->chargeMultiplier == Catch::Approx(2.0f));
}

TEST_CASE("Wall Breakers Evolution Runner damage is 184 after the 2025-11-03 patch",
          "[live_sync][patch-2025-11-03]") {
    GameManager game({ 123, 1, 2, 3, 4, 5, 6, 7 }, { 0, 1, 2, 3, 4, 5, 6, 7 });
    game.playerAI.elixir = 100.0f;
    for (int i = 0; i < 3; ++i) {
        game.playerAI.hand[0] = 123;
        game.playerAI.handCooldownTicks[0] = 0;
        game.playCard(0, 123, 9.0f, 10.0f);
        game.step();
    }
    std::shared_ptr<CombatEntity> evolved;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name != "Wall Breakers") continue;
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->deathEffect) { evolved = ce; break; }
    }
    REQUIRE(evolved != nullptr);
    evolved->takeDamage(evolved->hp);
    game.getBoard().cleanDeadEntities();
    game.getBoard().commitPendingEntities();
    std::shared_ptr<CombatEntity> runner;
    for (const auto& e : game.getBoard().getEntities()) {
        if (e->name == "Runner") runner = std::dynamic_pointer_cast<CombatEntity>(e);
    }
    REQUIRE(runner != nullptr);
    REQUIRE(runner->getBaseDamage() == 184);
}

// ===== Patch 2025-12-01 (December balance changes) =====

TEST_CASE("Freeze deals 148 damage to a troop in radius and still stuns it for 4 s",
        "[live_sync][patch-2025-12-01]") {
    Board board;
    auto giant = firstCombat(spawnBare(board, 25, 9.0f, 10.0f, 1)); // any high-hp enemy troop
    REQUIRE(giant != nullptr);
    const int before = giant->hp;
    auto bodies = spawnBare(board, 107, 9.0f, 10.0f, 0);
    auto spell = std::dynamic_pointer_cast<AreaSpell>(bodies.front());
    REQUIRE(spell != nullptr);
    for (int i = 0; i < 200 && spell->isAlive(); ++i) spell->update(board);
    REQUIRE(before - giant->hp == 148);
    REQUIRE(giant->freezeTicks > 0); // control: the stun still lands
}

TEST_CASE("Lava Hound's Lava Pups spawn with 215 hitpoints",
        "[live_sync][patch-2025-12-01]") {
    Board board;
    auto hound = firstCombat(spawnBare(board, 91, 9.0f, 10.0f, 0));
    REQUIRE(hound != nullptr);
    std::set<int> before;
    for (const auto& e : board.getEntities()) before.insert(e->id);
    hound->takeDamage(hound->hp);
    board.cleanDeadEntities(); // fires the death effect
    board.commitPendingEntities();
    int pups = 0;
    for (const auto& e : board.getEntities()) {
        if (before.count(e->id)) continue;
        auto c = std::dynamic_pointer_cast<CombatEntity>(e);
        if (!c) continue;
        ++pups;
        REQUIRE(c->hp == 215);
    }
    REQUIRE(pups == 6);
}

TEST_CASE("Valkyrie Evolution's tornado pulls and damages at 5.0 tiles, not 5.5",
        "[live_sync][patch-2025-12-01]") {
    Board board;
    auto valk = firstCombat(spawnBare(board, 147, 9.0f, 10.0f, 0, true));
    REQUIRE(valk != nullptr);
    auto near = std::make_shared<DummyEntity>(9001, 9.0f, 11.0f, 1000, 1);   // in attack range
    auto edge = std::make_shared<DummyEntity>(9002, 9.0f, 15.2f, 1000, 1);   // 5.2 tiles: between 5.0 and 5.5
    auto inside = std::make_shared<DummyEntity>(9003, 9.0f, 14.5f, 1000, 1); // 4.5 tiles: inside both
    board.addEntity(near); board.addEntity(edge); board.addEntity(inside);
    board.commitPendingEntities();
    for (int i = 0; i < 40; ++i) { valk->update(board); board.commitPendingEntities(); }
    REQUIRE(near->hp < 1000);   // control: she attacked, so the pull fired
    REQUIRE(inside->hp < 1000); // control: 4.5 tiles is still in the tornado
    REQUIRE(edge->hp == 1000);  // 5.2 tiles is now outside
}

// ---- patch 2026-01-12 ----

TEST_CASE("Skeleton Army Evolution ghost skeletons move at Medium speed",
          "[live_sync][patch-2026-01-12]") {
    // 2026-01-12 balance update: "Ghost Skeletons' Speed: Fast -> Medium".
    Match m;
    auto squad = m.spawnCard(133, 9.0f, 10.0f, 0, true);
    std::shared_ptr<Troop> skel;
    for (const auto& e : squad)
        if (e->cardId == 12) { skel = std::dynamic_pointer_cast<Troop>(e); break; }
    REQUIRE(skel);
    const float skelSpeed = skel->getSpeed();
    skel->hp = 0;
    m.step(2);
    std::shared_ptr<Troop> ghost;
    for (const auto& e : m.board().getEntities())
        if (e->cardId == -43) { ghost = std::dynamic_pointer_cast<Troop>(e); break; }
    REQUIRE(ghost);
    // The living skeletons stay Fast; the ghost is Medium (60/90 of Fast).
    REQUIRE(ghost->getSpeed() == Catch::Approx(skelSpeed * SPEED_MEDIUM / SPEED_FAST).epsilon(1e-4));
}

// ---- 2026-02-02 balance update ----------------------------------------------
TEST_CASE("Three Musketeers hit speed is 1.3 s (13 ticks)", "[live_sync][patch-2026-02-02]") {
    Board board;
    auto bodies = spawnBare(board, 80);
    REQUIRE(bodies.size() == 3);
    for (const auto& e : bodies) {
        auto c = std::dynamic_pointer_cast<CombatEntity>(e);
        REQUIRE(c);
        REQUIRE(c->getAttackCooldown() == 13);
    }
}

// ---- patch 2026-02-23 (february-balance-changes-26) ----

// Hero Goblins "Banner Duration: 7s -> 5s": the post-death window is 50 ticks.
TEST_CASE("Hero Goblins Banner Brigade window is 5 s",
          "[live_sync][patch-2026-02-23]") {
    const CardDefinition* def = CardRegistry::getInstance().getCard(172);
    REQUIRE(def != nullptr);
    CHECK(def->abilityUsableAfterDeathTicks == 50);

    GameManager game({ 1, 172, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.hand[1] = 172;
    game.playCard(0, 172, 9.0f, 10.0f);
    game.step();
    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->cardId == 172 && ce->team == 0) ce->takeDamage(ce->hp);
    }
    game.step(); // wipe detected
    game.playerAI.elixir = 10.0f;
    for (int i = 0; i < 55; ++i) { // past 5 s, inside the old 7 s
        game.step();
        game.playerAI.elixir = 10.0f;
    }
    CHECK_FALSE(game.isChampionAbilityReady(0, 1));
}
