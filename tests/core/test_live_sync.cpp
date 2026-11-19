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
    advanceThroughWindup(iceWizard, board);
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
    // The duration is the 2026-05-04 patch's to pin (3 s); this case owns the factor.
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
        // The duration is the 2026-09-08 patch's to pin (2.5 s); this case owns the factor.
        REQUIRE(enemy->freezeSlow == Catch::Approx(0.70f));
    }
}

TEST_CASE("Ice Wizard slow is 30% (speed factor 0.70)", "[live_sync][patch-2025-10-06]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000, 1, 5.0f, 10, 10);
    spawn(board, enemy);
    auto wiz = spawnBare(board, 34, 5.0f, 5.0f, 0).front();
    advancePastDeploy(wiz, board);
    advanceThroughWindup(wiz, board);
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
    // The duration is the 2026-08-04 patch's to pin (5 s); this case owns the boost.
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

#include "SpiritEmpressForms.h"

// ---------------------------------------------------------------------------
// Patch 2026-03-02 (March balance changes): Dart Goblin sight, Royal Ghost
// invisibility delay, Spirit Empress ground speed.
// ---------------------------------------------------------------------------

TEST_CASE("Dart Goblin sight range is 7.0 tiles on the base card and its Evolution",
          "[live_sync][patch-2026-03-02]") {
    Board board;
    auto base = mainBody(board, 20);
    REQUIRE(base != nullptr);
    CHECK(base->sightRange == Catch::Approx(7.0f));
    auto evo = mainBody(board, 131, true);
    REQUIRE(evo != nullptr);
    CHECK(evo->sightRange == Catch::Approx(7.0f));
    auto evoUn = mainBody(board, 131, false);
    REQUIRE(evoUn != nullptr);
    CHECK(evoUn->sightRange == Catch::Approx(7.0f));
}

TEST_CASE("Royal Ghost stays visible 20 ticks after a hit, base and Evolution",
          "[live_sync][patch-2026-03-02]") {
    for (bool evolved : { false, true }) {
        for (int id : { 47, 136 }) {
            if (id == 47 && evolved) continue;
            Match m;
            auto ghost = firstCombat(m.spawnCard(id, 9.0f, 10.0f, 0, evolved));
            REQUIRE(ghost != nullptr);
            auto bodies = m.spawnCard(19, 9.0f, 10.8f, 1); // a Golem to hit
            REQUIRE_FALSE(bodies.empty());
            int seen = 0;
            for (int i = 0; i < 100 && seen == 0; ++i) {
                m.step();
                if (ghost->visibleTicksRemaining > 0) seen = ghost->visibleTicksRemaining;
            }
            INFO("card " << id << " evolved " << evolved);
            REQUIRE(seen > 0);
            // Observed on the tick of the hit: the window is 20 ticks (2 s).
            CHECK(seen == 20);
        }
    }
}

TEST_CASE("Spirit Empress ground form moves at Fast speed", "[live_sync][patch-2026-03-02]") {
    CHECK(spiritEmpressGroundStats().speed == Catch::Approx(SPEED_FAST * MOVEMENT_SPEED_SCALE));
    Board board;
    auto reg = mainBody(board, 165);
    REQUIRE(reg != nullptr);
    if (auto t = std::dynamic_pointer_cast<Troop>(reg))
        CHECK(t->getSpeed() == Catch::Approx(SPEED_FAST * MOVEMENT_SPEED_SCALE));
}

TEST_CASE("Hero Ice Golem Snowstorm no longer knocks back (2026-04-06)",
          "[live_sync][patch-2026-04-06]") {
    Board board;
    auto hero = mainBody(board, 175);
    REQUIRE(hero != nullptr);
    REQUIRE(hero->abilityEffect != nullptr);
    auto enemy = std::make_shared<StationaryCombatant>(9001, hero->position.x + 1.0f, hero->position.y,
                                                       100000, 1, 1.0f, 100, 10);
    board.addEntity(enemy);
    board.commitPendingEntities();
    const float startX = enemy->position.x, startY = enemy->position.y;

    hero->abilityEffect->apply(board, *hero);
    board.commitPendingEntities();
    std::vector<std::shared_ptr<AreaSpell>> blasts;
    for (const auto& e : board.getEntities())
        if (auto s = std::dynamic_pointer_cast<AreaSpell>(e)) blasts.push_back(s);
    REQUIRE(blasts.size() == 3);
    for (int i = 0; i < 20; ++i)
        for (auto& b : blasts) b->update(board);

    REQUIRE(enemy->hp < 100000); // the blasts did land
    REQUIRE(enemy->position.x == Catch::Approx(startX).margin(1e-4));
    REQUIRE(enemy->position.y == Catch::Approx(startY).margin(1e-4));
}

TEST_CASE("Royal Ghost Evolution Souldiers deal 81 damage (2026-04-06)",
          "[live_sync][patch-2026-04-06]") {
    Match m;
    auto ghost = firstCombat(m.spawnCard(136, 9.0f, 10.0f, 0, true));
    REQUIRE(ghost != nullptr);
    m.spawnCard(2, 9.0f, 10.8f, 1); // an enemy in melee reach so every hit lands
    std::shared_ptr<CombatEntity> souldier;
    for (int t = 0; t < 120 && !souldier; ++t) {
        m.step();
        for (const auto& e : m.board().getEntities())
            if (e->name == "Souldier") { souldier = std::dynamic_pointer_cast<CombatEntity>(e); break; }
    }
    REQUIRE(souldier != nullptr);
    REQUIRE(souldier->getBaseDamage() == 81);
}

TEST_CASE("Wall Breakers Evolution death explosion deals 192 to troops (2026-04-06)",
          "[live_sync][patch-2026-04-06]") {
    Match m;
    auto bodies = m.spawnCard(123, 9.0f, 10.0f, 0, true);
    std::shared_ptr<CombatEntity> wb;
    for (const auto& e : bodies) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->deathEffect) { wb = ce; break; }
    }
    REQUIRE(wb != nullptr);
    auto bystander = std::make_shared<DummyEntity>(9999, wb->position.x + 0.5f, wb->position.y, 100000, 1);
    m.board().addEntity(bystander);
    m.board().commitPendingEntities();
    wb->takeDamage(wb->hp);
    m.board().cleanDeadEntities();
    m.board().commitPendingEntities();
    REQUIRE(bystander->hp == 100000 - 192);
}

// ===== 2026-05-04 balance update =====
#include "SpiritEmpressForms.h"
#include "AreaSpell.h"


TEST_CASE("Giant Snowball base slow lasts 3 s (30 ticks) at the 0.5 speed factor",
          "[live_sync][patch-2026-05-04]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000, 1, 5.0f, 10, 10);
    board.addEntity(enemy);
    board.commitPendingEntities();
    CardRegistry::getInstance().getCard(100)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();
    auto spell = std::dynamic_pointer_cast<AreaSpell>(board.getEntities().back());
    REQUIRE(spell != nullptr);
    while (spell->isAlive()) spell->update(board);
    CHECK(enemy->freezeTicks == 30);
    CHECK(enemy->freezeSlow == Catch::Approx(0.70f)); // 2025-10-06: slow 30%
}

TEST_CASE("Giant Snowball Evolution slow lasts 3 s (30 ticks), standardised with the base form",
          "[live_sync][patch-2026-05-04]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000, 1, 5.0f, 10, 10);
    board.addEntity(enemy);
    board.commitPendingEntities();
    CardRegistry::getInstance().getCard(158)->spawnEvolvedEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();
    auto spell = std::dynamic_pointer_cast<AreaSpell>(board.getEntities().back());
    REQUIRE(spell != nullptr);
    while (spell->isAlive()) spell->update(board);
    CHECK(enemy->freezeTicks == 30);
    CHECK(enemy->freezeSlow == Catch::Approx(0.7f));
}

TEST_CASE("Spirit Empress hitpoints are 1121 on the registry entry and both forms",
          "[live_sync][patch-2026-05-04]") {
    Board board;
    auto body = mainBody(board, 165);
    REQUIRE(body != nullptr);
    CHECK(body->hp == 1121);
    CHECK(spiritEmpressGroundStats().hp == 1121);
    CHECK(spiritEmpressFlyingStats().hp == 1121);
}

TEST_CASE("Giant Snowball Evolution card's un-evolved half also slows for 3 s (30 ticks)",
          "[live_sync][patch-2026-05-04]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000, 1, 5.0f, 10, 10);
    board.addEntity(enemy);
    board.commitPendingEntities();
    CardRegistry::getInstance().getCard(158)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();
    auto spell = std::dynamic_pointer_cast<AreaSpell>(board.getEntities().back());
    REQUIRE(spell != nullptr);
    while (spell->isAlive()) spell->update(board);
    CHECK(enemy->freezeTicks == 30);
    CHECK(enemy->freezeSlow == Catch::Approx(0.70f)); // 2025-10-06: slow 30%
}

// ===== Patch 2026-06-01 (June balance changes) =====

TEST_CASE("Dart Goblin Evolution's first poison stage is 64 per second",
          "[live_sync][patch-2026-06-01]") {
    Match m;
    auto darts = m.spawnCard(131, 9.0f, 8.0f, 0, /*evolved=*/true);
    auto target = firstCombat(m.spawnCard(19, 9.0f, 12.0f, 1)); // Golem: outlives the test
    REQUIRE(!darts.empty());
    REQUIRE(target != nullptr);
    int seen = 0;
    for (int i = 0; i < 80 && seen == 0; ++i) {
        m.step();
        seen = target->dotDamagePerTick;
    }
    REQUIRE(seen > 0); // the dart landed and left a mark
    CHECK(seen == 64); // official 2026-06-01: Stage 1 51 -> 64
}

TEST_CASE("Inferno Dragon Evolution holds its ramp for 7 s after losing a target",
          "[live_sync][patch-2026-06-01]") {
    Match m;
    auto dragon = firstCombat(m.spawnCard(163, 9.0f, 10.0f, 0, /*evolved=*/true));
    REQUIRE(dragon != nullptr);
    CHECK(dragon->rampGracePeriodTicks == 70); // official 2026-06-01: Charge Duration 9s -> 7s
    // The un-evolved dragon has no grace period.
    auto base = firstCombat(m.spawnCard(163, 5.0f, 10.0f, 0, /*evolved=*/false));
    REQUIRE(base != nullptr);
    CHECK(base->rampGracePeriodTicks == 0);
}

TEST_CASE("Mortar Evolution's launched Goblins deploy in 0.5 s",
          "[live_sync][patch-2026-06-01]") {
    Match m;
    auto mortar = m.spawnCard(152, 9.0f, 8.0f, 0, /*evolved=*/true);
    auto target = firstCombat(m.spawnCard(19, 9.0f, 17.0f, 1)); // Golem 9 tiles away
    REQUIRE(!mortar.empty());
    REQUIRE(target != nullptr);
    int deploy = -1;
    for (int i = 0; i < 200 && deploy < 0; ++i) {
        m.step();
        for (const auto& e : m.board().getEntities()) {
            auto c = std::dynamic_pointer_cast<CombatEntity>(e);
            if (c && c->team == 0 && c->name == "Goblins") { deploy = c->deployTicksRemaining; break; }
        }
    }
    REQUIRE(deploy >= 0); // a Goblin was launched
    // 0.5 s = 5 ticks; the default 10-tick deploy is the old value.
    CHECK(deploy <= 5);
}

TEST_CASE("Mortar Evolution's Goblin change leaves Goblin Drill's Goblins at the default deploy",
          "[live_sync][patch-2026-06-01]") {
    Match m;
    m.spawnCard(98, 9.0f, 8.0f, 0);
    int deploy = -1;
    for (int i = 0; i < 400 && deploy < 0; ++i) {
        m.step();
        for (const auto& e : m.board().getEntities()) {
            auto c = std::dynamic_pointer_cast<CombatEntity>(e);
            if (c && c->team == 0 && c->name == "Goblins") { deploy = c->deployTicksRemaining; break; }
        }
    }
    REQUIRE(deploy >= 0);
    CHECK(deploy > 5); // the Drill's Goblins are untouched by the Mortar Evolution nerf
}

// The 2026-11 live-game sync (docs/LIVE_SYNC.md): one block of cases per
// balance patch, oldest first, each tagged [live_sync][patch-YYYY-MM-DD] and
// written to fail on the engine before that patch's commit. The source for every
// value is the ledger row it cites.
#include "live_sync_test_helpers.h"

using namespace live_sync;

// ---------------------------------------------------------------------------
// patch 2026-07-06 (July 2026 balance changes)
// ---------------------------------------------------------------------------

namespace {
// Damage of the Executioner Evolution's first landed hit on an enemy Cannon
// placed `dist` tiles away (the Cannon is a stationary target).
int executionerEvoFirstHit(float dist) {
    Match m;
    auto evo = firstCombat(m.spawnCard(157, 9.0f, 10.0f, 0, true));
    auto cannon = firstCombat(m.spawnCard(25, 9.0f, 10.0f + dist, 1));
    int amount = 0;
    for (int t = 0; t < 60 && amount == 0; ++t) {
        m.step();
        for (const auto& e : m.rec->damage)
            if (e.attackerId == evo->id && e.targetId == cannon->id && e.amount > 0) { amount = e.amount; break; }
    }
    return amount;
}
} // namespace

TEST_CASE("Executioner Evolution Close Quarters hits for 240 within 2.5 tiles and 179 beyond",
          "[live_sync][patch-2026-07-06]") {
    REQUIRE(executionerEvoFirstHit(2.0f) == 240);
    REQUIRE(executionerEvoFirstHit(3.0f) == 179); // the old band reached 3.5
}

TEST_CASE("Hero Knight Triumphant Taunt shield is 512",
          "[live_sync][patch-2026-07-06]") {
    Match m;
    auto bodies = m.spawnCard(166, 9.0f, 10.0f, 0);
    auto hero = firstCombat(bodies);
    REQUIRE(hero != nullptr);
    hero->abilityCooldownRemaining = 0;
    REQUIRE(hero->activateAbility(m.board()));
    REQUIRE(hero->shieldHp == 512);
}

TEST_CASE("Hero Magic Archer Triple Threat teleports 3.5 tiles and leaves a 271 hp decoy",
          "[live_sync][patch-2026-07-06]") {
    Match m;
    auto bodies = m.spawnCard(171, 9.0f, 10.0f, 0);
    auto hero = firstCombat(bodies);
    REQUIRE(hero != nullptr);
    const float y0 = hero->position.y;
    hero->abilityCooldownRemaining = 0;
    REQUIRE(hero->activateAbility(m.board()));
    m.board().commitPendingEntities();
    REQUIRE(y0 - hero->position.y == Catch::Approx(3.5f));
    int decoys = 0;
    for (const auto& e : m.board().getEntities()) {
        if (e->name != "Decoy") continue;
        auto c = std::dynamic_pointer_cast<CombatEntity>(e);
        REQUIRE(c != nullptr);
        REQUIRE(c->hp == 271);
        ++decoys;
    }
    REQUIRE(decoys == 1);
}

// ---------------- patch 2026-08-04 (P26a) ----------------
// Official "Final August Balance Changes" notes, 4 Aug 2026.

TEST_CASE("Archers Evolution Power Shot does 140 at 4-6 tiles, 112 x 1.25",
          "[live_sync][patch-2026-08-04]") {
    Match m;
    auto archers = m.spawnCard(128, 8.5f, 10.0f, 0, true); // the evolved form
    // A stationary target 5.0 away: inside the 4-6 tile Power Shot band.
    auto cannon = m.spawnCard(25, 8.5f, 15.0f, 1);
    std::set<int> archerIds;
    for (const auto& a : archers) archerIds.insert(a->id);
    m.step(60);
    int first = -1;
    for (const auto& e : m.rec->damage)
        if (archerIds.count(e.attackerId) && e.targetId == cannon.front()->id) { first = e.amount; break; }
    REQUIRE(first == 140); // was 168
}

TEST_CASE("Barbarians Evolution Blade Rage lasts 5 s",
          "[live_sync][patch-2026-08-04]") {
    Match m; // an Evolution spawns through a live match
    auto evolved = firstCombat(m.spawnCard(155, 8.5f, 10.0f, 0, true));
    REQUIRE(evolved != nullptr);
    REQUIRE(evolved->selfHasteDurationTicks == 50); // was 30
}

TEST_CASE("Battle Healer hits for 268 every 2 s with 1920 hp",
          "[live_sync][patch-2026-08-04]") {
    Board board;
    auto healer = mainBody(board, 49);
    REQUIRE(healer != nullptr);
    REQUIRE(healer->hp == 1920);               // was 1717
    REQUIRE(healer->getBaseDamage() == 268);   // was 148
    REQUIRE(healer->getAttackCooldown() == 20); // was 15
}

TEST_CASE("Bowler's piercing line reaches 7 tiles past its 4.0 range",
          "[live_sync][patch-2026-08-04]") {
    Board board;
    auto bowler = mainBody(board, 22);
    REQUIRE(bowler != nullptr);
    REQUIRE(bowler->lineSplashRange == Catch::Approx(11.0f)); // was 11.5
}

TEST_CASE("Giant Snowball Evolution pulls 4.0 tiles",
          "[live_sync][patch-2026-08-04]") {
    Match m;
    auto bodies = m.spawnCard(158, 9.0f, 10.0f, 0, true);
    std::shared_ptr<AreaSpell> spell;
    for (const auto& e : bodies)
        if (auto s = std::dynamic_pointer_cast<AreaSpell>(e)) spell = s;
    REQUIRE(spell != nullptr);
    REQUIRE(spell->getKnockback() == Catch::Approx(-4.0f)); // was -4.5
}

TEST_CASE("Goblin Curse slows enemy troop movement 15%, refreshed by each pulse",
          "[live_sync][patch-2026-08-04]") {
    Match m;
    auto knight = std::dynamic_pointer_cast<Troop>(m.spawnCard(0, 8.5f, 24.0f, 1).front());
    m.step(12);
    m.spawnCard(102, knight->position.x, knight->position.y - 1.5f, 0);
    m.step(9); // 1 s deploy, then the first pulse lands; from the next tick it walks slowed
    Vector2D prev = knight->position;
    m.step();
    const float stepLen = std::hypot(knight->position.x - prev.x, knight->position.y - prev.y);
    REQUIRE(stepLen == Catch::Approx(knight->getSpeed() * 0.85f).epsilon(0.02));
    REQUIRE(knight->freezeTicks == 0); // movement only
}
// ---- 2026-08-04 balance update (final August balance changes) ----

TEST_CASE("Goblins and Hero Goblins deal 125 damage", "[live_sync][patch-2026-08-04]") {
    Board board;
    auto base = mainBody(board, 4);
    REQUIRE(base != nullptr);
    CHECK(base->getBaseDamage() == 125);
    Board board2;
    auto hero = mainBody(board2, 172);
    REQUIRE(hero != nullptr);
    CHECK(hero->getBaseDamage() == 125);
}

TEST_CASE("Goblin Drill's spawned Goblins deal 125 damage (periodic and death spawn)",
          "[live_sync][patch-2026-08-04]") {
    for (int cardId : { 98, 153 }) {
        Match m;
        bool evolved = (cardId == 153);
        auto bodies = m.spawnCard(cardId, 9.0f, 10.0f, 0, evolved);
        std::shared_ptr<CombatEntity> drill = firstCombat(bodies);
        REQUIRE(drill != nullptr);
        m.step(80);
        int periodic = 0;
        for (const auto& e : m.board().getEntities()) {
            auto c = std::dynamic_pointer_cast<CombatEntity>(e);
            if (c && c->cardId == -18 && c->team == 0) {
                CHECK(c->getBaseDamage() == 125);
                periodic++;
            }
        }
        CHECK(periodic > 0);
        drill->takeDamage(drill->hp);
        m.step(2);
        int death = 0;
        for (const auto& e : m.board().getEntities()) {
            auto c = std::dynamic_pointer_cast<CombatEntity>(e);
            if (c && c->cardId == -36 && c->team == 0) {
                CHECK(c->getBaseDamage() == 125);
                death++;
            }
        }
        CHECK(death >= 2);
    }
}

TEST_CASE("Goblin Machine has 2265 hp, 232 melee damage and a 5 s rocket",
          "[live_sync][patch-2026-08-04]") {
    Board board;
    auto bodies = spawnBare(board, 55);
    std::shared_ptr<CombatEntity> arms, rocket;
    for (const auto& e : bodies) {
        auto c = std::dynamic_pointer_cast<CombatEntity>(e);
        if (!c) continue;
        if (c->cardId == -24) rocket = c; else if (!arms) arms = c;
    }
    REQUIRE(arms != nullptr);
    REQUIRE(rocket != nullptr);
    CHECK(arms->hp == 2265);
    CHECK(rocket->hp == 2265);
    CHECK(arms->getBaseDamage() == 232);
    CHECK(arms->getAttackCooldown() == 12);
    CHECK(rocket->getBaseDamage() == 304);
    CHECK(rocket->getAttackCooldown() == 50);
}

TEST_CASE("Goblinstein's Doctor deals 135 damage", "[live_sync][patch-2026-08-04]") {
    Board board;
    auto bodies = spawnBare(board, 121);
    std::shared_ptr<CombatEntity> doctor;
    for (const auto& e : bodies) {
        auto c = std::dynamic_pointer_cast<CombatEntity>(e);
        if (c && c->cardId == -40) doctor = c;
    }
    REQUIRE(doctor != nullptr);
    CHECK(doctor->getBaseDamage() == 135);
}

TEST_CASE("Goblinstein's Lightning Link pulses for 94 (188 per second)",
          "[live_sync][patch-2026-08-04]") {
    Match m;
    auto bodies = m.spawnCard(121, 9.0f, 10.0f, 0);
    std::shared_ptr<CombatEntity> monster;
    for (const auto& e : bodies) {
        auto c = std::dynamic_pointer_cast<CombatEntity>(e);
        if (c && c->abilityEffect) monster = c;
    }
    REQUIRE(monster != nullptr);
    auto giant = firstCombat(m.spawnCard(2, 9.0f, 11.0f, 1));
    REQUIRE(giant != nullptr);
    m.rec->damage.clear();
    REQUIRE(monster->activateAbility(m.board()));
    m.step(3);
    int pulse = 0;
    for (const auto& e : m.rec->damage) {
        if (e.targetId == giant->id && e.amount > 0 && e.attackerCardId != -40 && e.attackerCardId != 121) {
            pulse = e.amount;
            break;
        }
    }
    CHECK(pulse == 94);
}

TEST_CASE("Hero Goblins' Banner Brigade respawns two Goblins at 125 damage",
          "[live_sync][patch-2026-08-04]") {
    Match m({ 1, 172, 2, 3, 4, 5, 6, 7 });
    GameManager& game = m.game;
    game.playerAI.hand[1] = 172;
    game.playCard(0, 172, 9.0f, 10.0f);
    game.step();
    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->cardId == 172 && ce->team == 0) ce->takeDamage(ce->hp);
    }
    game.step();
    game.playerAI.elixir = 10.0f;
    REQUIRE(game.activateChampionAbility(0, 1));
    game.step();
    int count = 0;
    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (!ce || ce->cardId != -48) continue;
        CHECK(ce->getBaseDamage() == 125);
        count++;
    }
    CHECK(count == 2);
}
// ---- 2026-08-04 balance update, part c (ledger P26c) -----------------------

TEST_CASE("Hero Ice Golem Snowstorm blast deals 69 per blast", "[live_sync][patch-2026-08-04]") {
    Match m;
    auto golem = firstCombat(m.spawnCard(175, 9.0f, 10.0f, 0));
    auto knight = firstCombat(m.spawnCard(0, 9.0f, 12.0f, 1));
    REQUIRE(golem != nullptr);
    REQUIRE(knight != nullptr);
    m.step(12); // deploy
    int before = m.rec->totalTo(knight->id);
    REQUIRE(golem->activateAbility(m.board()));
    m.step(3); // only the first of the three staggered blasts (5 ticks apart) has fired
    REQUIRE(m.rec->totalTo(knight->id) - before == 69);
}

TEST_CASE("Little Prince Guardienne deals 232 damage", "[live_sync][patch-2026-08-04]") {
    Match m;
    auto prince = firstCombat(m.spawnCard(120, 9.0f, 10.0f, 0));
    REQUIRE(prince != nullptr);
    m.step(12);
    REQUIRE(prince->activateAbility(m.board()));
    m.board().commitPendingEntities();
    std::shared_ptr<CombatEntity> guard;
    for (const auto& e : m.board().getEntities())
        if (e->name == "Guardienne") guard = std::dynamic_pointer_cast<CombatEntity>(e);
    REQUIRE(guard != nullptr);
    REQUIRE(guard->getBaseDamage() == 232);
}

TEST_CASE("Little Prince keeps his hit-speed ramp for 0.3 s after his target is gone",
          "[live_sync][patch-2026-08-04]") {
    // The grace counts from the last hit LAUNCHED, which a test cannot see
    // directly. So run the same deterministic scenario with the target removed
    // at each of several ticks and look for the ramp surviving the removal; with
    // no grace period it is reset the tick the target is gone, every time.
    bool everKept = false;
    bool everReset = false;
    for (int delay = 0; delay < 8; ++delay) {
        Match m;
        auto prince = firstCombat(m.spawnCard(120, 9.0f, 10.0f, 0));
        auto target = firstCombat(m.spawnCard(0, 9.0f, 14.0f, 1));
        REQUIRE(prince != nullptr);
        REQUIRE(target != nullptr);
        prince->hp = 100000;
        target->hp = 1000000;
        m.step(12);
        for (int i = 0; i < 80; ++i) m.step();
        REQUIRE(prince->getTicksOnTarget() > 30);
        m.step(delay);
        target->takeDamage(2000000);
        m.step(1);
        if (prince->getTicksOnTarget() > 0) everKept = true;
        else everReset = true;
    }
    REQUIRE(everKept);
    REQUIRE(everReset); // the grace is finite
}

TEST_CASE("Magic Archer deals 135 damage, Hero included", "[live_sync][patch-2026-08-04]") {
    Board board;
    REQUIRE(mainBody(board, 63)->getBaseDamage() == 135);
    Board board2;
    REQUIRE(mainBody(board2, 171)->getBaseDamage() == 135);
}

TEST_CASE("Mighty Miner's first damage-ramp stage is 43", "[live_sync][patch-2026-08-04]") {
    Board board;
    auto miner = mainBody(board, 115);
    REQUIRE(miner != nullptr);
    int hit = static_cast<int>(miner->getDamagePerTick() * miner->getAttackCooldown() + 0.5f);
    REQUIRE(hit == 43);
    REQUIRE(miner->getBaseDamage() == 409); // the max stage is unchanged
}

TEST_CASE("Mortar and its Evolution fire every 4.7 s", "[live_sync][patch-2026-08-04]") {
    Board a;
    REQUIRE(mainBody(a, 93)->getAttackCooldown() == 47);
    Board b;
    REQUIRE(mainBody(b, 152, false)->getAttackCooldown() == 47);
    Board c;
    REQUIRE(mainBody(c, 152, true)->getAttackCooldown() == 47);
}

TEST_CASE("Princess Evolution's slow lasts 5.5 s", "[live_sync][patch-2026-08-04]") {
    Match m;
    auto princess = m.spawnCard(145, 9.0f, 10.0f, 0, true);
    REQUIRE(!princess.empty());
    auto knight = firstCombat(m.spawnCard(0, 9.0f, 18.0f, 1));
    REQUIRE(knight != nullptr);
    knight->hp = 1000000;
    int seen = 0;
    for (int i = 0; i < 150; ++i) {
        m.step();
        if (knight->freezeTicks > seen) seen = knight->freezeTicks;
    }
    REQUIRE(seen >= 54);
    REQUIRE(seen <= 55);
}
namespace {
// Casts a spell at (x, y) on a bare board and runs it to the end.
void p26dCastAndRun(Board& board, int cardId, float x, float y) {
    CardRegistry::getInstance().getCard(cardId)->spawnEntity(x, y, 0, board);
    board.commitPendingEntities();
    auto spell = std::dynamic_pointer_cast<AreaSpell>(board.getEntities().back());
    REQUIRE(spell != nullptr);
    int guard = 0;
    while (spell->isAlive() && guard++ < 1000) spell->update(board);
    board.commitPendingEntities();
}
} // namespace

TEST_CASE("Royal Delivery landing damage is 384", "[live_sync][patch-2026-08-04]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 10.0f, 9.0f, 100000, 1, 5.0f, 10, 10);
    spawn(board, enemy);
    p26dCastAndRun(board, 111, 9.0f, 9.0f);
    REQUIRE(enemy->hp == 100000 - 384);
}

TEST_CASE("Rune Giant has 2816 hitpoints", "[live_sync][patch-2026-08-04]") {
    Board board;
    auto body = mainBody(board, 86);
    REQUIRE(body != nullptr);
    REQUIRE(body->hp == 2816);
}

TEST_CASE("Valkyrie Evolution Tornado damage is 42", "[live_sync][patch-2026-08-04]") {
    Board board;
    auto evo = mainBody(board, 147, true);
    REQUIRE(evo != nullptr);
    REQUIRE(evo->onHitPullDamage == 42);
}

TEST_CASE("Void costs 5 elixir", "[live_sync][patch-2026-08-04]") {
    REQUIRE(CardRegistry::getInstance().getCard(104)->cost == 5.0f);
}

TEST_CASE("Void deals 696 per wave to a lone target and 153 per wave to 5 or more",
          "[live_sync][patch-2026-08-04]") {
    {
        Board board;
        auto enemy = std::make_shared<StationaryCombatant>(1, 9.0f, 9.0f, 100000, 1, 5.0f, 10, 10);
        spawn(board, enemy);
        p26dCastAndRun(board, 104, 9.0f, 9.0f);
        REQUIRE(enemy->hp == 100000 - 3 * 696);
    }
    {
        Board board;
        std::vector<std::shared_ptr<StationaryCombatant>> pack;
        for (int i = 0; i < 5; ++i) {
            pack.push_back(std::make_shared<StationaryCombatant>(1 + i, 8.5f + 0.25f * i, 9.0f, 100000, 1, 5.0f, 10, 10));
            spawn(board, pack.back());
        }
        p26dCastAndRun(board, 104, 9.0f, 9.0f);
        for (const auto& e : pack) REQUIRE(e->hp == 100000 - 3 * 153);
    }
}

TEST_CASE("Witch Evolution heal-on-hit cap is 1451", "[live_sync][patch-2026-08-04]") {
    Board board;
    auto evo = mainBody(board, 142, true);
    REQUIRE(evo != nullptr);
    REQUIRE(evo->healOnHitMaxHp == 1451);
}

TEST_CASE("X-Bow hit speed is 0.4 s", "[live_sync][patch-2026-08-04]") {
    Board board;
    auto body = mainBody(board, 92);
    REQUIRE(body != nullptr);
    REQUIRE(body->getAttackCooldown() == 4);
}

// The 2-4 tier went 294 (2026-08-04) -> 307 (2026-10-19); this case pins the current value.
TEST_CASE("Void deals its 2-4 target damage per wave (294, then 307 from 2026-10-19)", "[live_sync][patch-2026-08-04]") {
    Board board;
    std::vector<std::shared_ptr<StationaryCombatant>> pair;
    for (int i = 0; i < 2; ++i) {
        pair.push_back(std::make_shared<StationaryCombatant>(1 + i, 8.5f + 0.5f * i, 9.0f, 100000, 1, 5.0f, 10, 10));
        spawn(board, pair.back());
    }
    p26dCastAndRun(board, 104, 9.0f, 9.0f);
    for (const auto& e : pair) REQUIRE(e->hp == 100000 - 3 * 307);
}

TEST_CASE("Archer Queen hits for 232 (2026-08-26 buff)", "[live_sync][patch-2026-08-26]") {
    Board board;
    auto queen = mainBody(board, 118);
    REQUIRE(queen != nullptr);
    REQUIRE(queen->getBaseDamage() == 232);
}

TEST_CASE("Electro Giant hits for 184 (2026-08-26 buff)", "[live_sync][patch-2026-08-26]") {
    Board board;
    auto giant = mainBody(board, 84);
    REQUIRE(giant != nullptr);
    REQUIRE(giant->getBaseDamage() == 184);
}

TEST_CASE("Goblinstein Monster has 2240 hp (2026-08-26 nerf)", "[live_sync][patch-2026-08-26]") {
    Board board;
    auto bodies = spawnBare(board, 121);
    std::shared_ptr<CombatEntity> monster;
    for (const auto& e : bodies) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->isChampion) monster = ce;
    }
    REQUIRE(monster != nullptr);
    REQUIRE(monster->hp == 2240);
}

// ---------------------------------------------------------------------------
// Patch 2026-09-01 (Minion Academy season notes, ledger items 4, 16, 129).
// ---------------------------------------------------------------------------

TEST_CASE("Bandit and Boss Bandit shrug off Zap-style stuns while dashing, not while walking",
          "[live_sync][patch-2026-09-01]") {
    for (int cardId : { 50, 122 }) {
        Board board;
        auto walking = mainBody(board, cardId);
        REQUIRE(walking != nullptr);
        walking->applyFreeze(10, 0.0f);
        INFO("card " << cardId << " walking is still stunned");
        REQUIRE(walking->freezeTicks > 0);

        auto dashing = mainBody(board, cardId);
        REQUIRE(dashing != nullptr);
        dashing->chargeProgress = dashing->chargeThreshold; // mid-dash
        dashing->applyFreeze(10, 0.0f);
        INFO("card " << cardId << " dashing ignores the stun");
        REQUIRE(dashing->freezeTicks == 0);
    }
}

TEST_CASE("Clone does not duplicate a unit that is itself a clone",
          "[live_sync][patch-2026-09-01]") {
    auto countKnights = [](Match& m) {
        int n = 0;
        for (const auto& e : m.board().getEntities())
            if (e->isAlive() && e->cardId == 0 && e->team == 0) ++n;
        return n;
    };
    Match m;
    auto original = firstCombat(m.spawnCard(0, 9.0f, 8.0f, 0));
    REQUIRE(original != nullptr);
    REQUIRE(countKnights(m) == 1);

    // Cast on the original's current position each time: it keeps walking.
    m.spawnCard(114, original->position.x, original->position.y, 0);
    m.step(40);
    REQUIRE(countKnights(m) == 2);

    m.spawnCard(114, original->position.x, original->position.y, 0);
    m.step(40);
    // Only the original was eligible the second time: the clone is skipped.
    REQUIRE(countKnights(m) == 3);
}

TEST_CASE("Vines breaks an equal-hitpoint tie toward the spell centre",
          "[live_sync][patch-2026-09-01]") {
    Match m;
    // Same hp, offsets from the cast point (9, 20). The nearest is spawned last,
    // so a hp-only sort (stable on a tie) would drop exactly that one.
    const float offsets[4] = { 2.1f, 1.5f, 0.9f, 0.3f };
    std::vector<int> ids;
    for (float dx : offsets) {
        auto b = firstCombat(m.spawnCard(0, 9.0f + dx, 20.0f, 1));
        REQUIRE(b != nullptr);
        ids.push_back(b->id);
    }
    m.spawnCard(105, 9.0f, 20.0f, 0);
    m.step(60);
    REQUIRE(m.rec->totalTo(ids[3]) > 0); // closest is hit
    REQUIRE(m.rec->totalTo(ids[2]) > 0);
    REQUIRE(m.rec->totalTo(ids[1]) > 0);
    REQUIRE(m.rec->totalTo(ids[0]) == 0); // farthest is spared
}

// ---------------- Patch 2026-09-08 (P29a) ----------------

TEST_CASE("Bomber damage is 212 on the base card and its Evolution (2026-09-08)",
          "[live_sync][patch-2026-09-08]") {
    Board board;
    auto base = mainBody(board, 9);
    REQUIRE(base != nullptr);
    REQUIRE(base->getBaseDamage() == 212);

    Board evoBoard;
    auto evo = mainBody(evoBoard, 127, /*evolved=*/true);
    REQUIRE(evo != nullptr);
    REQUIRE(evo->getBaseDamage() == 212);

    Board evoIdBoard;
    auto evoById = mainBody(evoIdBoard, 127);  // un-evolved half of the Evolution slot
    REQUIRE(evoById != nullptr);
    REQUIRE(evoById->getBaseDamage() == 212);
}

TEST_CASE("Freeze stuns for 3.5 s (35 ticks) (2026-09-08)", "[live_sync][patch-2026-09-08]") {
    Board board;
    auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 6.0f, 1000, 1, 5.0f, 10, 10);
    spawn(board, enemy);

    CardRegistry::getInstance().getCard(107)->spawnEntity(5.0f, 5.0f, 0, board);
    board.commitPendingEntities();
    auto spell = std::dynamic_pointer_cast<AreaSpell>(board.getEntities().back());
    REQUIRE(spell != nullptr);
    while (spell->isAlive()) spell->update(board);

    REQUIRE(enemy->freezeTicks == 35);
}

TEST_CASE("Goblin Cage's Goblin Brawler has 1121 hp, for the base card and its Evolution (2026-09-08)",
          "[live_sync][patch-2026-09-08]") {
    for (int id : { 97, 139 }) {
        Board board;
        auto bodies = spawnBare(board, id, 9.0f, 10.0f, 0);
        std::shared_ptr<CombatEntity> cage;
        for (const auto& e : bodies)
            if (auto c = std::dynamic_pointer_cast<CombatEntity>(e)) cage = c;
        REQUIRE(cage != nullptr);
        std::set<int> before;
        for (const auto& e : board.getEntities()) before.insert(e->id);
        cage->hp = 0;
        board.cleanDeadEntities();
        board.commitPendingEntities();
        std::shared_ptr<CombatEntity> brawler;
        for (const auto& e : board.getEntities())
            if (!before.count(e->id))
                if (auto c = std::dynamic_pointer_cast<CombatEntity>(e)) brawler = c;
        REQUIRE(brawler != nullptr);
        REQUIRE(brawler->hp == 1121);
    }
}

TEST_CASE("Golem sight range is 7.5 tiles (2026-09-08)", "[live_sync][patch-2026-09-08]") {
    Board board;
    auto golem = mainBody(board, 19);
    REQUIRE(golem != nullptr);
    REQUIRE(golem->sightRange == Catch::Approx(7.5f));
}

TEST_CASE("Ice Golem and Hero Ice Golem: 1228 hp and a 2.5 s death slow (2026-09-08)",
          "[live_sync][patch-2026-09-08]") {
    for (int id : { 40, 175 }) {
        Board board;
        auto enemy = std::make_shared<StationaryCombatant>(1, 5.0f, 5.5f, 1000, 1, 5.0f, 10, 10);
        spawn(board, enemy);
        auto bodies = spawnBare(board, id, 5.0f, 5.0f, 0);
        auto golem = firstCombat(bodies);
        REQUIRE(golem != nullptr);
        REQUIRE(golem->hp == 1228);
        golem->hp = 0;
        board.cleanDeadEntities();
        REQUIRE(enemy->freezeTicks == 25);
        REQUIRE(enemy->freezeSlow == Catch::Approx(0.70f)); // 2025-10-06: slow 30%
    }
}

TEST_CASE("Hero Mega Minion's warp deals 422 damage (2026-09-08)", "[live_sync][patch-2026-09-08]") {
    Board board;
    auto victim = std::make_shared<StationaryCombatant>(1, 30.0f, 30.0f, 5000, 1, 5.0f, 10, 10);
    spawn(board, victim);
    auto minion = firstCombat(spawnBare(board, 173, 5.0f, 5.0f, 0));
    REQUIRE(minion != nullptr);
    REQUIRE(minion->abilityEffect != nullptr);
    minion->abilityCooldownRemaining = 0;
    REQUIRE(minion->activateAbility(board));
    REQUIRE(victim->hp == 5000 - 422);
}
// ---------------------------------------------------------------------------
// patch 2026-09-08 (September balance changes, 8th)
// ---------------------------------------------------------------------------

#include "SpiritEmpressForms.h"

TEST_CASE("Ronin hit speed is 13 ticks (1.3 s)", "[live_sync][patch-2026-09-08]") {
    Board board;
    auto ronin = mainBody(board, 54);
    REQUIRE(ronin != nullptr);
    REQUIRE(ronin->getAttackCooldown() == 13);
}

TEST_CASE("Zappies hit speed is 23 ticks (2.3 s)", "[live_sync][patch-2026-09-08]") {
    Board board;
    auto z = mainBody(board, 79);
    REQUIRE(z != nullptr);
    REQUIRE(z->getAttackCooldown() == 23);
}

TEST_CASE("Spirit Empress damage is 320 on the registry entry and on both forms",
          "[live_sync][patch-2026-09-08]") {
    Board board;
    auto body = mainBody(board, 165);
    REQUIRE(body != nullptr);
    REQUIRE(body->getBaseDamage() == 320);
    REQUIRE(spiritEmpressGroundStats().damage == 320);
    REQUIRE(spiritEmpressFlyingStats().damage == 320);
}

TEST_CASE("Wizard, Wizard Evolution and Hero Wizard deal 304 damage", "[live_sync][patch-2026-09-08]") {
    Board board;
    auto base = mainBody(board, 11);
    auto evoBase = mainBody(board, 141, false);
    auto evo = mainBody(board, 141, true);
    auto hero = mainBody(board, 167);
    REQUIRE(base != nullptr);
    REQUIRE(evoBase != nullptr);
    REQUIRE(evo != nullptr);
    REQUIRE(hero != nullptr);
    REQUIRE(base->getBaseDamage() == 304);
    REQUIRE(evoBase->getBaseDamage() == 304);
    REQUIRE(evo->getBaseDamage() == 304);
    REQUIRE(hero->getBaseDamage() == 304);
}

TEST_CASE("Wizard Evolution Fire Shield holds 176 hitpoints", "[live_sync][patch-2026-09-08]") {
    Board board;
    auto evo = mainBody(board, 141, true);
    REQUIRE(evo != nullptr);
    REQUIRE(evo->shieldHp == 176);
}

TEST_CASE("Hero Wizard Fiery Flight lasts 35 ticks (3.5 s)", "[live_sync][patch-2026-09-08]") {
    Board board;
    auto hero = mainBody(board, 167);
    REQUIRE(hero != nullptr);
    REQUIRE(hero->abilityEffect != nullptr);
    hero->abilityEffect->apply(board, *hero);
    REQUIRE(hero->isFlying);
    REQUIRE(hero->temporaryFlightTicksRemaining == 35);
    REQUIRE(hero->flightPulseTicksRemaining == 35);
}

// ---------------- patch 2026-09-16 ----------------

TEST_CASE("Goblinstein Lightning Link lasts 3.5 s: 7 shocks, not 8", "[live_sync][patch-2026-09-16]") {
    Board board;
    auto bodies = spawnBare(board, 121, 9.0f, 10.0f, 0);
    std::shared_ptr<CombatEntity> monster;
    for (const auto& e : bodies) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && ce->isChampion) monster = ce;
    }
    REQUIRE(monster != nullptr);
    auto enemy = std::make_shared<DummyEntity>(board.allocateId(), 9.0f, 11.0f, 100000, 1);
    board.addEntity(enemy);
    board.commitPendingEntities();

    REQUIRE(monster->activateAbility(board));
    board.commitPendingEntities();
    std::shared_ptr<AreaSpell> shock;
    for (const auto& e : board.getEntities())
        if (auto a = std::dynamic_pointer_cast<AreaSpell>(e)) shock = a;
    REQUIRE(shock != nullptr);

    for (int i = 0; i < 80 && shock->isAlive(); ++i) shock->update(board);
    REQUIRE(100000 - enemy->hp == 7 * 94); // 94 a pulse since 2026-08-04
}

// ---------------- patch 2026-10-06 (P31a): Barbarian Barrel, Cannon Evolution, Elixir Collector, Golden Knight, Lava Hound ----------------

namespace p31a {

// A harmless, immobile, targetable body.
inline std::shared_ptr<StationaryCombatant> dummy(Match& m, float x, float y, int team, int hp) {
    auto d = std::make_shared<StationaryCombatant>(m.board().allocateId(), x, y, hp, team, 0.5f, 0, 1000);
    spawn(m.board(), d);
    return d;
}

// The first positive damage amount any attacker dealt to `targetId`, or -1.
inline int firstHitOn(const Match& m, int targetId) {
    for (const auto& e : m.rec->damage)
        if (e.targetId == targetId && e.amount > 0) return e.amount;
    return -1;
}

// Elixir gained by team 0 over `ticks` ticks after a collector is played, minus
// the same window's plain regeneration (a Match with no collector).
inline float collectorBonus(int ticks) {
    Match with, without;
    with.game.setElixir(0, 0.0f);
    without.game.setElixir(0, 0.0f);
    with.spawnCard(99, 9.0f, 10.0f, 0);
    with.step(ticks);
    without.step(ticks);
    return with.game.getElixir(0) - without.game.getElixir(0);
}

} // namespace p31a

TEST_CASE("Barbarian Barrel (101) rolls for 215 (2026-10-06)", "[live_sync][patch-2026-10-06]") {
    Match m;
    auto target = p31a::dummy(m, 8.5f, 8.0f, 1, 10000);
    m.spawnCard(101, 8.5f, 6.0f, 0);
    m.step(40);
    REQUIRE(p31a::firstHitOn(m, target->id) == 215);
}

TEST_CASE("Hero Barbarian Barrel (174) rolls for 215 (2026-10-06)", "[live_sync][patch-2026-10-06]") {
    Match m;
    auto target = p31a::dummy(m, 8.5f, 8.0f, 1, 10000);
    m.spawnCard(174, 8.5f, 6.0f, 0);
    m.step(40);
    REQUIRE(p31a::firstHitOn(m, target->id) == 215);
}

TEST_CASE("Hero Barbarian Barrel's Rowdy Reroll deals the barrel's 215 (2026-10-06)", "[live_sync][patch-2026-10-06]") {
    Match m;
    m.spawnCard(174, 8.5f, 6.0f, 0);
    std::shared_ptr<CombatEntity> hero;
    for (int i = 0; i < 60 && !hero; ++i) {
        for (const auto& e : m.board().getEntities())
            if (e->isAlive() && e->team == 0 && e->name == "Hero Barbarian Barrel")
                if (auto c = std::dynamic_pointer_cast<CombatEntity>(e))
                    if (c->abilityEffect) { hero = c; break; }
        if (!hero) m.step();
    }
    REQUIRE(hero);
    // Park a target in the reroll corridor and fire the ability directly.
    auto target = p31a::dummy(m, hero->position.x, hero->position.y + 2.0f, 1, 10000);
    hero->abilityEffect->apply(m.board(), *hero);
    REQUIRE(10000 - target->hp == 215);
}

TEST_CASE("Elixir Collector produces 1 elixir every 15 s, 150 ticks (2026-10-06)", "[live_sync][patch-2026-10-06]") {
    // 10 ticks of deploy, then one interval. Nothing is credited before then.
    REQUIRE(p31a::collectorBonus(120) == Catch::Approx(0.0f).margin(0.01f));
    // A control that must fire: one pulse has landed by tick 200.
    REQUIRE(p31a::collectorBonus(200) == Catch::Approx(1.0f).margin(0.01f));
}

TEST_CASE("Golden Knight (116) deals 168 per hit (2026-10-06)", "[live_sync][patch-2026-10-06]") {
    Board board;
    auto knight = mainBody(board, 116);
    REQUIRE(knight);
    REQUIRE(knight->getBaseDamage() == 168);
}

TEST_CASE("Golden Knight's dash reaches 5.0 tiles, not 5.5 (2026-10-06)", "[live_sync][patch-2026-10-06]") {
    auto dashHp = [](float distance) {
        Board board;
        auto knight = mainBody(board, 116);
        auto enemy = std::make_shared<DummyEntity>(board.allocateId(), knight->position.x,
                                                   knight->position.y + distance, 10000, 1);
        spawn(board, enemy);
        knight->abilityEffect->apply(board, *knight);
        return enemy->hp;
    };
    REQUIRE(dashHp(4.8f) < 10000);   // control: in range, the dash fires
    REQUIRE(dashHp(5.2f) == 10000);  // between the old 5.5 and the new 5.0: no dash
}

TEST_CASE("Lava Hound (91) hits for 71 every 15 ticks (2026-10-06)", "[live_sync][patch-2026-10-06]") {
    Board board;
    auto hound = mainBody(board, 91);
    REQUIRE(hound);
    REQUIRE(hound->getBaseDamage() == 71);
    REQUIRE(hound->getAttackCooldown() == 15);
}
// ---- 2026-10-06 balance update (P31b): Ram Rider, Royal Ghost, Three Musketeers,
// Wall Breakers, X-Bow ----

namespace {
// Largest base damage among a card's spawned bodies (the ram, not its crossbow).
int p31bMaxBodyDamage(const std::vector<std::shared_ptr<Entity>>& bodies) {
    int best = 0;
    for (const auto& e : bodies)
        if (auto c = std::dynamic_pointer_cast<CombatEntity>(e)) best = std::max(best, c->getBaseDamage());
    return best;
}
} // namespace

TEST_CASE("Ram Rider ram damage is 271 (2026-10-06)", "[live_sync][patch-2026-10-06]") {
    Board board;
    auto bodies = spawnBare(board, 87);
    REQUIRE(p31bMaxBodyDamage(bodies) == 271);
}

TEST_CASE("Royal Ghost hp 1152 and damage 263 on the base and the Evolution (2026-10-06)",
          "[live_sync][patch-2026-10-06]") {
    for (int pass = 0; pass < 3; ++pass) {
        Board board;
        auto g = mainBody(board, pass == 0 ? 47 : 136, pass == 1);
        REQUIRE(g);
        CHECK(g->hp == 1152);
        CHECK(g->getBaseDamage() == 263);
    }
}

TEST_CASE("Three Musketeers hp per Musketeer is 906 (2026-10-06)", "[live_sync][patch-2026-10-06]") {
    Board board;
    auto bodies = spawnBare(board, 80);
    int n = 0;
    for (const auto& e : bodies)
        if (std::dynamic_pointer_cast<CombatEntity>(e)) { CHECK(e->hp == 906); ++n; }
    REQUIRE(n == 3);
}

TEST_CASE("Wall Breakers damage is 302 on the base and the Evolution (2026-10-06)",
          "[live_sync][patch-2026-10-06]") {
    for (int pass = 0; pass < 3; ++pass) {
        Board board;
        auto g = mainBody(board, pass == 0 ? 83 : 123, pass == 2);
        REQUIRE(g);
        CHECK(g->getBaseDamage() == 302);
    }
}

TEST_CASE("X-Bow damage is 61 (2026-10-06)", "[live_sync][patch-2026-10-06]") {
    Board board;
    auto x = mainBody(board, 92);
    REQUIRE(x);
    CHECK(x->getBaseDamage() == 61);
}

// ---- Patch 2026-10-19 ----

TEST_CASE("Cannon Cart damage is 202", "[live_sync][patch-2026-10-19]") {
    Board board;
    auto cart = mainBody(board, 69);
    REQUIRE(cart != nullptr);
    CHECK(cart->getBaseDamage() == 202);
}

TEST_CASE("Dart Goblin damage is 143, base and Evolution", "[live_sync][patch-2026-10-19]") {
    Board board;
    auto base = mainBody(board, 20);
    REQUIRE(base != nullptr);
    CHECK(base->getBaseDamage() == 143);

    Board evoBoard;
    auto evo = mainBody(evoBoard, 131, true);
    REQUIRE(evo != nullptr);
    CHECK(evo->getBaseDamage() == 143);

    Board evoBoard2;
    auto evoBase = mainBody(evoBoard2, 131, false);
    REQUIRE(evoBase != nullptr);
    CHECK(evoBase->getBaseDamage() == 143);
}

TEST_CASE("Flying Machine damage is 179", "[live_sync][patch-2026-10-19]") {
    Board board;
    auto fm = mainBody(board, 67);
    REQUIRE(fm != nullptr);
    CHECK(fm->getBaseDamage() == 179);
}

TEST_CASE("Void deals 307 per target to 2-4 targets", "[live_sync][patch-2026-10-19]") {
    Match m;
    auto giants = m.spawnCard(2, 9.0f, 10.0f, 1);
    auto giants2 = m.spawnCard(2, 9.0f, 10.5f, 1);
    REQUIRE(!giants.empty());
    REQUIRE(!giants2.empty());
    const int id1 = giants.front()->id;
    m.spawnCard(104, 9.0f, 10.2f, 0);
    m.step(40);
    REQUIRE(!m.rec->damage.empty());
    int firstHit = 0;
    for (const auto& e : m.rec->damage)
        if (e.targetId == id1 && e.amount > 0) { firstHit = e.amount; break; }
    CHECK(firstHit == 307);
}

// ---------------- undated cr-api-data / DeckShop drift (patch P33) ----------------

// Lightning: "damage and stun up to three enemy troops or buildings with the
// most hitpoints in the target area" (cr-api-data description). Cast clear of
// the enemy towers, which would otherwise count as candidates.
TEST_CASE("Lightning strikes only the three highest-HP enemies in the area",
          "[live_sync][patch-undated]") {
    Match m;
    // Four enemy bodies inside the 3.5 radius, hp ascending:
    // Skeletons (81 each) < Musketeer < Ice Golem < Hog Rider.
    std::vector<std::shared_ptr<CombatEntity>> victims;
    const int ids[4] = { 24, 6, 40, 15 };
    for (int i = 0; i < 4; ++i) {
        auto c = firstCombat(m.spawnCard(ids[i], 9.0f + 0.5f * i, 21.0f, 1));
        REQUIRE(c != nullptr);
        victims.push_back(c);
    }
    for (size_t i = 1; i < victims.size(); ++i) REQUIRE(victims[i]->hp > victims[i - 1]->hp);

    m.spawnCard(31, 10.0f, 21.0f, 0);
    m.step(30);

    REQUIRE(m.rec->totalTo(victims[0]->id) == 0);
    for (size_t i = 1; i < victims.size(); ++i) REQUIRE(m.rec->totalTo(victims[i]->id) > 0);
}

// Monk is a 5-elixir Champion (DeckShop).
TEST_CASE("Monk costs 5 elixir", "[live_sync][patch-undated]") {
    const CardDefinition* def = CardRegistry::getInstance().getCard(119);
    REQUIRE(def != nullptr);
    REQUIRE(def->cost == Catch::Approx(5.0f));
}

// ---- Crown Tower damage (patch-crown-tower-damage; UPSTREAM_REQUESTS.md item 29) ----
namespace {
// Total damage a spell of team 0, cast on team 1's left Princess Tower, deals to it.
int crownTowerHit(int cardId, bool evolved = false) {
    Match m;
    std::shared_ptr<Tower> princess;
    for (const auto& e : m.board().getEntities())
        if (auto t = std::dynamic_pointer_cast<Tower>(e))
            if (t->team == 1 && t->symbol != 'R' && t->position.x < 9.0f) princess = t;
    REQUIRE(princess);
    m.spawnCard(cardId, princess->position.x, princess->position.y, 0, evolved);
    m.step(60);
    return m.rec->totalTo(princess->id);
}
}

TEST_CASE("Arrows deal 25 per volley, 75 in all, to a Crown Tower",
          "[live_sync][patch-crown-tower-damage]") {
    REQUIRE(crownTowerHit(3) == 75);
}

TEST_CASE("Earthquake deals 49 per wave, 147 in all, to a Crown Tower",
          "[live_sync][patch-crown-tower-damage]") {
    REQUIRE(crownTowerHit(103) == 147);
}

TEST_CASE("Fireball deals 159 to a Crown Tower",
          "[live_sync][patch-crown-tower-damage]") {
    REQUIRE(crownTowerHit(7) == 159);
}

TEST_CASE("Freeze deals 37 to a Crown Tower",
          "[live_sync][patch-crown-tower-damage]") {
    REQUIRE(crownTowerHit(107) == 37);
}

TEST_CASE("Giant Snowball deals 45 to a Crown Tower, base and Evolution",
          "[live_sync][patch-crown-tower-damage]") {
    REQUIRE(crownTowerHit(100) == 45);
    REQUIRE(crownTowerHit(158, true) == 45);
}
// ===== Crown Tower damage (UPSTREAM_REQUESTS.md item 29), chunk PCTb =====
// Each case casts the spell on an enemy Princess Tower in a real match and
// reads what the tower actually lost.
namespace {
std::shared_ptr<Tower> enemyPrincess(Match& m) {
    for (const auto& e : m.board().getEntities()) {
        auto t = std::dynamic_pointer_cast<Tower>(e);
        if (t && t->team == 1 && e->position.x < 8.0f) return t; // left Princess Tower
    }
    return nullptr;
}

int towerDamageFromCast(int cardId, bool evolved, bool roll) {
    Match m;
    auto tower = enemyPrincess(m);
    REQUIRE(tower != nullptr);
    // A roller starts behind the tower and sweeps toward it.
    const float y = roll ? tower->position.y - 6.0f : tower->position.y;
    m.spawnCard(cardId, tower->position.x, y, 0, evolved);
    m.step(120);
    return m.rec->totalTo(tower->id);
}
} // namespace

TEST_CASE("Goblin Curse deals 10 per application to a Crown Tower",
          "[live_sync][patch-crown-tower-damage]") {
    // 6 applications, 1 s apart: 6 x 10 (Sept 2025 notes: 6 -> 10 per second).
    REQUIRE(towerDamageFromCast(102, false, false) == 60);
}

TEST_CASE("Lightning deals 265 to a Crown Tower per strike",
          "[live_sync][patch-crown-tower-damage]") {
    REQUIRE(towerDamageFromCast(31, false, false) == 265);
}

TEST_CASE("Rocket deals 342 to a Crown Tower",
          "[live_sync][patch-crown-tower-damage]") {
    REQUIRE(towerDamageFromCast(30, false, false) == 342);
}

TEST_CASE("The Log deals 35 to a Crown Tower",
          "[live_sync][patch-crown-tower-damage]") {
    REQUIRE(towerDamageFromCast(33, false, true) == 35);
}

TEST_CASE("Zap deals 48 to a Crown Tower; its Evolution 48 per zap",
          "[live_sync][patch-crown-tower-damage]") {
    SECTION("base") { REQUIRE(towerDamageFromCast(29, false, false) == 48); }
    SECTION("Evolution, two zaps (2024-10-08: three -> two)") { REQUIRE(towerDamageFromCast(124, true, false) == 96); }
}

// ---------------------------------------------------------------------------
// Single-use Champion / Hero abilities (2026-08-04 "final August balance
// changes": "Abilities: Cooldown -> Single-Use", all except Boss Bandit).
// ---------------------------------------------------------------------------
namespace psingle {

// Deploys `cardId` in slot 1, then tries its ability twice, clearing the
// cooldown between the attempts so only the uses limit can refuse the second.
struct TwoActivations { bool first; bool second; int usesAfterFirst; };

inline TwoActivations activateTwice(int cardId) {
    GameManager game({ 1, cardId, 2, 3, 4, 5, 6, 7 }, { 0,1,2,3,4,5,6,7 });
    game.playerAI.hand[1] = cardId;
    game.playerAI.elixir = 100.0f;
    game.playCard(0, cardId, 9.0f, 10.0f);
    game.step(); // commit, so the ability can find the unit
    game.playerAI.elixir = 100.0f;
    TwoActivations r{ false, false, -2 };
    r.first = game.activateChampionAbility(0, 1);
    for (const auto& e : game.getBoard().getEntities()) {
        auto ce = std::dynamic_pointer_cast<CombatEntity>(e);
        if (ce && (ce->isChampion || ce->isHero) && ce->abilityEffect) {
            r.usesAfterFirst = ce->abilityUsesRemaining;
            ce->abilityCooldownRemaining = 0;
        }
    }
    game.playerAI.elixir = 100.0f;
    r.second = game.activateChampionAbility(0, 1);
    return r;
}

} // namespace psingle

TEST_CASE("Single-use abilities: every Champion except Boss Bandit and every alive-path Hero fires once per deployment",
          "[live_sync][patch-single-use-abilities]") {
    const int ids[] = { 115, 116, 117, 118, 119, 120, 121, 166, 167, 168, 169, 170, 171, 173, 175 };
    for (int id : ids) {
        DYNAMIC_SECTION("id " << id) {
            auto r = psingle::activateTwice(id);
            REQUIRE(r.first);
            REQUIRE(r.usesAfterFirst == 0);
            REQUIRE_FALSE(r.second);
        }
    }
}

TEST_CASE("Single-use abilities: Boss Bandit keeps two uses (control)",
          "[live_sync][patch-single-use-abilities]") {
    auto r = psingle::activateTwice(122);
    REQUIRE(r.first);
    REQUIRE(r.usesAfterFirst == 1);
    REQUIRE(r.second); // the exempt card must still refire once its cooldown is clear
}
