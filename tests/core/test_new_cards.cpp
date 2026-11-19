// Cards, Heroes and Evolutions added in the 2026-11 live sync (docs/LIVE_SYNC.md).
// One tag per entry: [new_card][<key>]. Each case fails without its commit.
#include <catch_amalgamated.hpp>
#include "live_sync_test_helpers.h"
#include "ClashEnv.h"
#include <random>

// ---------------- Hero Balloon (176) ----------------

TEST_CASE("Hero Balloon (176) is registered as a Hero with the base Balloon's body",
          "[new_card][hero_balloon]") {
    const CardDefinition* def = CardRegistry::getInstance().getCard(176);
    REQUIRE(def != nullptr);
    CHECK(def->name == "Hero Balloon");
    CHECK(def->cost == Catch::Approx(5.0f));
    CHECK(def->isHero);
    CHECK_FALSE(def->isChampion);
    CHECK_FALSE(def->isSpell);
    CHECK(def->isFlying);

    Board board;
    auto body = live_sync::mainBody(board, 176);
    REQUIRE(body != nullptr);
    CHECK(body->hp == 1676);
    CHECK(body->getBaseDamage() == 640);
    CHECK(body->getAttackCooldown() == 20);
    CHECK(body->getAttackRange() == Catch::Approx(0.1f));
    CHECK(body->sightRange == Catch::Approx(7.7f));
    CHECK(body->firstHitTicks == 2);
    CHECK(body->isFlying);
    CHECK(body->isHero);
}

TEST_CASE("Hero Balloon (176) matches base Balloon (45) stat for stat",
          "[new_card][hero_balloon]") {
    Board a, b;
    auto base = live_sync::mainBody(a, 45);
    auto hero = live_sync::mainBody(b, 176);
    REQUIRE(base != nullptr);
    REQUIRE(hero != nullptr);
    CHECK(hero->hp == base->hp);
    CHECK(hero->getBaseDamage() == base->getBaseDamage());
    CHECK(hero->getAttackCooldown() == base->getAttackCooldown());
}

TEST_CASE("Hero Balloon (176) is flagged ability-pending and is never dealt in random decks",
          "[new_card][hero_balloon]") {
    const CardDefinition* def = CardRegistry::getInstance().getCard(176);
    REQUIRE(def != nullptr);
    CHECK(def->abilityPending);
    // Control: a Hero with a modelled ability is not flagged.
    CHECK_FALSE(CardRegistry::getInstance().getCard(175)->abilityPending);

    // Control for the sampler: another Hero does get dealt over the same draws.
    std::mt19937 rng(12345);
    int withBalloon = 0, withIceGolem = 0;
    for (int i = 0; i < 4000; ++i) {
        for (int id : sampleRandomDeck(rng)) {
            if (id == 176) ++withBalloon;
            if (id == 175) ++withIceGolem;
        }
    }
    CHECK(withIceGolem > 0);
    CHECK(withBalloon == 0);
}

// ---------------- Hero Dark Prince (177) ----------------

TEST_CASE("Hero Dark Prince (177) is registered as a Hero with the base Dark Prince's body",
          "[new_card][hero_dark_prince]") {
    const CardDefinition* def = CardRegistry::getInstance().getCard(177);
    REQUIRE(def != nullptr);
    CHECK(def->name == "Hero Dark Prince");
    CHECK(def->cost == Catch::Approx(4.0f));
    CHECK(def->isHero);
    CHECK_FALSE(def->isChampion);
    CHECK_FALSE(def->isSpell);

    Board board;
    auto body = live_sync::mainBody(board, 177);
    REQUIRE(body != nullptr);
    CHECK(body->hp == 1200);
    CHECK(body->shieldHp == 240);
    CHECK(body->getBaseDamage() == 266);
    CHECK(body->getAttackCooldown() == 14);
    CHECK(body->getAttackRange() == Catch::Approx(1.2f));
    CHECK(body->sightRange == Catch::Approx(5.5f));
    CHECK(body->deployTicksRemaining == DEPLOY_TIME_TICKS);
    CHECK(body->isHero);
}

TEST_CASE("Hero Dark Prince (177) starts its first charge after 2.5 tiles, the base card after 3.0",
          "[new_card][hero_dark_prince]") {
    Board a, b;
    auto hero = live_sync::mainBody(a, 177);
    auto base = live_sync::mainBody(b, 46);
    REQUIRE(hero != nullptr);
    REQUIRE(base != nullptr);
    CHECK(hero->chargeThreshold == Catch::Approx(2.5f));
    CHECK(hero->chargeMultiplier == Catch::Approx(2.0f));
    CHECK(hero->chargeSpeedMultiplier == Catch::Approx(2.0f));
    // Control: the two thresholds must differ.
    CHECK(base->chargeThreshold == Catch::Approx(3.0f));
}

TEST_CASE("Hero Dark Prince (177) is Hero-slot legal but flagged ability-pending",
          "[new_card][hero_dark_prince]") {
    CHECK(validateDeckSlots({1, 177, 2, 3, 4, 5, 6, 7}).empty());
    CHECK(validateDeckSlots({1, 2, 177, 3, 4, 5, 6, 7}).empty());
    CHECK_FALSE(validateDeckSlots({177, 1, 2, 3, 4, 5, 6, 7}).empty()); // slot 0 is not a Hero slot
    CHECK(CardRegistry::getInstance().getCard(177)->abilityPending);

    std::mt19937 rng(777);
    int withDarkPrince = 0, withIceGolem = 0;
    for (int i = 0; i < 4000; ++i) {
        for (int id : sampleRandomDeck(rng)) {
            if (id == 177) ++withDarkPrince;
            if (id == 175) ++withIceGolem;
        }
    }
    CHECK(withIceGolem > 0); // control: another Hero is dealt over the same draws
    CHECK(withDarkPrince == 0);
}

TEST_CASE("Hero Dark Prince (177) has no ability yet: activating it fails and spends nothing",
          "[new_card][hero_dark_prince]") {
    // Delete this case when Destructive Dismount lands.
    std::vector<int> deck = {1, 177, 2, 3, 4, 5, 6, 7};
    GameManager game(deck, deck);
    game.seed(1);
    game.reset();
    game.setElixir(0, 10.0f);
    // Deploy it from slot 1 if it is in hand; the ability must be refused either way.
    CHECK_FALSE(game.activateChampionAbility(0, 1));
    CHECK(game.getElixir(0) == Catch::Approx(10.0f));
}

TEST_CASE("Hero Dark Prince (177) hits for double damage once it has charged 2.5 tiles",
          "[new_card][hero_dark_prince]") {
    auto firstHit = [](float targetDist) {
        Board board;
        auto dummy = std::make_shared<DummyEntity>(1, 9.0f, 5.0f + targetDist, 100000, 1);
        spawn(board, dummy);
        CardRegistry::getInstance().getCard(177)->spawnEntity(9.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto hero = board.getEntities().back();
        advancePastDeploy(hero, board);
        for (int i = 0; i < 200 && dummy->hp == 100000; ++i) hero->update(board);
        return 100000 - dummy->hp;
    };
    CHECK(firstHit(5.0f) == 532); // travel passes 2.5 tiles before contact
    CHECK(firstHit(1.5f) == 266); // control: too close to charge
}

// ---------------- Hero Bowler (178) ----------------

TEST_CASE("Hero Bowler (178) is registered as a Hero with the base Bowler's body",
          "[new_card][hero_bowler]") {
    const CardDefinition* def = CardRegistry::getInstance().getCard(178);
    REQUIRE(def != nullptr);
    CHECK(def->name == "Hero Bowler");
    CHECK(def->cost == Catch::Approx(5.0f));
    CHECK(def->isHero);
    CHECK_FALSE(def->isChampion);
    CHECK_FALSE(def->isSpell);

    Board board;
    auto body = live_sync::mainBody(board, 178);
    REQUIRE(body != nullptr);
    CHECK(body->hp == 2081);
    CHECK(body->getBaseDamage() == 289);
    CHECK(body->getAttackCooldown() == 25);
    CHECK(body->getAttackRange() == Catch::Approx(4.0f));
    CHECK(body->sightRange == Catch::Approx(4.0f));
    CHECK(body->firstHitTicks == 5);
    CHECK(body->isHero);
    CHECK(body->splashRadius == Catch::Approx(1.8f));
    CHECK(body->lineSplash);
    CHECK(body->lineSplashRange == Catch::Approx(11.0f));
}

TEST_CASE("Hero Bowler (178) matches base Bowler (22) stat for stat",
          "[new_card][hero_bowler]") {
    Board a, b;
    auto base = live_sync::mainBody(a, 22);
    auto hero = live_sync::mainBody(b, 178);
    REQUIRE(base != nullptr);
    REQUIRE(hero != nullptr);
    CHECK(hero->hp == base->hp);
    CHECK(hero->getBaseDamage() == base->getBaseDamage());
    CHECK(hero->getAttackCooldown() == base->getAttackCooldown());
    CHECK(hero->firstHitTicks == base->firstHitTicks);
}

TEST_CASE("Hero Bowler (178) is flagged ability-pending and is never dealt in random decks",
          "[new_card][hero_bowler]") {
    const CardDefinition* def = CardRegistry::getInstance().getCard(178);
    REQUIRE(def != nullptr);
    CHECK(def->abilityPending);

    std::mt19937 rng(777);
    int withBowler = 0, withIceGolem = 0;
    for (int i = 0; i < 4000; ++i) {
        for (int id : sampleRandomDeck(rng)) {
            if (id == 178) ++withBowler;
            if (id == 175) ++withIceGolem;
        }
    }
    CHECK(withIceGolem > 0);
    CHECK(withBowler == 0);
}

// ---------------- Hero Tombstone (179) ----------------

TEST_CASE("Hero Tombstone (179) is registered as a 3-elixir Hero with the base Tombstone's body",
          "[new_card][hero_tombstone]") {
    const CardDefinition* def = CardRegistry::getInstance().getCard(179);
    REQUIRE(def != nullptr);
    CHECK(def->name == "Hero Tombstone");
    CHECK(def->cost == Catch::Approx(3.0f));
    CHECK(def->isHero);
    CHECK_FALSE(def->isChampion);
    CHECK_FALSE(def->isSpell);

    Board a, b;
    auto base = live_sync::mainBody(a, 96);
    auto hero = live_sync::mainBody(b, 179);
    REQUIRE(base != nullptr);
    REQUIRE(hero != nullptr);
    CHECK(hero->isHero);
    CHECK(hero->hp == 529);
    CHECK(hero->hp == base->hp);
    CHECK(hero->sightRange == Catch::Approx(base->sightRange));
}

TEST_CASE("Hero Tombstone (179) is flagged ability-pending and is never dealt in random decks",
          "[new_card][hero_tombstone]") {
    const CardDefinition* def = CardRegistry::getInstance().getCard(179);
    REQUIRE(def != nullptr);
    CHECK(def->abilityPending);

    std::mt19937 rng(778);
    int withTombstone = 0, withIceGolem = 0;
    for (int i = 0; i < 4000; ++i) {
        for (int id : sampleRandomDeck(rng)) {
            if (id == 179) ++withTombstone;
            if (id == 175) ++withIceGolem;
        }
    }
    CHECK(withIceGolem > 0);
    CHECK(withTombstone == 0);
}

// ---------------- Hero Valkyrie (180) ----------------

namespace {

struct ValkRig {
    Board board;
    std::shared_ptr<CombatEntity> valk;
    live_sync::DamageRecorder* rec = nullptr;
    std::shared_ptr<live_sync::DamageRecorder> recHolder = std::make_shared<live_sync::DamageRecorder>();

    ValkRig(float x = 9.0f, float y = 10.0f) {
        board.statsEvents.subscribe(recHolder);
        valk = live_sync::firstCombat(live_sync::spawnBare(board, 180, x, y, 0));
        valk->deployTicksRemaining = 0; // the ability is usable once deployed
    }
    std::shared_ptr<CombatEntity> enemy(int cardId, float x, float y) {
        auto e = live_sync::firstCombat(live_sync::spawnBare(board, cardId, x, y, 1));
        e->hp = 100000; // a damage sponge: totals are never clipped by death
        return e;
    }
    void tick() { board.currentTick++; valk->update(board); }
};

} // namespace

TEST_CASE("Hero Valkyrie (180) is registered with base Valkyrie's body and a one-use Wild Whirlwind",
          "[new_card][hero_valkyrie]") {
    const CardDefinition* def = CardRegistry::getInstance().getCard(180);
    REQUIRE(def != nullptr);
    CHECK(def->name == "Hero Valkyrie");
    CHECK(def->cost == Catch::Approx(4.0f));
    CHECK(def->isHero);
    CHECK_FALSE(def->isChampion);
    CHECK_FALSE(def->abilityPending);

    Board a, b;
    auto base = live_sync::mainBody(a, 10);
    auto hero = live_sync::mainBody(b, 180);
    REQUIRE(hero != nullptr);
    CHECK(hero->hp == 1907);
    CHECK(hero->getBaseDamage() == 266);
    CHECK(hero->getAttackCooldown() == 15);
    CHECK(hero->firstHitTicks == base->firstHitTicks);
    CHECK(hero->sightRange == Catch::Approx(5.5f));
    CHECK(hero->abilityElixirCost == Catch::Approx(3.0f));
    CHECK(hero->abilityCooldownTicks == 0);
    CHECK(hero->abilityUsesRemaining == 1);
}

TEST_CASE("Hero Valkyrie (180) Whirlwind dashes to the nearest enemy ground non-tower within 5.5 tiles",
          "[new_card][hero_valkyrie]") {
    {
        ValkRig r;
        auto foe = r.enemy(0, 9.0f, 15.0f); // 5.0 away
        REQUIRE(r.valk->activateAbility(r.board));
        CHECK(r.valk->position.distanceTo(foe->position) == Catch::Approx(1.0f).margin(0.01f));
    }
    {
        ValkRig r; // control: out of dash range, she does not move
        r.enemy(0, 9.0f, 16.0f); // 6.0 away
        REQUIRE(r.valk->activateAbility(r.board));
        CHECK(r.valk->position.y == Catch::Approx(10.0f));
    }
    {
        ValkRig r; // air units are not dash targets
        r.enemy(44, 9.0f, 15.0f);
        REQUIRE(r.valk->activateAbility(r.board));
        CHECK(r.valk->position.y == Catch::Approx(10.0f));
    }
}

TEST_CASE("Hero Valkyrie (180) Whirlwind never dashes to a tower (2026-09-01 fix)",
          "[new_card][hero_valkyrie]") {
    live_sync::Match m;
    std::shared_ptr<Entity> tower;
    for (const auto& e : m.board().getEntities())
        if (e->isTower() && e->team == 1) { tower = e; break; }
    REQUIRE(tower != nullptr);
    auto bodies = m.spawnCard(180, tower->position.x, tower->position.y - 4.0f, 0);
    auto valk = live_sync::firstCombat(bodies);
    REQUIRE(valk != nullptr);
    valk->deployTicksRemaining = 0;
    const Vector2D before = valk->position;
    REQUIRE(valk->activateAbility(m.board()));
    CHECK(valk->position.distanceTo(before) == Catch::Approx(0.0f));
    CHECK(valk->whirlTicksRemaining == 35); // she still spins
}

TEST_CASE("Hero Valkyrie (180) spin lands 14 hits of 97 over 35 ticks, then stops",
          "[new_card][hero_valkyrie]") {
    ValkRig r;
    auto foe = r.enemy(0, 9.0f, 11.5f); // she closes to 1.0 tile (y 10.5), staying inside 2.5
    REQUIRE(r.valk->activateAbility(r.board));
    CHECK(r.valk->position.y == Catch::Approx(10.5f));
    std::vector<int> hitTicks;
    int last = 0;
    for (int t = 1; t <= 35; ++t) {
        r.tick();
        const int now = r.recHolder->totalTo(foe->id);
        if (now != last) { hitTicks.push_back(t); CHECK(now - last == 97); }
        last = now;
    }
    CHECK(last == 14 * 97);
    CHECK(hitTicks.size() == 14);
    CHECK(hitTicks.front() == 3);
    CHECK(hitTicks.back() == 35);
    CHECK(r.valk->whirlTicksRemaining == 0);
    r.tick(); // tick 36: the spin is over
    CHECK(r.recHolder->totalTo(foe->id) == last);
}

TEST_CASE("Hero Valkyrie (180) spin ignores units beyond 2.5 tiles, air units and allies",
          "[new_card][hero_valkyrie]") {
    ValkRig r;
    auto inside = r.enemy(0, 9.0f, 11.5f);
    auto outside = r.enemy(0, 9.0f, 14.5f);  // 4.0 tiles from where she ends up (y 10.5)
    auto air = r.enemy(44, 9.0f, 11.0f);
    auto ally = live_sync::firstCombat(live_sync::spawnBare(r.board, 0, 9.0f, 11.0f, 0));
    REQUIRE(r.valk->activateAbility(r.board));
    for (int t = 0; t < 35; ++t) r.tick();
    CHECK(r.recHolder->totalTo(inside->id) > 0);
    CHECK(r.recHolder->totalTo(outside->id) == 0);
    CHECK(r.recHolder->totalTo(air->id) == 0);
    CHECK(r.recHolder->totalTo(ally->id) == 0);
}

TEST_CASE("Hero Valkyrie (180) spin hits an invisible unit that a normal attack cannot (2026-09-03 fix)",
          "[new_card][hero_valkyrie]") {
    ValkRig r;
    auto ghost = r.enemy(0, 9.0f, 11.5f);
    ghost->temporaryInvisibilityTicksRemaining = 100;
    REQUIRE_FALSE(ghost->isTargetable());
    REQUIRE(r.valk->activateAbility(r.board));
    for (int t = 0; t < 35; ++t) r.tick();
    CHECK(r.recHolder->totalTo(ghost->id) == 14 * 97);
}

TEST_CASE("Hero Valkyrie (180) spin deals 48 to a Crown Tower per hit",
          "[new_card][hero_valkyrie]") {
    live_sync::Match m;
    std::shared_ptr<Entity> tower;
    for (const auto& e : m.board().getEntities())
        if (e->isTower() && e->team == 1) { tower = e; break; }
    REQUIRE(tower != nullptr);
    auto valk = live_sync::firstCombat(m.spawnCard(180, tower->position.x, tower->position.y - 1.5f, 0));
    REQUIRE(valk != nullptr);
    valk->deployTicksRemaining = 0;
    REQUIRE(valk->activateAbility(m.board()));
    for (int t = 0; t < 35; ++t) valk->update(m.board());
    CHECK(m.rec->totalTo(tower->id) == 14 * 48);
}

TEST_CASE("Hero Valkyrie (180) takes 15% less damage while spinning, and full damage after",
          "[new_card][hero_valkyrie]") {
    ValkRig r;
    REQUIRE(r.valk->activateAbility(r.board));
    const int hp0 = r.valk->hp;
    r.valk->takeDamage(100);
    CHECK(hp0 - r.valk->hp == 85);
    for (int t = 0; t < 35; ++t) r.tick();
    const int hp1 = r.valk->hp;
    r.valk->takeDamage(100);
    CHECK(hp1 - r.valk->hp == 100);
}

TEST_CASE("Hero Valkyrie (180) Whirlwind costs 3 elixir, is refused at 2.9 and is single-use",
          "[new_card][hero_valkyrie]") {
    GameManager game({ 1, 180, 2, 3, 4, 5, 6, 7 }, { 0, 1, 2, 3, 4, 5, 6, 7 });
    game.playerAI.hand[1] = 180;
    game.playerAI.elixir = 100.0f;
    game.playCard(0, 180, 9.0f, 10.0f);
    for (int i = 0; i < 12; ++i) game.step(); // deployed
    game.playerAI.elixir = 2.9f;
    REQUIRE_FALSE(game.activateChampionAbility(0, 1));
    game.playerAI.elixir = 10.0f;
    REQUIRE(game.activateChampionAbility(0, 1));
    CHECK(game.getElixirAI() == Catch::Approx(7.0f));
    REQUIRE_FALSE(game.activateChampionAbility(0, 1));
    CHECK(game.getElixirAI() == Catch::Approx(7.0f));
}

TEST_CASE("A deck with Hero Valkyrie (180) passes validateDeckSlots in slot 1",
          "[new_card][hero_valkyrie]") {
    CHECK(validateDeckSlots({ 1, 180, 2, 3, 4, 5, 6, 7 }).empty());
    CHECK(countHeroes({ 180, 1, 2, 3, 4, 5, 6, 7 }) == 1);
}
