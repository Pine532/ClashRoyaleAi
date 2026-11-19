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

// ---------------- Hero Berserker (181) ----------------

namespace {

struct BerserkRig {
    Board board;
    std::shared_ptr<CombatEntity> hero;
    std::shared_ptr<CombatEntity> foe;
    std::shared_ptr<live_sync::DamageRecorder> rec = std::make_shared<live_sync::DamageRecorder>();

    BerserkRig() {
        board.statsEvents.subscribe(rec);
        hero = live_sync::firstCombat(live_sync::spawnBare(board, 181, 9.0f, 10.0f, 0));
        hero->deployTicksRemaining = 0; // the ability is usable once deployed
        foe = live_sync::firstCombat(live_sync::spawnBare(board, 0, 9.0f, 10.8f, 1));
        foe->hp = 100000; // a damage sponge: hits are never clipped by death
    }
    void tick() { board.currentTick++; hero->update(board); }
    // Ticks at which the foe took damage over the next n ticks.
    std::vector<int> hitTicks(int n, int startTick) {
        std::vector<int> out;
        int last = rec->totalTo(foe->id);
        for (int t = 0; t < n; ++t) {
            tick();
            const int now = rec->totalTo(foe->id);
            if (now != last) out.push_back(startTick + t);
            last = now;
        }
        return out;
    }
};

} // namespace

TEST_CASE("Hero Berserker (181) is registered with base Berserker's body and a one-use Savage Survival",
          "[new_card][hero_berserker]") {
    const CardDefinition* def = CardRegistry::getInstance().getCard(181);
    REQUIRE(def != nullptr);
    CHECK(def->name == "Hero Berserker");
    CHECK(def->cost == Catch::Approx(2.0f));
    CHECK(def->isHero);
    CHECK_FALSE(def->isChampion);
    CHECK_FALSE(def->abilityPending);

    Board a, b;
    auto base = live_sync::mainBody(a, 51);
    auto hero = live_sync::mainBody(b, 181);
    REQUIRE(hero != nullptr);
    CHECK(hero->hp == 896);
    CHECK(hero->getBaseDamage() == 102);
    CHECK(hero->getAttackCooldown() == 6);
    CHECK(hero->getAttackRange() == Catch::Approx(0.8f));
    CHECK(std::dynamic_pointer_cast<Troop>(hero)->getSpeed()
          == Catch::Approx(std::dynamic_pointer_cast<Troop>(base)->getSpeed()));
    CHECK(hero->sightRange == Catch::Approx(5.5f));
    CHECK(hero->abilityElixirCost == Catch::Approx(3.0f));
    CHECK(hero->abilityCooldownTicks == 0);
    CHECK(hero->abilityUsesRemaining == 1);
}

TEST_CASE("Hero Berserker (181) swings every 2 ticks for 35 ticks after Savage Survival, then every 6",
          "[new_card][hero_berserker]") {
    BerserkRig r;
    const auto before = r.hitTicks(30, 0);
    REQUIRE(before.size() >= 3);
    for (size_t i = 1; i < before.size(); ++i) CHECK(before[i] - before[i - 1] == 6);

    REQUIRE(r.hero->activateAbility(r.board));
    const auto during = r.hitTicks(35, 100);
    REQUIRE(during.size() >= 10);
    // The swing in progress finishes on its old cooldown; every one after it is 2 ticks.
    for (size_t i = 2; i < during.size(); ++i) CHECK(during[i] - during[i - 1] == 2);
    CHECK(r.hero->abilityHitCooldownTicksRemaining == 0); // window length: 35 ticks
    const auto after = r.hitTicks(30, 200);
    REQUIRE(after.size() >= 3);
    for (size_t i = 2; i < after.size(); ++i) CHECK(after[i] - after[i - 1] == 6);
}

TEST_CASE("Hero Berserker (181) window is exactly 35 ticks long",
          "[new_card][hero_berserker]") {
    BerserkRig r;
    REQUIRE(r.hero->activateAbility(r.board));
    for (int t = 0; t < 34; ++t) r.tick();
    CHECK(r.hero->abilityHpFloorTicksRemaining == 1); // still active at 34
    r.tick();
    CHECK(r.hero->abilityHpFloorTicksRemaining == 0);
}

TEST_CASE("Hero Berserker (181) cannot be reduced below 1 hp during Savage Survival",
          "[new_card][hero_berserker]") {
    {
        BerserkRig r;
        REQUIRE(r.hero->activateAbility(r.board));
        r.hero->takeDamage(5000);
        CHECK(r.hero->hp == 1);
        CHECK(r.hero->isAlive());
    }
    {
        BerserkRig r; // control: without the ability the same hit kills
        r.hero->takeDamage(5000);
        CHECK_FALSE(r.hero->isAlive());
    }
    {
        BerserkRig r; // after the window the floor is gone
        REQUIRE(r.hero->activateAbility(r.board));
        for (int t = 0; t < 35; ++t) r.tick();
        r.hero->takeDamage(5000);
        CHECK_FALSE(r.hero->isAlive());
    }
}

TEST_CASE("Hero Berserker (181) survives damage over time during Savage Survival",
          "[new_card][hero_berserker]") {
    for (bool active : { true, false }) {
        BerserkRig r;
        r.foe->hp = 100000;
        if (active) REQUIRE(r.hero->activateAbility(r.board));
        r.hero->applyDot(2000, 3, 1);
        for (int t = 0; t < 5; ++t) r.tick();
        CHECK(r.hero->isAlive() == active); // control: the DoT kills her without it
    }
}

TEST_CASE("Hero Berserker (181) Savage Survival costs 3 elixir, is refused at 2.9 and is single-use",
          "[new_card][hero_berserker]") {
    GameManager game({ 1, 181, 2, 3, 4, 5, 6, 7 }, { 0, 1, 2, 3, 4, 5, 6, 7 });
    game.playerAI.hand[1] = 181;
    game.playerAI.elixir = 100.0f;
    game.playCard(0, 181, 9.0f, 10.0f);
    for (int i = 0; i < 12; ++i) game.step(); // deployed
    game.playerAI.elixir = 2.9f;
    REQUIRE_FALSE(game.activateChampionAbility(0, 1));
    game.playerAI.elixir = 10.0f;
    REQUIRE(game.activateChampionAbility(0, 1));
    CHECK(game.getElixirAI() == Catch::Approx(7.0f));
    REQUIRE_FALSE(game.activateChampionAbility(0, 1));
    CHECK(game.getElixirAI() == Catch::Approx(7.0f));
}

TEST_CASE("Hero Berserker (181) deals 25 to a Crown Tower during Savage Survival and 102 otherwise",
          "[new_card][hero_berserker]") {
    auto towerHit = [](bool active) {
        live_sync::Match m;
        std::shared_ptr<Entity> tower;
        for (const auto& e : m.board().getEntities())
            if (e->isTower() && e->team == 1) { tower = e; break; }
        auto hero = live_sync::firstCombat(m.spawnCard(181, tower->position.x, tower->position.y - 1.5f, 0));
        hero->deployTicksRemaining = 0;
        if (active) hero->activateAbility(m.board());
        for (int t = 0; t < 40 && m.rec->totalTo(tower->id) == 0; ++t) hero->update(m.board());
        return m.rec->totalTo(tower->id);
    };
    CHECK(towerHit(true) == 25);   // 102 * 0.25, truncated
    CHECK(towerHit(false) == 102); // control: the penalty is only inside the window
}

TEST_CASE("Hero Berserker (181) deals full damage to a non-tower during Savage Survival",
          "[new_card][hero_berserker]") {
    BerserkRig r;
    REQUIRE(r.hero->activateAbility(r.board));
    for (int t = 0; t < 12; ++t) r.tick();
    const int total = r.rec->totalTo(r.foe->id);
    REQUIRE(total > 0);
    CHECK(total % 102 == 0); // every hit 102, none scaled
}

TEST_CASE("Hero Berserker (181) Savage Survival does not change her movement speed",
          "[new_card][hero_berserker]") {
    // Pins the omission: "Ultra Fast" has no published numeric speed. Remove when one is sourced.
    BerserkRig r;
    auto troop = std::dynamic_pointer_cast<Troop>(r.hero);
    const float before = troop->getSpeed();
    REQUIRE(r.hero->activateAbility(r.board));
    CHECK(troop->getSpeed() == Catch::Approx(before));
}

TEST_CASE("A deck with Hero Berserker (181) passes validateDeckSlots in slot 1",
          "[new_card][hero_berserker]") {
    CHECK(validateDeckSlots({ 1, 181, 2, 3, 4, 5, 6, 7 }).empty());
    CHECK(countHeroes({ 181, 1, 2, 3, 4, 5, 6, 7 }) == 1);
}

// ---------------- Elite Barbarians Evolution (182) ----------------

#include "EliteBarbarianSpearEffect.h"

namespace {
// A ground CombatEntity at centre distance gap + 0.8 above (9,10): surface gap
// `gap` for two troop-radius bodies.
std::shared_ptr<StationaryCombatant> eliteDummy(Board& b, int id, float gap, int team = 1) {
    auto d = std::make_shared<StationaryCombatant>(id, 9.0f, 10.0f + gap + 0.8f, 100000, team, 0.0f, 0, 10);
    b.addEntity(d);
    b.commitPendingEntities();
    return d;
}
int eliteHit(float gap, bool flying = false) {
    Board b;
    auto d = eliteDummy(b, 900, gap);
    d->isFlying = flying;
    EliteBarbarianSpearEffect(3.0f, 4.5f, 220).apply(b, Vector2D{9.0f, 10.0f}, 0);
    return 100000 - d->hp;
}
} // namespace

TEST_CASE("Elite Barbarians Evolution (182) is registered with the base body and a 5 s spear timer",
          "[new_card][evo_elite_barbarians]") {
    const CardDefinition* def = CardRegistry::getInstance().getCard(182);
    REQUIRE(def != nullptr);
    CHECK(def->name == "Elite Barbarians");
    CHECK(def->cost == Catch::Approx(6.0f));
    CHECK(def->isEvolution);
    CHECK_FALSE(def->isSpell);
    CHECK(def->evolutionCycleThreshold == 1);
    CHECK(def->evolvedUsesGranted == 1);

    Board board;
    auto bodies = live_sync::spawnBare(board, 182, 9.0f, 10.0f, 0, true);
    int n = 0;
    for (const auto& e : bodies) {
        auto c = std::dynamic_pointer_cast<CombatEntity>(e);
        if (!c) continue;
        ++n;
        CHECK(c->hp == 1341);
        CHECK(c->getBaseDamage() == 384);
        CHECK(c->getAttackCooldown() == 14);
        CHECK(c->getAttackRange() == Catch::Approx(1.2f));
        CHECK(c->sightRange == Catch::Approx(5.5f));
        CHECK(c->firstHitTicks == 5);
        CHECK(c->periodicIntervalTicks == 50);
        CHECK(c->periodicEffect != nullptr);
        CHECK(c->periodicTicksUntilNext == 50);
    }
    CHECK(n == 2);
}

TEST_CASE("Elite Barbarians Evolution (182): the base card and the un-evolved form have no spear",
          "[new_card][evo_elite_barbarians]") {
    // Control that must NOT fire.
    Board a, b;
    auto base = live_sync::mainBody(a, 17);
    auto plain = live_sync::mainBody(b, 182, false);
    REQUIRE(base != nullptr);
    REQUIRE(plain != nullptr);
    CHECK(base->periodicIntervalTicks == 0);
    CHECK(base->periodicEffect == nullptr);
    CHECK(plain->periodicIntervalTicks == 0);
    CHECK(plain->periodicEffect == nullptr);
}

TEST_CASE("Elite Barbarians Evolution (182) spear band edges are 3 and 4.5 tiles surface-to-surface",
          "[new_card][evo_elite_barbarians]") {
    CHECK(eliteHit(3.05f) == 220); // control: must fire
    CHECK(eliteHit(2.9f) == 0);
    CHECK(eliteHit(4.45f) == 220);
    CHECK(eliteHit(4.6f) == 0);
}

TEST_CASE("Elite Barbarians Evolution (182) spear ignores flying targets",
          "[new_card][evo_elite_barbarians]") {
    CHECK(eliteHit(4.0f, false) == 220);
    CHECK(eliteHit(4.0f, true) == 0);
}

TEST_CASE("Elite Barbarians Evolution (182) spear hits only the nearest enemy, never a friendly",
          "[new_card][evo_elite_barbarians]") {
    Board b;
    auto far = std::make_shared<StationaryCombatant>(901, 9.0f, 10.0f + 4.0f + 0.8f, 100000, 1, 0.0f, 0, 10);
    auto near = std::make_shared<StationaryCombatant>(902, 10.0f, 10.0f + 3.5f + 0.8f, 100000, 1, 0.0f, 0, 10);
    auto ally = std::make_shared<StationaryCombatant>(903, 8.0f, 10.0f + 3.2f + 0.8f, 100000, 0, 0.0f, 0, 10);
    b.addEntity(far); b.addEntity(near); b.addEntity(ally);
    b.commitPendingEntities();
    // near is at a slightly larger centre offset in x; recompute so it is still nearest.
    EliteBarbarianSpearEffect(3.0f, 4.5f, 220).apply(b, Vector2D{9.0f, 10.0f}, 0);
    CHECK(near->hp == 100000 - 220);
    CHECK(far->hp == 100000);
    CHECK(ally->hp == 100000);
}

TEST_CASE("Elite Barbarians Evolution (182) throws through update() once per interval",
          "[new_card][evo_elite_barbarians]") {
    Board b;
    auto bodies = live_sync::spawnBare(b, 182, 9.0f, 10.0f, 0, true);
    std::vector<std::shared_ptr<CombatEntity>> bs;
    for (const auto& e : bodies)
        if (auto c = std::dynamic_pointer_cast<CombatEntity>(e)) bs.push_back(c);
    REQUIRE(bs.size() == 2);
    auto foe = std::make_shared<StationaryCombatant>(950, 9.0f, 14.0f, 100000, 1, 0.0f, 0, 10);
    b.addEntity(foe);
    b.commitPendingEntities();
    for (auto& c : bs) { advancePastDeploy(c, b); c->periodicTicksUntilNext = 1; }
    for (auto& c : bs) c->update(b);
    CHECK(foe->hp == 100000 - 2 * 220);
    for (auto& c : bs) CHECK(c->periodicTicksUntilNext == 50);
}

// ---------------- Minion Giant (183) ----------------

TEST_CASE("Minion Giant (183) resolves as a 4-elixir troop with the live level-11 stats",
          "[new_card][minion_giant]") {
    const CardDefinition* def = CardRegistry::getInstance().getCard(183);
    REQUIRE(def != nullptr);
    CHECK(def->name == "Minion Giant");
    CHECK(def->cost == Catch::Approx(4.0f));
    CHECK_FALSE(def->isSpell);
    CHECK_FALSE(def->isBuilding);
    CHECK(def->isFlying);

    Board board;
    auto body = live_sync::mainBody(board, 183);
    REQUIRE(body != nullptr);
    CHECK(body->hp == 1817);
    CHECK(body->getBaseDamage() == 153);
    CHECK(body->getAttackCooldown() == 17);
    CHECK(body->getAttackRange() == Catch::Approx(4.0f));
    CHECK(body->symbol == 'Y');
    CHECK(body->isFlying);
    CHECK_FALSE(body->targetsAir);
    CHECK(std::dynamic_pointer_cast<RangedBuildingTargeter>(body) != nullptr);
}

TEST_CASE("Minion Giant (183) ignores troops and hits a tower for 153 every 17 ticks",
          "[new_card][minion_giant]") {
    live_sync::Match m;
    std::shared_ptr<Entity> tower;
    for (const auto& e : m.board().getEntities())
        if (e->isTower() && e->team == 1) { tower = e; break; }
    REQUIRE(tower != nullptr);
    auto giant = live_sync::firstCombat(m.spawnCard(183, tower->position.x, tower->position.y - 3.0f, 0));
    REQUIRE(giant != nullptr);
    // A troop right beside it must be ignored by a building targeter.
    auto knight = live_sync::firstCombat(m.spawnCard(0, tower->position.x + 1.0f, tower->position.y - 3.0f, 1));
    REQUIRE(knight != nullptr);
    giant->deployTicksRemaining = 0;
    const int knightHp = knight->hp;
    std::vector<int> hitTicks;
    int last = 0;
    for (int t = 0; t < 60; ++t) {
        m.step();
        const int now = m.rec->totalTo(tower->id);
        if (now != last) {
            CHECK((now - last) == 153);
            hitTicks.push_back(t);
            last = now;
        }
    }
    REQUIRE(hitTicks.size() >= 3);
    for (size_t i = 1; i < hitTicks.size(); ++i) CHECK(hitTicks[i] - hitTicks[i - 1] == 17);
    CHECK(m.rec->totalTo(knight->id) == 0);
}

TEST_CASE("Minion Giant (183) is hit by an air-targeting unit but not by a ground-only one",
          "[new_card][minion_giant]") {
    Board board;
    auto giant = live_sync::firstCombat(live_sync::spawnBare(board, 183, 9.0f, 10.0f, 1));
    auto knight = live_sync::firstCombat(live_sync::spawnBare(board, 0, 9.0f, 11.0f, 0));
    auto archers = live_sync::spawnBare(board, 1, 9.0f, 12.0f, 0);
    REQUIRE(giant != nullptr);
    REQUIRE(knight != nullptr);
    CHECK_FALSE(knight->targetsAir);
    bool anyAir = false;
    for (const auto& e : archers)
        if (auto c = std::dynamic_pointer_cast<CombatEntity>(e)) anyAir = anyAir || c->targetsAir;
    CHECK(anyAir);
}

TEST_CASE("Minion Giant (183) takes damage from Archers but none from a Knight in melee range",
          "[new_card][minion_giant]") {
    // Behavioural check of the flying rule: same setup, only the attacker differs.
    auto damageTaken = [](int attackerId) {
        live_sync::Match m;
        auto giant = live_sync::firstCombat(m.spawnCard(183, 9.0f, 12.0f, 1));
        auto foe = live_sync::firstCombat(m.spawnCard(attackerId, 9.0f, 13.0f, 0));
        REQUIRE(giant != nullptr);
        REQUIRE(foe != nullptr);
        giant->deployTicksRemaining = 0;
        foe->deployTicksRemaining = 0;
        m.step(40);
        // Count only the placed attacker; a tower may also fire at the giant.
        int sum = 0;
        for (const auto& e : m.rec->damage)
            if (e.targetId == giant->id && e.attackerCardId == attackerId) sum += e.amount;
        return sum;
    };
    CHECK(damageTaken(0) == 0);   // Knight: ground only
    CHECK(damageTaken(1) > 0);    // Archers: hit air
}
