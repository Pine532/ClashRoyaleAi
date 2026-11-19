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
