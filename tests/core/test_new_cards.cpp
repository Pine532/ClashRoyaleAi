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
