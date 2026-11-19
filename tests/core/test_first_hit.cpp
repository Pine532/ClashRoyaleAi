#include <catch_amalgamated.hpp>
#include "Board.h"
#include "CardFactories.h"
#include "CardRegistry.h"
#include "CardStats.h"
#include "ClashEnv.h"
#include "MeleeTroop.h"
#include <vector>

// First-hit time: the wind-up a unit pays when it STARTS engaging a target
// (CardStats::firstHitTicks; RoyaleAPI cr-api-data "First Hit = Hit Time - Load
// Time", docs/LIVE_SYNC.md). Everything is driven through Board::update-order
// ticking with hand-built units, so no registered card needs a value.
//
// Tick convention: an engagement that starts (the target is in range) on tick X
// with F > 0 fires on tick X + F; F = 0 fires on tick X.

namespace {

void tickBoard(Board& board, int tick) {
    board.currentTick = tick;
    board.commitPendingEntities(tick);
    for (const auto& e : board.getEntities()) {
        if (e->isAlive()) e->update(board);
    }
}

constexpr int COOLDOWN = 12;

// A stationary attacker (id 1, team 0, damage 100) and an inert target (id 2).
struct Duel {
    Board board;
    std::shared_ptr<MeleeTroop> attacker;
    std::shared_ptr<MeleeTroop> target;
    explicit Duel(int firstHit, int cooldown = COOLDOWN, int targetHp = 100000) {
        attacker = std::make_shared<MeleeTroop>(1, 9.0f, 9.0f, 2000, 0, 0.0f, 1.2f, 100, cooldown, 'K');
        target = std::make_shared<MeleeTroop>(2, 9.0f, 9.5f, targetHp, 1, 0.0f, 1.2f, 0, 12, 'T');
        attacker->firstHitTicks = firstHit;
        board.addEntity(attacker);
        board.addEntity(target);
        board.commitPendingEntities(0);
    }
};

int hpOf(Board& b, int id) {
    for (const auto& e : b.getEntities()) if (e->id == id) return e->hp;
    return -1;
}

// Ticks (0-based) on which entity `id` lost hp, over [0, n).
template <typename BeforeTick>
std::vector<int> hitTicks(Board& b, int id, int n, BeforeTick before) {
    std::vector<int> out;
    int last = hpOf(b, id);
    for (int t = 0; t < n; ++t) {
        before(t);
        tickBoard(b, t);
        int now = hpOf(b, id);
        if (now < last) out.push_back(t);
        last = now;
    }
    return out;
}
std::vector<int> hitTicks(Board& b, int id, int n) {
    return hitTicks(b, id, n, [](int) {});
}

}  // namespace

TEST_CASE("the first hit lands firstHitTicks after the engagement starts", "[first_hit]") {
    Duel d(7);
    auto hits = hitTicks(d.board, 2, 30);
    REQUIRE_FALSE(hits.empty());
    REQUIRE(hits.front() == 7);

    // Control: without the wind-up the same duel hits at once.
    Duel c(0);
    REQUIRE(hitTicks(c.board, 2, 30).front() == 0);
}

TEST_CASE("continuous attacks stay attackCooldown apart", "[first_hit]") {
    Duel d(7);
    auto hits = hitTicks(d.board, 2, 80);
    REQUIRE(hits.size() >= 4);
    REQUIRE(hits[0] == 7);
    for (size_t i = 1; i < hits.size(); ++i) {
        REQUIRE(hits[i] - hits[i - 1] == COOLDOWN);
    }
}

TEST_CASE("retargeting after a kill costs max(cooldown left, wind-up), not the sum", "[first_hit]") {
    // Two targets in range, the nearer one dies to the first hit.
    auto run = [](int firstHit, int cooldown) {
        Duel d(firstHit, cooldown, 100);  // target 2 dies to one 100-damage hit
        auto second = std::make_shared<MeleeTroop>(3, 9.0f, 9.9f, 100000, 1, 0.0f, 1.2f, 0, 12, 'T');
        d.board.addEntity(second);
        d.board.commitPendingEntities(0);
        auto hits = hitTicks(d.board, 3, 120);
        REQUIRE_FALSE(hits.empty());
        return hits.front();
    };
    // Kill on tick 7 (cooldown 12 -> ready on 19); the wind-up on target 3 is
    // armed on tick 8 and runs out on 15. Cooldown dominates: 19, not 19 + 7.
    REQUIRE(run(7, 12) == 19);
    // Wind-up 20 vs cooldown 5: kill on 20, re-armed on 21, fires on 41.
    REQUIRE(run(20, 5) == 41);
}

TEST_CASE("a stun mid-wind-up restarts it; a slow stretches it", "[first_hit]") {
    SECTION("stun") {
        Duel d(10);
        // Stun for ticks 4 and 5; the unit re-engages on tick 6 and winds up again.
        auto hits = hitTicks(d.board, 2, 40, [&](int t) { if (t == 4) d.attacker->applyFreeze(2, 0.0f); });
        REQUIRE_FALSE(hits.empty());
        REQUIRE(hits.front() == 16);
    }
    SECTION("slow") {
        Duel d(10);
        d.attacker->applyFreeze(1000, 0.5f);
        auto hits = hitTicks(d.board, 2, 60);
        REQUIRE_FALSE(hits.empty());
        // Armed on tick 0, then 0.5 per tick: 20 ticks, against 10 unslowed.
        REQUIRE(hits.front() == 20);
    }
}

TEST_CASE("a target leaving range resets the wind-up", "[first_hit]") {
    Duel d(10);
    auto hits = hitTicks(d.board, 2, 40, [&](int t) {
        if (t == 5) d.target->position = Vector2D{9.0f, 12.0f};   // out of attack range
        if (t == 8) d.target->position = Vector2D{9.0f, 9.5f};    // back in range
    });
    REQUIRE_FALSE(hits.empty());
    REQUIRE(hits.front() == 18);
}

TEST_CASE("firstHitTicks = 0 is the pre-existing behaviour", "[first_hit]") {
    Duel d(0);
    auto hits = hitTicks(d.board, 2, 200);
    std::vector<int> expected;
    for (int t = 0; t < 200; t += COOLDOWN) expected.push_back(t);
    REQUIRE(hits == expected);
    REQUIRE(d.attacker->getWindupTicksRemaining() == 0.0f);
}

TEST_CASE("a deepCopy taken mid-wind-up continues identically", "[first_hit][deepcopy]") {
    Duel d(9);
    for (int t = 0; t < 3; ++t) tickBoard(d.board, t);
    REQUIRE(hpOf(d.board, 2) == 100000);   // still winding up

    Board copy = d.board.deepCopy();
    auto trace = [](Board& b) {
        std::vector<int> out;
        for (int t = 3; t < 40; ++t) { tickBoard(b, t); out.push_back(hpOf(b, 2)); }
        return out;
    };
    auto a = trace(d.board);
    auto b = trace(copy);
    REQUIRE(a == b);
    REQUIRE(a.front() > a.back());   // non-vacuous: hits landed
}

TEST_CASE("inject with deploy_ticks 0 waives the first wind-up, -1 does not", "[first_hit][inject]") {
    constexpr int KNIGHT = 0;
    const std::vector<int> deck = { 15, 6, 25, 40, 24, 72, 33, 7 };
    auto firstHitTick = [&](int deployTicks) {
        ClashEnv env(deck, deck, 100);
        env.reset();
        env.inject(KNIGHT, 9.0f, 10.0f, 0, -1.0f, deployTicks);
        env.inject(KNIGHT, 9.0f, 10.5f, 1, -1.0f, 0);
        Board& board = env.debugGame().getBoard();
        board.commitPendingEntities(0);
        std::shared_ptr<CombatEntity> mine, theirs;
        for (const auto& e : board.getEntities()) {
            auto c = std::dynamic_pointer_cast<CombatEntity>(e);
            if (!c || c->isTower() || c->cardId != KNIGHT) continue;
            (c->team == 0 ? mine : theirs) = c;
        }
        REQUIRE(mine);
        REQUIRE(theirs);
        mine->firstHitTicks = 7;
        int last = theirs->hp;
        for (int t = 0; t < 40; ++t) {
            board.currentTick = t;
            if (mine->isAlive()) mine->update(board);
            if (theirs->isAlive()) theirs->update(board);
            if (theirs->hp < last) return t;
        }
        return -1;
    };
    // Deploying units are inert; deploy_ticks 0 leaves none.
    const int waived = firstHitTick(0);
    const int normal = firstHitTick(-1);
    REQUIRE(waived == 0);
    // 10 deploy ticks, then the 7-tick wind-up (armed on the first active tick).
    REQUIRE(normal == 17);
}

TEST_CASE("firstHitTicks is copied from CardStats; registry cards read 0", "[first_hit][registry]") {
    CardStats stats;
    stats.name = "Windup";
    stats.hp = 100; stats.speed = 1.0f; stats.attackRange = 1.0f;
    stats.damage = 10; stats.attackCooldown = 10; stats.symbol = 'K';
    stats.withFirstHit(9);
    {
        Board board;
        CardFactories::spawnMeleeSquad(stats, 5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto troop = std::dynamic_pointer_cast<CombatEntity>(board.getEntities().back());
        REQUIRE(troop);
        REQUIRE(troop->firstHitTicks == 9);
    }
    {
        // Control: Lava Hound, which has no sourced first hit (its hit speed
        // changed after the 2023 RoyaleAPI data; docs/LIVE_SYNC.md).
        Board board;
        CardRegistry::getInstance().getCard(91)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto troop = std::dynamic_pointer_cast<CombatEntity>(board.getEntities().back());
        REQUIRE(troop);
        REQUIRE(troop->firstHitTicks == 0);
    }
}

// ---- Card values: official balance notes ------------------------------------
// Every unit whose first hit an official post states (latest value; see the
// comment at each literal in CardRegistry.h and docs/LIVE_SYNC.md).

#include "live_sync_test_helpers.h"
#include <set>

TEST_CASE("Official first-hit values reach the spawned units", "[first_hit][first_hit_values][official]") {
    struct Row { int id; bool evolved; int ticks; };
    const std::vector<Row> rows = {
        {4, false, 6}, {6, false, 7}, {11, false, 5}, {23, false, 5}, {54, false, 4},
        {57, false, 5}, {61, false, 3}, {62, false, 7}, {76, false, 5}, {79, false, 8},
        {80, false, 7}, {84, false, 10}, {93, false, 10}, {167, false, 5}, {168, false, 7},
        {172, false, 6},
        {140, false, 7}, {140, true, 7}, {141, false, 5}, {141, true, 5},
        {145, false, 3}, {145, true, 3}, {146, false, 7}, {146, true, 7},
        {151, false, 5}, {151, true, 5}, {152, false, 10}, {152, true, 10},
    };
    for (const auto& r : rows) {
        INFO("card " << r.id << (r.evolved ? " (evolved)" : ""));
        Board board;
        auto body = live_sync::mainBody(board, r.id, r.evolved);
        REQUIRE(body != nullptr);
        CHECK(body->firstHitTicks == r.ticks);
    }
}

TEST_CASE("Goblin Gang: Goblins wind up 0.6 s, its Spear Goblins 0.5 s", "[first_hit][first_hit_values][official]") {
    Board board;
    auto bodies = live_sync::spawnBare(board, 112);
    std::set<int> seen;
    for (const auto& e : bodies)
        if (auto c = std::dynamic_pointer_cast<CombatEntity>(e)) seen.insert(c->firstHitTicks);
    CHECK(seen == std::set<int>{5, 6});
}

TEST_CASE("Goblin Barrel's Goblins wind up 0.3 s (2026-10-06)", "[first_hit][first_hit_values][official]") {
    live_sync::Match m;
    m.spawnCard(109, 9.0f, 20.0f, 0);
    std::shared_ptr<CombatEntity> goblin;
    for (int i = 0; i < 60 && !goblin; ++i) {
        m.step();
        for (const auto& e : m.board().getEntities())
            if (e->team == 0 && e->name == "Goblins")
                if (auto c = std::dynamic_pointer_cast<CombatEntity>(e)) { goblin = c; break; }
    }
    REQUIRE(goblin != nullptr);
    CHECK(goblin->firstHitTicks == 3);
}

// ---- Card values: RoyaleAPI cr-api-data (2023-10), hit_speed - load_time -----
// Only where load_time > 0, the result is >= 0, load_first_hit is false, and the
// card's hit speed is still the 2023 one; every other unit is listed as
// unresolved in docs/LIVE_SYNC.md.

TEST_CASE("RoyaleAPI-derived first-hit values reach the spawned units", "[first_hit][first_hit_values][derived]") {
    struct Row { int id; bool evolved; int ticks; };
    const std::vector<Row> rows = {
        {0, false, 5}, {1, false, 5}, {2, false, 5}, {5, false, 5}, {9, false, 2},
        {10, false, 1}, {13, false, 5}, {14, false, 5}, {15, false, 6}, {17, false, 5},
        {19, false, 10}, {21, false, 4}, {22, false, 5}, {26, false, 4}, {34, false, 5},
        {35, false, 6}, {40, false, 10}, {43, false, 4}, {44, false, 3}, {45, false, 2},
        {47, false, 6}, {48, false, 5}, {50, false, 4}, {53, false, 1}, {58, false, 8},
        {59, false, 5}, {64, false, 7}, {68, false, 3}, {71, false, 7}, {77, false, 5},
        {82, false, 4}, {83, false, 2}, {88, false, 8}, {90, false, 8}, {113, false, 4},
        {116, false, 2}, {117, false, 3}, {118, false, 3}, {119, false, 2}, {123, false, 2},
        {123, true, 2}, {127, false, 2}, {127, true, 2}, {128, false, 5}, {128, true, 5},
        {130, false, 7}, {130, true, 7}, {135, false, 5}, {135, true, 5}, {136, false, 6},
        {136, true, 6}, {137, false, 3}, {137, true, 3}, {142, false, 7}, {142, true, 7},
        {147, false, 1}, {147, true, 1}, {148, false, 5}, {148, true, 5}, {150, false, 5},
        {150, true, 5}, {154, false, 4}, {154, true, 4}, {156, false, 4}, {156, true, 4},
        {159, false, 8}, {159, true, 8}, {160, false, 5}, {160, true, 5}, {162, false, 4},
        {162, true, 4}, {166, false, 5}, {169, false, 5}, {170, false, 5}, {173, false, 4},
        {175, false, 10},
    };
    for (const auto& r : rows) {
        INFO("card " << r.id << (r.evolved ? " (evolved)" : ""));
        Board board;
        auto body = live_sync::mainBody(board, r.id, r.evolved);
        REQUIRE(body != nullptr);
        CHECK(body->firstHitTicks == r.ticks);
    }
}

TEST_CASE("A Knight's first swing lands 0.5 s after he engages, then every 1.2 s", "[first_hit][first_hit_values][derived]") {
    Board board;
    auto target = std::make_shared<DummyEntity>(900, 5.0f, 6.0f, 100000, 1);
    spawn(board, target);
    auto knight = live_sync::mainBody(board, 0);
    REQUIRE(knight != nullptr);
    knight->position = Vector2D{ 5.0f, 5.0f };
    advancePastDeploy(knight, board);
    std::vector<int> hitTicks;
    int last = target->hp;
    for (int t = 0; t < 40; ++t) {
        knight->update(board);
        if (target->hp != last) { hitTicks.push_back(t); last = target->hp; }
    }
    REQUIRE(hitTicks.size() >= 3);
    CHECK(hitTicks[0] == 5);                  // armed on tick 0, lands 5 ticks later
    CHECK(hitTicks[1] - hitTicks[0] == 12);   // continuous attacks keep the 1.2 s hit speed
    CHECK(hitTicks[2] - hitTicks[1] == 12);
}
