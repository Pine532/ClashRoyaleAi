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
        // Control: a registry Knight, which this change gives no value.
        Board board;
        CardRegistry::getInstance().getCard(0)->spawnEntity(5.0f, 5.0f, 0, board);
        board.commitPendingEntities();
        auto troop = std::dynamic_pointer_cast<CombatEntity>(board.getEntities().back());
        REQUIRE(troop);
        REQUIRE(troop->firstHitTicks == 0);
    }
}
