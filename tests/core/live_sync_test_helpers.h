#pragma once
// Shared fixtures for test_live_sync.cpp, which pins every value the 2026-11
// live-game sync changed (docs/LIVE_SYNC.md). A case reads what the engine
// actually builds, by spawning the card, or measures what it does in a real
// match, never what the registry source appears to say.
#include <catch_amalgamated.hpp>
#include "test_helpers.h"
#include "AreaSpell.h"
#include "Board.h"
#include "Building.h"
#include "CardRegistry.h"
#include "CardStats.h"
#include "CombatEntity.h"
#include "GameManager.h"
#include "StatsEventBus.h"
#include "Tower.h"
#include "Troop.h"
#include <memory>
#include <set>
#include <vector>

namespace live_sync {

// The shipped 2.6 Hog Cycle: any legal deck works, since cases place units
// with spawnCard, not playCard.
inline const std::vector<int>& defaultDeck() {
    static const std::vector<int> d = { 15, 6, 25, 40, 24, 72, 33, 7 };
    return d;
}

struct DamageRecorder : IStatsObserver {
    std::vector<DamageDealtEvent> damage;
    void onDamageDealt(const DamageDealtEvent& e) override { damage.push_back(e); }

    // Total damage with amount > 0 dealt to `targetId` (a 0-damage spell still
    // emits events).
    int totalTo(int targetId) const {
        int sum = 0;
        for (const auto& e : damage)
            if (e.targetId == targetId && e.amount > 0) sum += e.amount;
        return sum;
    }
};

// A real match with every tower asleep (Tower::sleep: acquires nothing until
// damaged), so only the units a case places act.
struct Match {
    GameManager game;
    std::shared_ptr<DamageRecorder> rec = std::make_shared<DamageRecorder>();

    explicit Match(const std::vector<int>& deck = defaultDeck()) : game(deck, deck) {
        game.seed(1);
        game.reset();
        game.getBoard().statsEvents.subscribe(rec);
        for (const auto& e : game.getBoard().getEntities())
            if (auto t = std::dynamic_pointer_cast<Tower>(e)) t->sleep();
    }
    Board& board() { return game.getBoard(); }
    void step(int n = 1) { for (int i = 0; i < n; ++i) game.step(); }

    // Every entity the card put down (or its evolved form, for an Evolution id).
    std::vector<std::shared_ptr<Entity>> spawnCard(int cardId, float x, float y, int team = 0,
                                                   bool evolved = false) {
        std::set<int> before;
        for (const auto& e : board().getEntities()) before.insert(e->id);
        const CardDefinition* def = CardRegistry::getInstance().getCard(cardId);
        if (evolved) def->spawnEvolvedEntity(x, y, team, board());
        else def->spawnEntity(x, y, team, board());
        board().commitPendingEntities();
        std::vector<std::shared_ptr<Entity>> out;
        for (const auto& e : board().getEntities())
            if (!before.count(e->id)) out.push_back(e);
        return out;
    }
};

// Spawns a card on a bare Board (no towers, no match) and returns its bodies.
inline std::vector<std::shared_ptr<Entity>> spawnBare(Board& board, int cardId, float x = 9.0f,
                                                      float y = 10.0f, int team = 0, bool evolved = false) {
    std::set<int> before;
    for (const auto& e : board.getEntities()) before.insert(e->id);
    const CardDefinition* def = CardRegistry::getInstance().getCard(cardId);
    if (evolved) def->spawnEvolvedEntity(x, y, team, board);
    else def->spawnEntity(x, y, team, board);
    board.commitPendingEntities();
    std::vector<std::shared_ptr<Entity>> out;
    for (const auto& e : board.getEntities())
        if (!before.count(e->id)) out.push_back(e);
    return out;
}

// The first spawned body that is a CombatEntity, or nullptr.
inline std::shared_ptr<CombatEntity> firstCombat(const std::vector<std::shared_ptr<Entity>>& bodies) {
    for (const auto& e : bodies)
        if (auto c = std::dynamic_pointer_cast<CombatEntity>(e)) return c;
    return nullptr;
}

// The main body of a card: spawned on a bare board, first CombatEntity.
inline std::shared_ptr<CombatEntity> mainBody(Board& board, int cardId, bool evolved = false) {
    return firstCombat(spawnBare(board, cardId, 9.0f, 10.0f, 0, evolved));
}

} // namespace live_sync
