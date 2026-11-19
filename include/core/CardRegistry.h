#pragma once
#include <string>
#include <functional>
#include <unordered_map>
#include <vector>
#include <memory>
#include "Board.h"
#include "CardStats.h"
#include "CardFactories.h"
#include "FreezeOnHit.h"
#include "SpawnOnDeath.h"
#include "AreaDamageOnDeath.h"
#include "DelayedAreaDamageOnDeath.h"
#include "MoveSlowOnHit.h"
#include "CompositeDeathEffect.h"
#include "PeriodicSpawnEffect.h"
#include "CurseOnHit.h"
#include "AreaBuffOnDeath.h"
#include "AreaStunEffect.h"
#include "ElixirGrantEffect.h"
#include "EnemyElixirGrantOnDeath.h"
#include "SpawnOnDeathForEnemyTeam.h"
#include "ProximityGatedPeriodicSpawnEffect.h"
#include "CursedHogOnHit.h"
#include "MightyMinerEscapeEffect.h"
#include "GoldenKnightDashEffect.h"
#include "SkeletonKingSoulSummonEffect.h"
#include "ArcherQueenCloakEffect.h"
#include "MonkDeflectEffect.h"
#include "SpawnOnAbility.h"
#include "GoblinsteinLightningLinkEffect.h"
#include "BossBanditGetawayGrenadeEffect.h"
#include "CappedSpawnOnHitEffect.h"
#include "PoisonOnHit.h"
#include "PeriodicFreezeNearestEffect.h"
#include "DarkGuardOnDamageEffect.h"
#include "HeroMiniPekkaBoostEffect.h"
#include "HeroKnightTauntEffect.h"
#include "HeroWizardFieryFlightEffect.h"
#include "HeroGiantHurlEffect.h"
#include "HeroMegaMinionWarpEffect.h"
#include "HeroMagicArcherTripleThreatEffect.h"
#include "HeroIceGolemSnowstormEffect.h"
#include "HeroBarbarianBarrelRerollEffect.h"
#include "HeroValkyrieWhirlwindEffect.h"
#include "HeroBerserkerSavageSurvivalEffect.h"
#include "HeroIceWizardFrostyFellaEffect.h"
#include "EliteBarbarianSpearEffect.h"

// A registered card. spawnEntity is one of the archetype factories in
// CardFactories.h bound to the card's CardStats.
struct CardDefinition {
    int id;
    std::string name;
    float cost;
    bool isSpell;
    // True only for DefensiveBuilding cards. Towers are not registered, and
    // neither are spawned helper bodies, so "not in the registry" never means
    // "a tower" (see DamageDealtEvent::targetIsTower).
    bool isBuilding;
    // The footprint isValidPlacement keeps clear of existing buildings
    // (CardFactories::placementRadius). Unused for spells.
    float placementRadius;
    // Rolling-sweep shape (The Log, Barbarian Barrel; 0 otherwise), emitted in
    // GameLogger's cardMeta so the viewer, which cannot derive engine geometry,
    // draws it from the replay.
    float rollWidth;
    float rollRange;
    // Miner, Goblin Drill: placeable anywhere (GameManager::isValidPlacement).
    bool deployAnywhere;
    // Barbarian Barrel: a spell cast on its own side only, as a troop is
    // (GameManager::isValidPlacement). False for every other card.
    bool castOwnSideOnly = false;
    // Surfaced at registry level so deck contents can be inspected before
    // anything is placed.
    bool isChampion;
    // Champion-slot code checks `isChampion || isHero` throughout.
    bool isHero = false;
    // An unfinished entry: a Hero registered with its body only, because the
    // numbers its ability needs have no source (no invented values), or an entry
    // whose signature mechanic is only partly built (Elite Barbarians Evolution
    // 182, Hero Ice Wizard 184, Electro Giant Evolution 186). sampleRandomDeck
    // never deals such a card, so random-deck training does not meet a card
    // that cannot do what the live one does. One flag covers both kinds. Set by
    // CardRegistry::markAbilityPending.
    bool abilityPending = false;
    // At registry level because Hero Goblins' post-death ability fires when
    // nothing is alive to read them from. 0 / nullptr for every card without a
    // post-death ability.
    float abilityElixirCost = 0.0f;
    int abilityUsableAfterDeathTicks = 0;
    std::shared_ptr<IPeriodicEffect> postDeathAbilityEffect;
    // Display metadata for GameLogger's cardMeta block, not used by gameplay
    // (spawnEntity captures CardStats). hp is 0 for spells.
    int hp = 0;
    char symbol = '?';
    bool isFlying = false;
    std::function<void(float x, float y, int team, Board& board)> spawnEntity;

    // Evolution slots (addEvolution, PlayerState::playCard). Cost and placement
    // come from the base stats, since an Evolution changes neither;
    // spawnEvolvedEntity is the only extra.
    bool isEvolution = false;
    int evolutionCycleThreshold = 0;
    int evolvedUsesGranted = 0;
    // For an Evolution def: the base card's id (CardStats::id of the baseStats
    // handed to addEvolution). -1 for every other card. Used by Evolved Mirror.
    int evolvesFromId = -1;
    std::function<void(float x, float y, int team, Board& board)> spawnEvolvedEntity;
};

// Exclusive upper bound on card ids: [0, CARD_ID_COUNT). Defined here, next to
// the ids it bounds, so ClashEnv (NUM_CARD_IDS, bound to Python) and
// GameManager share one definition. It sizes the observation (4 hand one-hots
// and the 2 opponent-cycle blocks) and the network's card-id layers, so a change
// breaks every checkpoint (python_ai/tools/migrate_checkpoint_card_ids.py pads
// one forward). Raised 185 -> 256 for the 2026-11 live sync, which needed 13
// new ids where 9 were free; the rest is headroom for later releases
// (docs/LIVE_SYNC.md).
inline constexpr int CARD_ID_COUNT = 256;

class CardRegistry {
private:
    std::unordered_map<int, CardDefinition> cards;
    // base card id -> Evolution id, filled by addEvolution. Lookup only, never
    // iterated, so determinism is untouched.
    std::unordered_map<int, int> evolutionIdByBase;

    static CardStats troop(int id, std::string name, float cost, Archetype archetype,
        int hp, float speed, float attackRange, int damage, int attackCooldown, char symbol) {
        CardStats s;
        s.id = id; s.name = std::move(name); s.cost = cost; s.archetype = archetype;
        // `speed` arrives in the engine's original units; MOVEMENT_SPEED_SCALE
        // converts it (CardStats.h).
        s.hp = hp; s.speed = speed * MOVEMENT_SPEED_SCALE; s.attackRange = attackRange;
        s.damage = damage; s.attackCooldown = attackCooldown; s.symbol = symbol;
        return s;
    }

    static CardStats building(int id, std::string name, float cost,
        int hp, char symbol, float attackRange, int damage, int attackCooldown) {
        CardStats s;
        s.id = id; s.name = std::move(name); s.cost = cost; s.archetype = Archetype::DefensiveBuilding;
        s.hp = hp; s.symbol = symbol; s.attackRange = attackRange;
        s.damage = damage; s.attackCooldown = attackCooldown;
        return s;
    }

    static CardStats spell(int id, std::string name, float cost,
        float radius, int damage, int delayTicks, char symbol) {
        CardStats s;
        s.id = id; s.name = std::move(name); s.cost = cost; s.archetype = Archetype::Spell;
        s.spellRadius = radius; s.damage = damage; s.spellDelayTicks = delayTicks; s.symbol = symbol;
        return s;
    }

    // Skeleton Army's 15 skeletons, on a 4-wide grid spaced 0.6.
    static std::vector<Vector2D> skeletonArmyOffsets() {
        std::vector<Vector2D> offsets;
        for (int i = 0; i < 15; ++i) {
            float ox = (static_cast<float>(i % 4) - 1.5f) * 0.6f;
            float oy = (static_cast<float>(i / 4) - 1.0f) * 0.6f;
            offsets.push_back({ ox, oy });
        }
        return offsets;
    }
    // Skeleton Army Evolution: the same grid plus its last cell, 16 total (the
    // 15 plus the Skeleton General).
    static std::vector<Vector2D> skeletonArmyEvolvedOffsets() {
        std::vector<Vector2D> offsets = skeletonArmyOffsets();
        offsets.push_back({ 0.9f, 1.2f });
        return offsets;
    }

    // Golemite: spawned only by a Golem's death, never registered.
    static CardStats golemiteStats() {
        // SPEED_SLOW, the tier of the Golem it splits from.
        // First hit 1.0 s: RoyaleAPI cr-api-data, hit_speed 2500 - load_time 1500 ms (docs/LIVE_SYNC.md).
        return troop(-1, "Golemite", 0.0f, Archetype::MeleeBuildingTargeter, 1039, SPEED_SLOW, 0.25f, 84, 25, 'q').withFirstHit(10)
            .withOffsets({ {-0.3f, 0.0f}, {0.3f, 0.0f} }).withSightRange(7.0f);
    }

    // Child units spawned by death, periodic and spell effects. They reuse the
    // stats of an already-registered card where one exists, and use negative
    // ids clear of -1 (Golemite) and the tower ids -2/-3.
    //
    // A child whose unit is also a playable card takes that card's SPEED_*
    // tier, never a literal; pinned by "every spawned unit moves at the speed
    // of its own playable card" in tests/core/test_card_registry.cpp.
    static CardStats battleRamBarbarianStats() {
        return troop(-10, "Barbarians", 0.0f, Archetype::MeleeSquad, 716, SPEED_MEDIUM, 0.7f, 192, 14, 'B')
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} });
    }
    static CardStats skeletonBarrelSkeletonStats() {
        // First hit 0.5 s (5 ticks): Skeletons, official Nov 2023 post implying base 0.5 (Evolution 0.5 -> 0.4): https://supercell.com/en/games/clashroyale/blog/release-notes/balance-changes-nov-2023 ; Skeleton Army skeletons 0.5 s: https://liquipedia.net/clashroyale/Skeleton_Army ; Tombstone skeletons 0.5 s: https://theriagames.com/guide/clash-royale-tombstone-guide/
        return troop(-11, "Skeletons", 0.0f, Archetype::MeleeSquad, 81, SPEED_FAST, 0.5f, 81, 11, 'k').withFirstHit(5)
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} });
    }
    // The death spawn is one Bat, not three.
    static CardStats nightWitchBatStats() {
        // SPEED_VERY_FAST, matching playable card 78 (the same unit).
        return troop(-12, "Bats", 0.0f, Archetype::MeleeSquad, 81, SPEED_VERY_FAST, 0.5f, 81, 12, 't')
            .withFlying().withTargetsAir();
    }

    // Periodic-spawn children (Witch, Night Witch, Furnace, Barbarian Hut,
    // Goblin Hut, Goblin Drill). Goblin Hut's count and the intervals are not
    // sourced.
    static CardStats witchSkeletonStats() {
        // First hit 0.5 s (5 ticks): Skeletons, official Nov 2023 post implying base 0.5 (Evolution 0.5 -> 0.4): https://supercell.com/en/games/clashroyale/blog/release-notes/balance-changes-nov-2023 ; Skeleton Army skeletons 0.5 s: https://liquipedia.net/clashroyale/Skeleton_Army ; Tombstone skeletons 0.5 s: https://theriagames.com/guide/clash-royale-tombstone-guide/
        return troop(-13, "Skeletons", 0.0f, Archetype::MeleeSquad, 81, SPEED_FAST, 0.5f, 81, 11, 'k').withFirstHit(5)
            .withOffsets({ {-0.4f, -0.4f}, {0.4f, -0.4f}, {-0.4f, 0.4f}, {0.4f, 0.4f} });
    }
    static CardStats nightWitchPeriodicBatStats() {
        // SPEED_VERY_FAST, as the -12 Bats.
        return troop(-14, "Bats", 0.0f, Archetype::MeleeSquad, 81, SPEED_VERY_FAST, 0.5f, 81, 12, 't')
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} })
            .withFlying().withTargetsAir();
    }
    // The Furnace's spirit is the playable Fire Spirit (card 73): one kamikaze
    // hit of area damage, then gone. Without withDieAfterFirstHit it was a
    // permanent 207-per-second turret (UPSTREAM_REQUESTS.md item 32b).
    static CardStats furnaceFireSpiritStats() {
        return troop(-15, "Fire Spirit", 0.0f, Archetype::RangedSquad, 215, SPEED_VERY_FAST, 2.5f, 215, 10, '<')
            .withTargetsAir()
            .withSplash(2.3f)
            .withDieAfterFirstHit();
    }
    static CardStats barbarianHutBarbarianStats() {
        return troop(-16, "Barbarians", 0.0f, Archetype::MeleeSquad, 716, SPEED_MEDIUM, 0.7f, 192, 14, 'B')
            .withOffsets({ {0.0f, 0.0f}, {-0.5f, -0.5f}, {0.5f, -0.5f} });
    }
    // The playable Spear Goblins (card 23), air included (UPSTREAM_REQUESTS.md
    // item 30), deploying in 0.5 s rather than 1 (official August 2025 notes:
    // "Spear Goblin Deploy Delay: 1sec -> 0.5sec").
    static CardStats goblinHutSpearGoblinStats() {
        // First hit 0.5 s: official balance notes (inferred: the same Spear Goblin character) (https://supercell.com/en/games/clashroyale/blog/release-notes/august-balance-changes-2/).
        return troop(-17, "Spear Goblins", 0.0f, Archetype::RangedSquad, 133, SPEED_VERY_FAST, 5.0f, 81, 16, 'S').withFirstHit(5)
            .withTargetsAir()
            .withDeployTime(5);
    }
    // 2026-08-04 balance update: the Drill's spawned Goblins deal 125 damage
    // (+4%, the Goblins card's 120 -> 125; docs/LIVE_SYNC.md). The post gives
    // no separate figure, so this reuses the Goblins line. This builder also
    // feeds the Evolved Mortar and Goblin Giant evolution spawns, which keep
    // the same Goblins stats.
    static CardStats goblinDrillGoblinStats() {
        // First hit 0.4 s: RoyaleAPI cr-api-data, hit_speed 1100 - load_time 700 ms; the
        // April 2025 Goblins change (0.4 -> 0.6 s) excluded Drill-spawned Goblins.
        return troop(-18, "Goblins", 0.0f, Archetype::MeleeSquad, 202, SPEED_VERY_FAST, 0.5f, 125, 11, 'g').withFirstHit(4);
    }
    // Phoenix's one-time revive: a second Phoenix with no death effect of its
    // own. The real card leaves a vulnerable egg for ~4 s; this revives
    // immediately at full hp.
    static CardStats phoenixReviveStats() {
        // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1000 - load_time 500 ms (docs/LIVE_SYNC.md).
        return troop(-19, "Phoenix", 0.0f, Archetype::MeleeSquad, 1052, SPEED_MEDIUM, 1.0f, 217, 10, '7').withFirstHit(5)
            .withFlying().withTargetsAir();
    }
    // Spell-spawn children (Goblin Barrel, Royal Delivery, Graveyard).
    static CardStats goblinBarrelGoblinStats() {
        // First hit 0.3 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/october-balance-changes-2026/).
        return troop(-20, "Goblins", 0.0f, Archetype::MeleeSquad, 202, SPEED_VERY_FAST, 0.5f, 120, 11, 'g').withFirstHit(3)
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f}, {0.0f, 0.4f} });
    }
    static CardStats graveyardSkeletonStats() {
        // First hit 0.5 s (5 ticks): Skeletons, official Nov 2023 post implying base 0.5 (Evolution 0.5 -> 0.4): https://supercell.com/en/games/clashroyale/blog/release-notes/balance-changes-nov-2023 ; Skeleton Army skeletons 0.5 s: https://liquipedia.net/clashroyale/Skeleton_Army ; Tombstone skeletons 0.5 s: https://theriagames.com/guide/clash-royale-tombstone-guide/
        return troop(-21, "Skeletons", 0.0f, Archetype::MeleeSquad, 81, SPEED_FAST, 0.5f, 81, 11, 'k').withFirstHit(5);
    }
    static CardStats royalDeliveryRecruitStats() {
        // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1300 - load_time 800 ms (docs/LIVE_SYNC.md).
        return troop(-22, "Royal Recruits", 0.0f, Archetype::MeleeSquad, 547, SPEED_MEDIUM, 1.0f, 133, 13, '@').withFirstHit(5)
            .withShield(240); // Royal Recruits' shield
    }
    static CardStats barbarianBarrelBarbarianStats() {
        return troop(-23, "Barbarians", 0.0f, Archetype::MeleeSquad, 716, SPEED_MEDIUM, 0.7f, 192, 14, 'B');
    }
    // Hero Barbarian Barrel makes the spawned Barbarian the Hero (the barrel
    // spell has no persistent entity): barbarianBarrelBarbarianStats plus
    // "Rowdy Reroll" -- a 3-tile reroll (May 2026) down the barrel's own 2.6
    // corridor, 215 like the barrel (2026-10-06 balance update, 232 -> 215;
    // the post is silent on the reroll, so it follows the barrel; docs/LIVE_SYNC.md),
    // healing half the damage it deals.
    // withAbilitySlotCard(174): this body spawns ~17 ticks after the play, as
    // a -47 helper, so the slot must be told whose ability it carries.
    static CardStats heroBarbarianBarrelBarbarianStats() {
        return troop(-47, "Hero Barbarian Barrel", 0.0f, Archetype::MeleeSquad, 716, SPEED_MEDIUM, 0.7f, 192, 14, 'B')
            .withHeroAbility(1.0f, 0, std::make_shared<HeroBarbarianBarrelRerollEffect>(3.0f, 1.3f, 215, 716), 1)
            .withAbilitySlotCard(174);
    }
    // Compound-card secondary units (Goblin Machine's turret, Ram Rider's
    // crossbow, Goblin Giant's Spear Goblins, the ranged halves of Goblin Gang
    // and Rascals), spawned via CardStats::secondaryUnit at the same point and
    // targeting independently.
    //
    // Goblin Machine's rocket: 304 per shot on a 50-tick cooldown (DPS ~61),
    // against the arms' 232 / 12 ticks. 2026-08-04 balance update: hit speed
    // 3.5 s -> 5 s and hp 2150 -> 2265 (docs/LIVE_SYNC.md).
    static CardStats goblinMachineTurretStats() {
        return troop(-24, "Goblin Machine", 0.0f, Archetype::RangedSquad, 2265, 0.5f, 5.0f, 304, 50, '3')
            .withSplash(1.5f).withTargetsAir();
    }
    // The crossbow: 104 per shot on 11 ticks (DPS 94), against the Ram's 250 /
    // 17. withIgnoresRiver matches the Ram, or the crossbow is left on the near
    // bank.
    static CardStats ramRiderCrossbowStats() {
        // First hit 0.4 s: RoyaleAPI cr-api-data, hit_speed 1100 - load_time 700 ms (docs/LIVE_SYNC.md).
        return troop(-25, "Ram Rider", 0.0f, Archetype::RangedSquad, 1766, 0.5f, 5.0f, 104, 11, '"').withFirstHit(4)
            .withTargetsAir()
            .withIgnoresRiver();
    }
    static CardStats goblinGiantSpearGoblinsStats() {
        // First hit 0.5 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/october-balance-changes-2/).
        return troop(-26, "Spear Goblins", 0.0f, Archetype::RangedSquad, 133, SPEED_VERY_FAST, 5.0f, 81, 16, 'S').withFirstHit(5)
            .withOffsets({ {-0.4f, 0.3f}, {0.4f, 0.3f} }).withTargetsAir();
    }

    // Further death-spawn children, from a later research pass. Ids continue
    // from -29.
    static CardStats lavaHoundPupStats() {
        // 2025-02-04 balance update: Lava Pups damage 90 -> 84 (docs/LIVE_SYNC.md).
        // 215 hp: official December 2025 balance notes, 217 -> 215 (level 11).
        // First hit 1.0 s: RoyaleAPI cr-api-data, hit_speed 1700 - load_time 700 ms (docs/LIVE_SYNC.md).
        return troop(-29, "Lava Pups", 0.0f, Archetype::MeleeSquad, 215, 0.5f, 1.6f, 84, 17, 'y').withFirstHit(10)
            .withOffsets({ {-0.8f, 0.0f}, {0.8f, 0.0f}, {-0.5f, 0.7f}, {0.5f, 0.7f}, {-0.5f, -0.7f}, {0.5f, -0.7f} })
            .withFlying().withTargetsAir();
    }
    static CardStats suspiciousBushGoblinStats() {
        return troop(-30, "Bush Goblins", 0.0f, Archetype::MeleeSquad, 304, 0.5f, 0.8f, 256, 14, 'g')
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} });
    }
    static CardStats goblinCageBrawlerStats() {
        // 2026-09-08 balance update (supercell.com September balance changes, docs/LIVE_SYNC.md): hitpoints 1080 -> 1121.
        // First hit 0.2 s: RoyaleAPI cr-api-data, hit_speed 1100 - load_time 900 ms (docs/LIVE_SYNC.md).
        return troop(-31, "Goblin Brawler", 0.0f, Archetype::MeleeSquad, 1121, 0.7f, 0.8f, 337, 11, 'o').withFirstHit(2);
    }
    // Elixir Golem's split chain: Golem -> 2 Golemites -> 2 Blobs each. Each
    // tier spawns the next and grants the OPPONENT elixir: 1.0 from the Golem,
    // 0.5 from each Golemite and each Blob, 4.0 in total. The Blob is the base
    // case.
    static CardStats elixirBlobStats() {
        // First hit 0.8 s: RoyaleAPI cr-api-data, hit_speed 1100 - load_time 300 ms (docs/LIVE_SYNC.md).
        return troop(-33, "Elixir Blob", 0.0f, Archetype::MeleeBuildingTargeter, 360, 0.7f, 0.5f, 64, 11, 'e').withFirstHit(8)
            .withDeathEffect(std::make_shared<EnemyElixirGrantOnDeath>(0.5f)).withSightRange(7.5f);
    }
    static CardStats elixirGolemiteStats() {
        // First hit 0.8 s: RoyaleAPI cr-api-data, hit_speed 1100 - load_time 300 ms (docs/LIVE_SYNC.md).
        return troop(-32, "Elixir Golemite", 0.0f, Archetype::MeleeBuildingTargeter, 762, 0.5f, 1.0f, 128, 11, 'e').withFirstHit(8)
            .withDeathEffect(std::make_shared<CompositeDeathEffect>(
                std::vector<std::shared_ptr<IDeathEffect>>{
                    std::make_shared<SpawnOnDeath>(
                        elixirBlobStats().withOffsets({ {-0.3f, 0.0f}, {0.3f, 0.0f} })),
                    std::make_shared<EnemyElixirGrantOnDeath>(0.5f)
                })).withSightRange(7.5f);
    }
    // Mother Witch's Cursed Hog, spawned for the opposite team via
    // SpawnOnDeathForEnemyTeam (see CursedHogOnHit).
    static CardStats cursedHogStats() {
        // First hit 0.3 s: RoyaleAPI cr-api-data, hit_speed 1200 - load_time 950 ms (docs/LIVE_SYNC.md).
        return troop(-34, "Cursed Hog", 0.0f, Archetype::MeleeBuildingTargeter, 629, 1.0f, 0.75f, 53, 12, 'h').withFirstHit(3);
    }
    // The single Barbarian released when the Hut dies; the periodic spawn's
    // stats.
    static CardStats barbarianHutDeathBarbarianStats() {
        return troop(-35, "Barbarians", 0.0f, Archetype::MeleeSquad, 716, SPEED_MEDIUM, 0.7f, 192, 14, 'B');
    }
    // Goblin Drill's death spawn: two Goblins (three before a 2021 balance
    // patch).
    // 2026-08-04 balance update: 125 damage, as the Goblins card (docs/LIVE_SYNC.md).
    static CardStats goblinDrillDeathGoblinStats() {
        // First hit 0.4 s: RoyaleAPI cr-api-data, hit_speed 1100 - load_time 700 ms; the
        // April 2025 Goblins change (0.4 -> 0.6 s) excluded Drill-spawned Goblins.
        return troop(-36, "Goblins", 0.0f, Archetype::MeleeSquad, 202, SPEED_VERY_FAST, 0.5f, 125, 11, 'g').withFirstHit(4)
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} });
    }
    // Goblin Demolisher's transformed form: a Very Fast, building-only kamikaze
    // with a 2.5-radius, 404-damage burst on its first hit, spawned when the
    // ranged form kills itself at <= 50% hp (CombatEntity::transformKillsSelf).
    // Reuses the ranged form's 1300 hp (not sourced). 2025-06-02 balance update:
    // lifetime after 50% HP 8 s -> 10 s (100 ticks), modelled as a fraction-1.0
    // withHpTransform timer that runs from the kamikaze's spawn (docs/LIVE_SYNC.md).
    static CardStats goblinDemolisherKamikazeStats() {
        return troop(-37, "Kamikaze Goblin Demolisher", 0.0f, Archetype::MeleeBuildingTargeter,
            1300, 1.0f, 0.5f, 404, 10, '&')
            .withSplash(2.5f).withDieAfterFirstHit().withHpTransform(1.0f, 100, false);
    }
    // Skeleton King's summoned skeletons, spawned one at a time in a loop by
    // SkeletonKingSoulSummonEffect.
    static CardStats skeletonKingSummonedSkeletonStats() {
        // First hit 0.5 s (5 ticks): Skeletons, official Nov 2023 post implying base 0.5 (Evolution 0.5 -> 0.4): https://supercell.com/en/games/clashroyale/blog/release-notes/balance-changes-nov-2023 ; Skeleton Army skeletons 0.5 s: https://liquipedia.net/clashroyale/Skeleton_Army ; Tombstone skeletons 0.5 s: https://theriagames.com/guide/clash-royale-tombstone-guide/
        return troop(-38, "Skeletons", 0.0f, Archetype::MeleeSquad, 81, SPEED_FAST, 0.5f, 81, 11, 'k').withFirstHit(5);
    }
    // Little Prince's Royal Rescue: the Guardienne. Her spawn-in dash and
    // knockback are not modelled. 2026-08-04 balance update (final August changes, supercell.com): damage 217 -> 232
    // (docs/LIVE_SYNC.md).
    static CardStats guardienneStats() {
        return troop(-39, "Guardienne", 0.0f, Archetype::MeleeSquad, 1600, 0.5f, 1.2f, 232, 12, 'u')
            .withCharge(3.0f, 2.0f);
    }
    // Hero Musketeer's "Trusty Turret": a short-range turret that
    // self-destructs after 10 s. withHpTransform with a fraction of exactly 1.0
    // makes the HP-threshold mechanism a plain lifetime timer.
    // becomesStationary stays false: the archetype is already stationary, and
    // that branch's freeze would silence the turret. hp and damage are not
    // sourced.
    static CardStats heroMusketeerTurretStats() {
        return building(-45, "Trusty Turret", 0.0f, 200, 't', 4.0f, 90, 5)
            .withTargetsAir()
            .withHpTransform(1.0f, 100, false);
    }
    // Hero Magic Archer's decoy: soaks hits and draws aggro but deals no
    // damage; a zero-damage, stationary unit. hp: 2026-07-06 balance update, 627
    // -> 271 (docs/LIVE_SYNC.md). Lifetime is not sourced.
    static CardStats heroMagicArcherDecoyStats() {
        return troop(-46, "Decoy", 0.0f, Archetype::MeleeSquad, 271, 0.0f, 1.0f, 0, 100, 'd');
    }
    // Lumberjack Evolution's death-spawn ghost. The real one is invincible
    // while inside its own Rage zone; here it is just invisible.
    static CardStats lumberjackGhostStats() {
        return troop(-44, "Lumberjack Ghost", 0.0f, Archetype::MeleeSquad, 400, 1.0f, 0.7f, 256, 8, 'l')
            .withInvisibility(10);
    }
    static CardStats runnerStats() {
        // Wall Breakers Evolution's Runner. hp scaled to level 11 from level-6
        // data (103 * 1.1^5 ~= 166); damage 184 is the official 2025-11-03
        // balance update ("Damage (Runner): 197 -> 184", docs/LIVE_SYNC.md).
        // Modelled as an ordinary building-targeter that keeps
        // attacking, not a single detonation, and without the real reduced
        // damage against Crown Towers.
        return troop(-41, "Runner", 0.0f, Archetype::MeleeBuildingTargeter, 166, 1.2f, 0.6f, 184, 8, 'g');
    }
    // Battle Ram Evolution's death spawn: a stronger pair of Barbarians.
    static CardStats battleRamEvolvedBarbarianStats() {
        return troop(-10, "Barbarians", 0.0f, Archetype::MeleeSquad, 830, SPEED_MEDIUM, 0.7f, 230, 14, 'B')
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} });
    }
    // Evolved Skeletons' "Never-ending Horde" child. Reuses id 24 on purpose:
    // CappedSpawnOnHitEffect counts by cardId, so the three originals and every
    // child form one population capped at the sourced 8. Children carry no
    // spawn-on-hit of their own, so growth has at most three sources.
    static CardStats evolvedSkeletonChildStats() {
        // First hit 0.4 s (4 ticks), official: https://supercell.com/en/games/clashroyale/blog/release-notes/balance-changes-nov-2023
        return troop(24, "Skeletons", 0.0f, Archetype::MeleeSquad, 81, SPEED_FAST, 0.5f, 81, 11, 'k').withFirstHit(4);
    }
    // Skeleton Army Evolution's death-spawn shadow. The real shadows have
    // unlimited hp and are targetable only by spells; here a targetable 500-hp
    // unit. 2026-01-12 balance update: "Ghost Skeletons' Speed: Fast -> Medium"
    // (docs/LIVE_SYNC.md); reading "Ghost Skeletons" as these shadows is an
    // inference, the post does not define the term.
    static CardStats skeletonArmyShadowStats() {
        return troop(-43, "Skeleton Shadow", 0.0f, Archetype::MeleeSquad, 500, SPEED_MEDIUM, 0.5f, 81, 11, 'k');
    }
    // Royal Ghost Evolution's on-hit Souldier: 81 hp. Damage 81: 2025-12-01
    // balance update 133 -> 102, then April 2026 balance update (2026-04-06)
    // 102 -> 81 (-20%) (docs/LIVE_SYNC.md).
    static CardStats royalGhostSouldierStats() {
        return troop(-42, "Souldier", 0.0f, Archetype::MeleeSquad, 81, 0.7f, 1.2f, 81, 18, 'Q');
    }

    void add(const CardStats& stats) {
        CardDefinition def;
        def.id = stats.id;
        def.name = stats.name;
        def.cost = stats.cost;
        def.isSpell = (stats.archetype == Archetype::Spell);
        def.isBuilding = (stats.archetype == Archetype::DefensiveBuilding);
        def.placementRadius = CardFactories::placementRadius(stats.archetype);
        def.rollWidth = stats.spellRollWidth;
        def.rollRange = stats.spellRollRange;
        def.deployAnywhere = stats.deployAnywhere;
        def.castOwnSideOnly = stats.spellOwnSideOnly;
        def.isChampion = stats.isChampion;
        def.isHero = stats.isHero;
        def.abilityElixirCost = stats.abilityElixirCost;
        def.abilityUsableAfterDeathTicks = stats.abilityUsableAfterDeathTicks;
        def.postDeathAbilityEffect = stats.postDeathAbilityEffect;
        def.hp = stats.hp;
        def.symbol = stats.symbol;
        def.isFlying = stats.isFlying;
        def.spawnEntity = [stats](float x, float y, int team, Board& board) {
            CardFactories::spawn(stats, x, y, team, board);
        };
        cards[stats.id] = std::move(def);
    }

    // Flags an already-added card as unfinished (CardDefinition::abilityPending).
    void markAbilityPending(int id) { cards.at(id).abilityPending = true; }

    // Registers an Evolution slot: `evolutionId` is what a deck holds instead
    // of the base card's id. Cost, spell/building flags, footprint and
    // deployAnywhere come from baseStats, since an Evolution changes only
    // combat behaviour. The slot is played cycleThreshold times un-evolved,
    // then evolvedUses times evolved, repeating all match (see
    // PlayerState::EvolutionSlotState).
    void addEvolution(int evolutionId, CardStats baseStats, CardStats evolvedStats,
            int cycleThreshold, int evolvedUses) {
        CardDefinition def;
        def.id = evolutionId;
        def.name = baseStats.name;
        def.cost = baseStats.cost;
        def.isSpell = (baseStats.archetype == Archetype::Spell);
        def.isBuilding = (baseStats.archetype == Archetype::DefensiveBuilding);
        def.placementRadius = CardFactories::placementRadius(baseStats.archetype);
        def.rollWidth = baseStats.spellRollWidth;
        def.rollRange = baseStats.spellRollRange;
        def.deployAnywhere = baseStats.deployAnywhere;
        def.castOwnSideOnly = baseStats.spellOwnSideOnly;
        def.isChampion = baseStats.isChampion;
        def.isHero = baseStats.isHero;
        def.abilityElixirCost = baseStats.abilityElixirCost;
        def.abilityUsableAfterDeathTicks = baseStats.abilityUsableAfterDeathTicks;
        def.postDeathAbilityEffect = baseStats.postDeathAbilityEffect;
        def.hp = baseStats.hp;
        def.symbol = baseStats.symbol;
        def.isFlying = baseStats.isFlying;
        def.spawnEntity = [baseStats](float x, float y, int team, Board& board) {
            CardFactories::spawn(baseStats, x, y, team, board);
        };
        def.isEvolution = true;
        def.evolutionCycleThreshold = cycleThreshold;
        def.evolvedUsesGranted = evolvedUses;
        def.spawnEvolvedEntity = [evolvedStats](float x, float y, int team, Board& board) {
            CardFactories::spawn(evolvedStats, x, y, team, board);
        };
        // Mirror Evolution (188) is a stub whose "base" is itself: no base card
        // maps to it, so it is left out of the lookup.
        if (baseStats.id != evolutionId) {
            def.evolvesFromId = baseStats.id;
            evolutionIdByBase[baseStats.id] = evolutionId;
        }
        cards[evolutionId] = std::move(def);
    }

    CardRegistry() {
        // Stats are level-11 tournament-standard data (cost, hp, damage, hit
        // speed, range, count). Speeds are the SPEED_* tiers (CardStats.h).

        // === Melee troops ===
        // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1200 - load_time 700 ms (docs/LIVE_SYNC.md).
        add(troop(0, "Knight", 3.0f, Archetype::MeleeSquad, 1766, SPEED_MEDIUM, 1.2f, 202, 12, 'K').withFirstHit(5));

        // 2026-08-04 balance update: Goblin damage 120 -> 125 (docs/LIVE_SYNC.md).
        // First hit 0.6 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/april-balance-changes/).
        add(troop(4, "Goblins", 2.0f, Archetype::MeleeSquad, 202, SPEED_VERY_FAST, 0.5f, 125, 11, 'g').withFirstHit(6)
            .withOffsets({ {-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, 0.5f} }));

        // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1600 - load_time 1100 ms (docs/LIVE_SYNC.md).
        add(troop(5, "Mini PEKKA", 4.0f, Archetype::MeleeSquad, 1390, SPEED_FAST, 0.8f, 755, 16, 'M').withFirstHit(5));

        add(troop(8, "Barbarians", 5.0f, Archetype::MeleeSquad, 716, SPEED_MEDIUM, 0.7f, 192, 14, 'B')
            .withOffsets({ {0.0f, 0.0f}, {-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, 0.5f} }));

        // First hit 0.1 s: RoyaleAPI cr-api-data, hit_speed 1500 - load_time 1400 ms (docs/LIVE_SYNC.md).
        add(troop(10, "Valkyrie", 4.0f, Archetype::MeleeSquad, 1907, SPEED_MEDIUM, 1.2f, 266, 15, 'V').withFirstHit(1)
            .withSplash(1.5f)); // 360-degree swing

        add(troop(12, "Skeleton Army", 3.0f, Archetype::MeleeSquad, 81, SPEED_FAST, 0.5f, 81, 11, 's')
            .withOffsets(skeletonArmyOffsets()));

        // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1800 - load_time 1300 ms (docs/LIVE_SYNC.md).
        add(troop(13, "P.E.K.K.A.", 7.0f, Archetype::MeleeSquad, 3760, SPEED_SLOW, 1.2f, 842, 18, 'E').withFirstHit(5).withSightRange(5.0f));

        // Prince jumps the river (withIgnoresRiver). The charge distance is
        // 2.5 tiles (official 24 Oct 2025 balance post, 2 -> 2.5; docs/LIVE_SYNC.md);
        // the damage is not sourced; the charge speed is (2x, the game files'
        // charge_speed_multiplier 200; UPSTREAM_REQUESTS.md item 35).
        // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1400 - load_time 900 ms (docs/LIVE_SYNC.md).
        add(troop(14, "Prince", 5.0f, Archetype::MeleeSquad, 1920, SPEED_MEDIUM, 1.6f, 391, 14, 'p').withFirstHit(5)
            .withCharge(2.5f, 2.0f).withChargeSpeed(2.0f)
            .withIgnoresRiver());

        // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1400 - load_time 900 ms (docs/LIVE_SYNC.md).
        add(troop(17, "Elite Barbarians", 6.0f, Archetype::MeleeSquad, 1341, SPEED_FAST, 1.2f, 384, 14, 'e').withFirstHit(5)
            .withOffsets({ {-0.5f, 0.0f}, {0.5f, 0.0f} }));

        // The death-potion radius and duration are not sourced. The Rage boost
        // is 30% (official 24 Oct 2025 balance post, 35% -> 30%; docs/LIVE_SYNC.md),
        // so the damage multiplier is 1.30 (the previous 1.75 matched no source).
        // First hit 0.4 s: RoyaleAPI cr-api-data, hit_speed 800 - load_time 400 ms (docs/LIVE_SYNC.md).
        add(troop(21, "Lumberjack", 4.0f, Archetype::MeleeSquad, 1282, SPEED_VERY_FAST, 0.7f, 256, 8, 'l').withFirstHit(4)
            .withDeathEffect(std::make_shared<AreaBuffOnDeath>(2.5f, 1.30f, 55)));

        // First hit 0.5 s (5 ticks): Skeletons, official Nov 2023 post implying base 0.5 (Evolution 0.5 -> 0.4): https://supercell.com/en/games/clashroyale/blog/release-notes/balance-changes-nov-2023 ; Skeleton Army skeletons 0.5 s: https://liquipedia.net/clashroyale/Skeleton_Army ; Tombstone skeletons 0.5 s: https://theriagames.com/guide/clash-royale-tombstone-guide/
        add(troop(24, "Skeletons", 1.0f, Archetype::MeleeSquad, 81, SPEED_FAST, 0.5f, 81, 11, 'k').withFirstHit(5)
            .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f} }));

        // Giant Skeleton, after the 2026-05-04 rework: hp -7% (3361 x 0.93),
        // and a death bomb that lands where he dies and explodes 3.0 s later
        // (fuse 29: see DelayedAreaDamageOnDeath) in a 3-tile radius, for
        // 688 x 1.29 ~= 886, the same against a Crown Tower since the rework
        // removed its 2x. The bomb's knockback (1.0, Fireball's) is not sourced.
        // UPSTREAM_REQUESTS.md item 32c.
        // First hit 0.3 s (3 ticks): https://liquipedia.net/clashroyale/Giant_Skeleton ; https://clashroyale.fandom.com/wiki/Giant_Skeleton
        add(troop(39, "Giant Skeleton", 6.0f, Archetype::MeleeSquad, 3126, SPEED_MEDIUM, 0.8f, 276, 13, 'J').withFirstHit(3)
            .withDeathEffect(std::make_shared<DelayedAreaDamageOnDeath>(3.0f, 886, 29, 1.0f, 39, "Giant Skeleton Bomb"))
            .withSightRange(5.0f));

        // Electro Wizard: the zap has no travel time, so the archetype is
        // MeleeSquad (archetypes describe how damage lands here, not range).
        // Splits to the 2 closest enemies (each takes damage/2; full damage if
        // only one is in range). Each hit and the deploy zap (radius 3) stun
        // for 0.5 s. 2025-03-04 balance update: attack damage 110 -> 116
        // (supercell.com march-balance-changes; the earlier 118 was unsourced).
        // The deploy zap keeps its own 118: no source states a deploy-zap
        // damage change.
        // First hit 0.6 s: RoyaleAPI cr-api-data, hit_speed 1800 - load_time 1200 ms (docs/LIVE_SYNC.md).
        add(troop(35, "Electro Wizard", 4.0f, Archetype::MeleeSquad, 714, SPEED_FAST, 5.0f, 116, 18, 'z').withFirstHit(6)
            .withTargetsAir()
            .withSplitTargets(2)
            .withOnHit(std::make_shared<FreezeOnHit>(5, 0.0f))
            .withSpawnEffect(3.0f, 118, std::make_shared<FreezeOnHit>(5, 0.0f)));

        // Flying troops. Minions and Minion Horde are one unit at two squad
        // sizes; Mega Minion is a single heavier flier. All hit air and ground.
        add(troop(41, "Minions", 3.0f, Archetype::MeleeSquad, 230, SPEED_FAST, 2.5f, 107, 12, 'm')
            .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f} })
            .withFlying().withTargetsAir());

        add(troop(42, "Minion Horde", 5.0f, Archetype::MeleeSquad, 230, SPEED_FAST, 2.5f, 107, 12, 'h')
            .withOffsets({ {-0.6f, -0.3f}, {0.0f, -0.3f}, {0.6f, -0.3f},
                           {-0.6f, 0.3f}, {0.0f, 0.3f}, {0.6f, 0.3f} })
            .withFlying().withTargetsAir());

        // First hit 0.4 s: RoyaleAPI cr-api-data, hit_speed 1500 - load_time 1100 ms (docs/LIVE_SYNC.md).
        add(troop(43, "Mega Minion", 3.0f, Archetype::MeleeSquad, 837, SPEED_MEDIUM, 1.6f, 312, 15, 'F').withFirstHit(4)
            .withFlying().withTargetsAir());

        // === Ranged troops ===
        // Each needs its own confirmed Target field before .withTargetsAir();
        // "ranged" does not imply it.
        // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 900 - load_time 400 ms (docs/LIVE_SYNC.md).
        add(troop(1, "Archers", 3.0f, Archetype::RangedSquad, 304, SPEED_MEDIUM, 5.0f, 112, 9, 'A').withFirstHit(5)
            .withOffsets({ {0.0f, 0.0f}, {1.0f, 0.0f} })
            .withTargetsAir());

        // First hit 0.7 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/february-balance-changes/).
        add(troop(6, "Musketeer", 4.0f, Archetype::RangedSquad, 721, SPEED_MEDIUM, 6.0f, 217, 10, 'U').withFirstHit(7)
            .withTargetsAir().withSightRange(6.0f));
        // Bomber is ground-only; so are Bowler, Sparky and Cannon Cart.
        // 2026-09-08 balance update (supercell.com September balance changes, docs/LIVE_SYNC.md): damage 225 -> 212.
        // First hit 0.2 s: RoyaleAPI cr-api-data, hit_speed 1800 - load_time 1600 ms (docs/LIVE_SYNC.md).
        add(troop(9, "Bomber", 2.0f, Archetype::RangedSquad, 304, SPEED_MEDIUM, 4.5f, 212, 18, 'b').withFirstHit(2));
        // 2024-08-06 balance update: Wizard hp 720 -> 754 (Supercell august-balance-changes; docs/LIVE_SYNC.md).
        // 2026-09-08 balance update: damage 281 -> 304 (base, Evolution and Hero; docs/LIVE_SYNC.md).
        // First hit 0.5 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/september-balance-changes-2027/).
        add(troop(11, "Wizard", 5.0f, Archetype::RangedSquad, 754, SPEED_MEDIUM, 5.5f, 304, 14, 'W').withFirstHit(5)
            .withTargetsAir().withSplash(1.5f));
        // 2026-10-19 balance update: damage 151 -> 143 (docs/LIVE_SYNC.md).
        add(troop(20, "Dart Goblin", 3.0f, Archetype::RangedSquad, 261, SPEED_VERY_FAST, 6.5f, 143, 8, 'd')
            .withTargetsAir().withSightRange(7.0f)); // 2026-03-02 balance update: 7.5 -> 7.0 tiles (docs/LIVE_SYNC.md).
        // A piercing line: 11.0 total (4.0 range + 7.0), half-width 1.8. The
        // 7.5 -> 7 tile projectile range is from the 2026-08-04 balance update
        // (docs/LIVE_SYNC.md).
        // Knockback is not modelled on the troop attack path.
        // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 2500 - load_time 2000 ms (docs/LIVE_SYNC.md).
        add(troop(22, "Bowler", 5.0f, Archetype::RangedSquad, 2081, SPEED_SLOW, 4.0f, 289, 25, 'w').withFirstHit(5)
            .withSplash(1.8f).withLineSplash(11.0f).withSightRange(4.0f));

        // First hit 0.5 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/august-balance-changes-2/).
        add(troop(23, "Spear Goblins", 2.0f, Archetype::RangedSquad, 133, SPEED_VERY_FAST, 5.0f, 81, 16, 'S').withFirstHit(5)
            .withTargetsAir()
            .withOffsets({ {0.0f, 0.0f}, {0.7f, 0.0f}, {-0.7f, 0.0f} }));

        // Ice Wizard fires a projectile; the slow lands on arrival.
        // 2024-08-06 balance update: attack splash radius 1 -> 1.5 tiles (Supercell august-balance-changes; docs/LIVE_SYNC.md).
        // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1700 - load_time 1200 ms (docs/LIVE_SYNC.md).
        add(troop(34, "Ice Wizard", 3.0f, Archetype::RangedSquad, 689, SPEED_MEDIUM, 5.5f, 90, 17, 'i').withFirstHit(5)
            .withTargetsAir().withSplash(1.5f)
            // 2025-10-06 balance update: slowdown 35% -> 30%, so speed factor 0.70
            // (docs/LIVE_SYNC.md).
            .withOnHit(std::make_shared<FreezeOnHit>(30, 0.70f)));

        // Executioner's axe hits on arrival and again 1.5 s later on the
        // return. The real axe pierces everything along its path both ways;
        // here it hits its one target twice (Projectile's returnsToSender).
        add(troop(36, "Executioner", 5.0f, Archetype::RangedSquad, 1280, SPEED_MEDIUM, 4.5f, 179, 24, 'x')
            .withTargetsAir()
            .withBoomerang(15));

        // First hit 0.3 s: RoyaleAPI cr-api-data, hit_speed 1500 - load_time 1200 ms (docs/LIVE_SYNC.md).
        add(troop(44, "Baby Dragon", 4.0f, Archetype::RangedSquad, 1152, SPEED_FAST, 3.5f, 168, 15, 'y').withFirstHit(3)
            .withFlying().withTargetsAir()
            .withSplash(1.5f));

        // === Building targeters ===
        // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1500 - load_time 1000 ms (docs/LIVE_SYNC.md).
        add(troop(2, "Giant", 5.0f, Archetype::MeleeBuildingTargeter, 3968, SPEED_SLOW, 1.2f, 253, 15, 'G').withFirstHit(5).withSightRange(7.5f));

        // First hit 0.6 s: RoyaleAPI cr-api-data, hit_speed 1600 - load_time 1000 ms (docs/LIVE_SYNC.md).
        add(troop(15, "Hog Rider", 4.0f, Archetype::MeleeBuildingTargeter, 1697, SPEED_VERY_FAST, 0.8f, 317, 16, 'H').withFirstHit(6)
            .withIgnoresRiver().withSightRange(9.5f));

        // Golem splits into two Golemites and deals a death explosion: two
        // death effects composed, since deathEffect is a single slot.
        // First hit 1.0 s: RoyaleAPI cr-api-data, hit_speed 2500 - load_time 1500 ms (docs/LIVE_SYNC.md).
        add(troop(19, "Golem", 8.0f, Archetype::MeleeBuildingTargeter, 5120, SPEED_SLOW, 0.75f, 312, 25, 'L').withFirstHit(10)
            .withDeathEffect(std::make_shared<CompositeDeathEffect>(
                std::vector<std::shared_ptr<IDeathEffect>>{
                    std::make_shared<SpawnOnDeath>(golemiteStats()),
                    std::make_shared<AreaDamageOnDeath>(2.5f, 200) }))
            .withSightRange(7.5f)); // 2026-09-08 balance update (supercell.com September balance changes, docs/LIVE_SYNC.md): sight range 7 -> 7.5

        // Ice Golem: the slow is on the death explosion, as in the real card;
        // there is no on-attack slow. The slow is 30% (factor 0.70), 2025-10-06
        // balance update; 2026-09-08 balance update: hitpoints 1315 -> 1228 and
        // slow duration 2.5 s (25 ticks) (docs/LIVE_SYNC.md).
        // First hit 1.0 s: RoyaleAPI cr-api-data, hit_speed 2500 - load_time 1500 ms (docs/LIVE_SYNC.md).
        add(troop(40, "Ice Golem", 2.0f, Archetype::MeleeBuildingTargeter, 1228, SPEED_SLOW, 0.75f, 84, 25, 'c').withFirstHit(10)
            .withDeathEffect(std::make_shared<AreaDamageOnDeath>(
                2.0f, 84, std::make_shared<FreezeOnHit>(25, 0.70f))).withSightRange(7.0f));

        // Balloon: flying and building-only, point-blank range; building
        // targeters never consider air, so no withTargetsAir. The real death
        // bomb drops only if shot down before bombing; here it always does.
        // First hit 0.2 s: RoyaleAPI cr-api-data, hit_speed 2000 - load_time 1800 ms (docs/LIVE_SYNC.md).
        add(troop(45, "Balloon", 5.0f, Archetype::MeleeBuildingTargeter, 1676, SPEED_MEDIUM, 0.1f, 640, 20, 'a').withFirstHit(2)
            .withFlying()
            .withDeathEffect(std::make_shared<AreaDamageOnDeath>(1.5f, 240)).withSightRange(7.7f));

        // === Ranged building targeter ===
        add(troop(18, "Royal Giant", 6.0f, Archetype::RangedBuildingTargeter, 3164, SPEED_SLOW, 5.0f, 307, 18, 'Y').withSightRange(7.5f));

        // === Defensive structures ===
        // First hit 1.0 s (10 ticks), the same as its hit speed: https://liquipedia.net/clashroyale/Cannon ; https://clashroyale.fandom.com/wiki/Cannon
        add(building(25, "Cannon", 3.0f, 824, 'C', 5.5f, 202, 10).withFirstHit(10));
        // First hit 0.4 s: RoyaleAPI cr-api-data, hit_speed 1100 - load_time 700 ms (docs/LIVE_SYNC.md).
        add(building(26, "Tesla", 4.0f, 1182, 'T', 5.5f, 220, 11).withFirstHit(4).withTargetsAir());
        // withSightRange(6.0f) is not redundant with the 5.5 default: with
        // attackRange 6.0, the default would let it attack what it cannot see.
        // Pinned by test_sight_range.cpp.
        // Its bombs splash (1.5 tiles: the game files' BombTowerProjectile
        // radius 1500), and when it is destroyed it drops a bomb that explodes
        // 3.0 s later (fuse 29: see DelayedAreaDamageOnDeath) in 3 tiles for
        // its own damage (BombTowerBomb: death_damage 105 at level 1, radius
        // 3000, deploy 3000 ms). It had neither: a single-target tower. Both
        // stay on the ground, as the tower does: the splash follows its
        // targetsAir, the bomb is groundOnly. UPSTREAM_REQUESTS.md item 35.
        add(building(27, "Bomb Tower", 4.0f, 1356, 'D', 6.0f, 222, 18).withSightRange(6.0f)
            .withSplash(1.5f)
            .withDeathEffect(std::make_shared<DelayedAreaDamageOnDeath>(3.0f, 222, 29, 0.0f, 27, "Bomb Tower Bomb",
                                                                        /*groundOnly=*/true)));
        // Inferno Tower: 5% of max damage for the first 2 s, 18.75% for the
        // next 2 s, then full, reset by a target switch or a stun. `damage` is
        // the fully ramped max; getCurrentDamage() scales it.
        add(building(28, "Inferno Tower", 5.0f, 1748, 'I', 6.0f, 847, 4)
            .withTargetsAir()
            .withDamageRamp(20, 40, 0.05f, 0.1875f).withSightRange(6.0f));

        // === Spells ===
        // Arrows: 3 volleys of 123, not one 369 hit. A Crown Tower takes 25 per
        // volley (75 in all): June 2026 balance update, 93 -> 75 total, per
        // volley being 75 / 3 (docs/LIVE_SYNC.md; UPSTREAM_REQUESTS.md item 29).
        add(spell(3, "Arrows", 3.0f, 3.5f, 123, 10, '*').withRepeats(3, 2)
            .withCrownTowerDamage(25));
        // Fireball: a Crown Tower takes 159 (2026-09-08 balance update: Crown
        // Tower Damage 172 -> 159; docs/LIVE_SYNC.md; UPSTREAM item 29).
        add(spell(7, "Fireball", 4.0f, 2.5f, 689, 10, 'O')
            .withCrownTowerDamage(159)
            .withKnockback(1.0f));
        // Crown Tower damage per application: 48 (official June 2026 notes, Zap 58 -> 48;
        // UPSTREAM_REQUESTS.md item 29).
        add(spell(29, "Zap", 2.0f, 2.5f, 192, 3, 'Z').withCrownTowerDamage(48));
        // Crown Tower damage 342 (official June 2026 notes, Rocket 371 -> 342; item 29).
        add(spell(30, "Rocket", 6.0f, 2.0f, 1485, 15, 'r')
            .withKnockback(1.0f).withCrownTowerDamage(342));
        // Lightning: strikes only the 3 enemies with the most hitpoints in the
        // radius (cr-api-data card description: "up to three enemy troops or
        // buildings with the most hitpoints in the target area"; the official
        // Sep 2025 / Sep 2026 notes presuppose it). Same mechanism as Vines.
        // Crown Tower damage 265 per strike (official June 2026 notes, Lightning 286 -> 265; item 29).
        add(spell(31, "Lightning", 6.0f, 3.5f, 1057, 5, 'j')
            .withSpellTopHpTargets(3).withCrownTowerDamage(265));
        // Poison: 92 damage every second for 8 s; whoever stands in it takes
        // each pulse. A Crown Tower takes 21 per pulse (168 in all: official
        // June 2026 notes, fixed in game on 26 Aug), and enemy troops inside
        // move 15% slower -- movement only, refreshed by each pulse.
        // UPSTREAM_REQUESTS.md items 29 and 32k.
        add(spell(32, "Poison", 4.0f, 3.5f, 92, 0, 'n').withRepeats(8, 10)
            .withCrownTowerDamage(21)
            .withSpellOnHit(std::make_shared<MoveSlowOnHit>(10, 0.85f)));
        // The Log: ground-only, a rolling sweep of published width 3.9 and
        // range 10.1. The radius argument is unused for rollers and kept at 3.9
        // to match. From the bridge the 10.1 reaches an enemy Princess Tower's
        // near edge (9.00 away) but not its centre (10.50); test_area_spell.cpp
        // pins this against ArenaLayout. Speed (1.0 tiles/tick, ~1 s for the
        // roll) and knockback (0.8) are not sourced.
        // 2025-09-01 "New Season: Back to Drool": damage 266 -> 268 (docs/LIVE_SYNC.md).
        add(spell(33, "The Log", 2.0f, 3.9f, 268, 8, 'o')
            .withGroundOnly()
            .withRollingSweep(10.1f, 3.9f, 1.0f, 0.8f)
            // Crown Tower damage 35 (official June 2026 notes, The Log 41 -> 35; item 29).
            .withCrownTowerDamage(35));

        // --- the 2026 roster expansion (ids 46+) ---
        // Every remaining non-Champion, non-Evolution, non-Tower-Troop card,
        // from the same data. Mechanics without a hook here are noted per card;
        // cards built entirely around one are listed as excluded at the end of
        // this file.
        //
        // Symbols are cosmetic (combat keys everything off cardId) and the
        // printable ASCII range is exhausted, so later cards reuse symbols.

        // === Melee troops (expansion) ===
        // Dark Prince jumps the river.
        add(troop(46, "Dark Prince", 4.0f, Archetype::MeleeSquad, 1200, SPEED_MEDIUM, 1.2f, 266, 14, 'N')
            .withShield(240) // sourced
            .withCharge(3.0f, 2.0f) // +100% on a charging hit
            .withChargeSpeed(2.0f)  // and runs at 2x (item 35)
            .withIgnoresRiver());
        // Royal Ghost jumps the river.
        // 2026-10-06 balance update: 1152 hp, 263 damage (docs/LIVE_SYNC.md).
        // First hit 0.6 s: RoyaleAPI cr-api-data, hit_speed 1800 - load_time 1200 ms (docs/LIVE_SYNC.md).
        add(troop(47, "Royal Ghost", 3.0f, Archetype::MeleeSquad, 1152, SPEED_FAST, 1.2f, 263, 18, 'Q').withFirstHit(6)
            .withInvisibility(20) // 2026-03-02 balance update: invisibility delay 1.8 s -> 2 s (docs/LIVE_SYNC.md); was 5 here, which contradicted the post's old 1.8 s
            .withIgnoresRiver());
        // Mega Knight: the deploy slam is the spawn-effect burst (radius 1.3,
        // 430). The periodic jump closes 3.5-5 tiles instantly with a 2.0x,
        // 2.2-radius splash hit (sourced 537 vs 268). The jump can carry him
        // over the river; with no "mid-jump" state, he ignores the river
        // always, as Bandit does. Stats are an exact level-11 row (hp 3993,
        // area 268, spawn 429).
        // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1700 - load_time 1200 ms (docs/LIVE_SYNC.md).
        add(troop(48, "Mega Knight", 7.0f, Archetype::MeleeSquad, 3993, SPEED_MEDIUM, 1.2f, 268, 17, 'X').withFirstHit(5)
            .withSplash(1.5f)
            .withSpawnEffect(1.3f, 430)
            .withJump(3.5f, 5.0f, 2.0f, 2.2f)
            .withIgnoresRiver());
        // The real heal is 4 pulses of 25.5 (102) per attack cycle, applied
        // here as one 102 heal per landed hit. The larger deploy heal (202) is
        // not modelled (spawn effects only damage). Battle Healer jumps the
        // river.
        // 2026-08-04 balance update: 1920 hp, 268 damage, 2 s hit speed
        // (docs/LIVE_SYNC.md); the post does not restate the heal.
        add(troop(49, "Battle Healer", 4.0f, Archetype::MeleeSquad, 1920, SPEED_MEDIUM, 1.2f, 268, 20, 'f')
            .withHealAura(3.0f, 102)
            .withIgnoresRiver());
        // Bandit crosses the river only by dashing; with no separate dash
        // state, she always ignores it.
        // First hit 0.4 s: RoyaleAPI cr-api-data, hit_speed 1000 - load_time 600 ms (docs/LIVE_SYNC.md).
        add(troop(50, "Bandit", 3.0f, Archetype::MeleeSquad, 906, SPEED_FAST, 1.0f, 194, 10, 'u').withFirstHit(4)
            .withCharge(3.0f, 2.0f)
            .withChargeInvulnerability() // invulnerable while charging
            .withSightRange(6.0f)
            .withIgnoresRiver());
        // Berserker: a flat 0.6 s hit speed, Fast, Melee: Short (0.8). There is
        // no rage: the self-heal and hit-speed bonuses the wiki lists are event
        // Modifiers, not the card (UPSTREAM_REQUESTS.md item 32d).
        add(troop(51, "Berserker", 2.0f, Archetype::MeleeSquad, 896, SPEED_FAST, 0.8f, 102, 6, 'v'));
        add(troop(52, "Miner", 3.0f, Archetype::MeleeSquad, 1210, SPEED_FAST, 1.0f, 194, 13, '0')
            .withDeployAnywhere());
        // First hit 0.1 s: RoyaleAPI cr-api-data, hit_speed 1300 - load_time 1200 ms (docs/LIVE_SYNC.md).
        add(troop(53, "Fisherman", 3.0f, Archetype::MeleeSquad, 870, SPEED_MEDIUM, 1.0f, 194, 13, '1').withFirstHit(1)
            .withHook(6.5f).withSightRange(7.5f));
        // Parry every 3.5 s. The real parry reflects 200% damage and only
        // against ground melee; neither is modelled (takeDamage carries no
        // attacker, and there is no melee/ranged distinction).
        // 2026-09-08 balance update: hit speed 1.4 s -> 1.3 s (13 ticks), docs/LIVE_SYNC.md.
        // First hit 0.4 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/news/final-august-balance-changes-826/).
        add(troop(54, "Ronin", 5.0f, Archetype::MeleeSquad, 1779, 0.7f, 1.2f, 371, 13, '2').withFirstHit(4)
            .withParry(35));
        // 2026-08-04 balance update: hp 2150 -> 2265, melee damage 212 -> 232
        // (docs/LIVE_SYNC.md); the rocket turret is goblinMachineTurretStats.
        add(troop(55, "Goblin Machine", 5.0f, Archetype::MeleeSquad, 2265, 0.5f, 1.2f, 232, 12, '3')
            .withSecondaryUnit(goblinMachineTurretStats())); // independently targeting rocket turret

        // Inferno Dragon and Electro Dragon: instant zaps, so MeleeSquad as
        // with Electro Wizard. Inferno Dragon ramps 35 -> 120 -> 422 every 1.5
        // s on one target; `damage` is the ramped max.
        add(troop(56, "Inferno Dragon", 4.0f, Archetype::MeleeSquad, 1295, SPEED_MEDIUM, 5.0f, 422, 4, '4')
            .withFlying().withTargetsAir()
            .withDamageRamp(15, 30, 0.083f, 0.284f));
        // A true chain, not splash: up to 3 distinct targets (hops up to 4
        // tiles), each taking the full 192.
        // First hit 0.5 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/october-balance-changes-2026/).
        add(troop(57, "Electro Dragon", 5.0f, Archetype::MeleeSquad, 1049, SPEED_MEDIUM, 3.5f, 192, 21, '5').withFirstHit(5)
            .withFlying().withTargetsAir()
            .withSplitTargets(3)
            .withSplitTargetsFullDamage()
            .withOnHit(std::make_shared<FreezeOnHit>(5, 0.0f)));

        // Night Witch: releases 3 Bats on death and spawns Bats periodically.
        // First hit 0.8 s: RoyaleAPI cr-api-data, hit_speed 1300 - load_time 550 ms (docs/LIVE_SYNC.md).
        add(troop(58, "Night Witch", 4.0f, Archetype::MeleeSquad, 906, SPEED_MEDIUM, 1.0f, 314, 13, '6').withFirstHit(8)
            .withDeathEffect(std::make_shared<SpawnOnDeath>(nightWitchBatStats()))
            .withPeriodicEffect(50, std::make_shared<PeriodicSpawnEffect>(nightWitchPeriodicBatStats())));
        // Phoenix: one-time revive (phoenixReviveStats).
        // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1000 - load_time 500 ms (docs/LIVE_SYNC.md).
        add(troop(59, "Phoenix", 4.0f, Archetype::MeleeSquad, 1052, SPEED_MEDIUM, 1.0f, 217, 10, '7').withFirstHit(5)
            .withFlying().withTargetsAir()
            .withDeathEffect(std::make_shared<SpawnOnDeath>(phoenixReviveStats())));

        // === Ranged troops (expansion) ===
        add(troop(60, "Sparky", 6.0f, Archetype::RangedSquad, 1451, SPEED_SLOW, 5.0f, 1331, 40, '8')
            .withSplash(1.5f)
            .withStunResetsCooldown() // a stun restarts her charge
            .withSightRange(5.0f));
        // First hit 0.3 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/march-balance-changes/).
        add(troop(61, "Princess", 3.0f, Archetype::RangedSquad, 261, SPEED_MEDIUM, 9.0f, 168, 30, '9').withFirstHit(3)
            .withTargetsAir()
            .withSplash(1.5f).withSightRange(9.5f)
            // 2025-03-04 balance update: Deploy Time 1.0 s -> 1.2 s
            // (supercell.com march-balance-changes).
            .withDeployTime(12));
        // Hunter: the real card fires 10 randomly spread pellets and no falloff
        // curve is published. withRangeFalloff is an invented deterministic
        // approximation: linear from full damage point-blank to half at max
        // range.
        // 2024-08-06 balance update: Hunter hp 838 -> 884 (Supercell august-balance-changes; docs/LIVE_SYNC.md).
        // First hit 0.7 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/june-balance-changes-2/).
        add(troop(62, "Hunter", 4.0f, Archetype::RangedSquad, 884, SPEED_MEDIUM, 4.0f, 84, 22, '!').withFirstHit(7)
            .withTargetsAir()
            .withSplash(1.5f)
            .withRangeFalloff(0.5f));
        // A piercing line: 11.0 long, half-width 0.25.
        // 2024-10-08 balance update: hp 532 -> 531 (docs/LIVE_SYNC.md). Mortar always takes two hits.
        // 2026-08-04 balance update (final August changes, supercell.com): damage 143 -> 135.
        add(troop(63, "Magic Archer", 4.0f, Archetype::RangedSquad, 531, SPEED_MEDIUM, 7.0f, 135, 11, '#')
            .withTargetsAir()
            .withSplash(0.25f).withLineSplash(11.0f).withSightRange(7.5f));
        // First hit 0.7 s: RoyaleAPI cr-api-data, hit_speed 3000 - load_time 2350 ms (docs/LIVE_SYNC.md).
        add(troop(64, "Firecracker", 3.0f, Archetype::RangedSquad, 304, SPEED_FAST, 6.0f, 64, 30, '$').withFirstHit(7)
            .withTargetsAir()
            .withSplash(1.5f)
            // 2025-08-06 balance update: sight range 8.5 -> 8 tiles (official post, docs/LIVE_SYNC.md).
            .withRecoil(1.0f).withSightRange(8.0f)); // kicks back 1 tile after every attack
        add(troop(65, "Skeleton Dragons", 4.0f, Archetype::RangedSquad, 560, SPEED_FAST, 3.5f, 151, 20, '%')
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} }).withFlying().withTargetsAir()
            .withSplash(1.5f));
        // At <= 50% hp it kills itself and the kamikaze form spawns in its
        // place (goblinDemolisherKamikazeStats).
        add(troop(66, "Goblin Demolisher", 4.0f, Archetype::RangedSquad, 1300, 0.5f, 5.0f, 186, 11, '&')
            .withSplash(1.5f)
            .withHpTransformIntoDeath(0.5f, std::make_shared<SpawnOnDeath>(goblinDemolisherKamikazeStats())));
        add(troop(67, "Flying Machine", 4.0f, Archetype::RangedSquad, 614, SPEED_FAST, 6.0f, 179, 11, '+')
            // 2026-10-19 balance update: damage 171 -> 179 (docs/LIVE_SYNC.md).
            .withFlying().withTargetsAir().withSightRange(6.0f));
        // The curse spawns a Cursed Hog for Mother Witch (CursedHogOnHit).
        // First hit 0.3 s: RoyaleAPI cr-api-data, hit_speed 1000 - load_time 700 ms (docs/LIVE_SYNC.md).
        add(troop(68, "Mother Witch", 4.0f, Archetype::RangedSquad, 529, SPEED_MEDIUM, 5.5f, 133, 10, ',').withFirstHit(3)
            .withTargetsAir()
            .withOnHit(std::make_shared<CursedHogOnHit>(1.3f, 60, cursedHogStats())));
        // Cannon Cart: a single 1809 hp pool; at 50% it grounds itself for 15
        // s, then self-destructs, keeping its damage, range and hit speed
        // (CombatEntity::transformBecomesStationary).
        add(troop(69, "Cannon Cart", 5.0f, Archetype::RangedSquad, 1809, SPEED_MEDIUM, 5.5f, 202, 9, '?')
            // 2026-10-19 balance update: damage 212 -> 202 (docs/LIVE_SYNC.md).
            .withHpTransform(0.5f, 150, true)
            // The sourced sight is 6.0 mobile and 5.5 once grounded; one entity
            // models both, and the mobile value is what matters for chasing.
            .withSightRange(6.0f));
        // Furnace: a walking troop since the August 2025 rework, Medium since
        // 2025-10-06, one Fire Spirit every 7 s.
        add(troop(70, "Furnace", 4.0f, Archetype::RangedSquad, 727, SPEED_MEDIUM, 5.5f, 179, 17, '/')
            .withTargetsAir()
            .withPeriodicEffect(70, std::make_shared<PeriodicSpawnEffect>(furnaceFireSpiritStats())));
        // Witch splash 1 -> 1.5 tiles: 2025-01-07 balance update (docs/LIVE_SYNC.md).
        // First hit 0.7 s: RoyaleAPI cr-api-data, hit_speed 1100 - load_time 400 ms (docs/LIVE_SYNC.md).
        add(troop(71, "Witch", 5.0f, Archetype::RangedSquad, 839, SPEED_MEDIUM, 5.5f, 135, 11, ':').withFirstHit(7)
            .withTargetsAir().withSplash(1.5f)
            .withPeriodicEffect(70, std::make_shared<PeriodicSpawnEffect>(witchSkeletonStats())));

        // "Spirit" troops detonate on their first hit and vanish
        // (CombatEntity::dieAfterFirstHit).
        //
        // Ice Spirit stuns (0.0f), not slows: a stun is FreezeOnHit(ticks,
        // 0.0f) (Electro Spirit, Zappies, Freeze), and 0.5-0.7 is the slow band
        // (Ice Wizard, Ice Golem's explosion). Pinned in
        // tests/core/test_default_deck_qa.cpp.
        //
        // Every Spirit is 215 hp since the official 4 Aug 2026 notes (230 ->
        // 215); Fire Spirit hits for 215 since 16 Sep 2026 (207 -> 215), in a
        // 2.3-tile area (UPSTREAM_REQUESTS.md item 32b).
        // Ice Spirit's freeze is 11 ticks (1.1 s): official 7 Jul 2025 notes,
        // "Freeze Duration: 1.2sec -> 1.1sec" (docs/LIVE_SYNC.md).
        add(troop(72, "Ice Spirit", 1.0f, Archetype::RangedSquad, 215, SPEED_VERY_FAST, 2.5f, 110, 10, ';')
            .withTargetsAir()
            .withOnHit(std::make_shared<FreezeOnHit>(11, 0.0f))
            .withDieAfterFirstHit());
        add(troop(73, "Fire Spirit", 1.0f, Archetype::RangedSquad, 215, SPEED_VERY_FAST, 2.5f, 215, 10, '<')
            .withTargetsAir()
            .withSplash(2.3f)
            .withDieAfterFirstHit());
        // The real heal is 4 pulses of 100.25 (401), applied here as one heal
        // on its single hit.
        add(troop(74, "Heal Spirit", 1.0f, Archetype::RangedSquad, 215, 0.85f, 2.5f, 110, 10, '=')
            .withTargetsAir()
            .withHealAura(2.5f, 401)
            .withDieAfterFirstHit());
        add(troop(75, "Electro Spirit", 1.0f, Archetype::RangedSquad, 215, SPEED_VERY_FAST, 2.5f, 99, 10, '>')
            .withTargetsAir()
            .withSplitTargets(2)
            .withOnHit(std::make_shared<FreezeOnHit>(8, 0.0f))
            .withDieAfterFirstHit());

        // === Swarms (expansion) ===
        // Guards and Royal Recruits shields: 256 and 240.
        // First hit 0.5 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/october-balance-changes-2/).
        add(troop(76, "Guards", 3.0f, Archetype::MeleeSquad, 81, SPEED_FAST, 1.0f, 117, 10, '?').withFirstHit(5)
            .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f} })
            .withShield(256));
        // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1300 - load_time 800 ms (docs/LIVE_SYNC.md).
        add(troop(77, "Royal Recruits", 7.0f, Archetype::MeleeSquad, 547, SPEED_MEDIUM, 1.0f, 133, 13, '@').withFirstHit(5)
            .withOffsets({ {-2.5f, 0.0f}, {-1.5f, 0.0f}, {-0.5f, 0.0f}, {0.5f, 0.0f}, {1.5f, 0.0f}, {2.5f, 0.0f} })
            .withShield(240));
        add(troop(78, "Bats", 2.0f, Archetype::MeleeSquad, 81, SPEED_VERY_FAST, 1.0f, 81, 12, 't')
            .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f}, {0.3f, 0.5f}, {-0.3f, 0.5f} })
            .withFlying().withTargetsAir());

        // 2026-09-08 balance update: hit speed 2.2 s -> 2.3 s (23 ticks), docs/LIVE_SYNC.md.
        // First hit 0.8 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/august-balance-changes-2/).
        add(troop(79, "Zappies", 4.0f, Archetype::RangedSquad, 529, SPEED_MEDIUM, 4.5f, 117, 23, '[').withFirstHit(8)
            .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f} })
            .withTargetsAir()
            .withOnHit(std::make_shared<FreezeOnHit>(3, 0.0f)).withSightRange(5.0f));
        // withSightRange(6.0f), like the single Musketeer (id 6): three of her,
        // with attackRange 6.0.
        // 2026-02-02 balance update: hit speed 1.2 s -> 1.3 s, so 13 ticks (docs/LIVE_SYNC.md;
        // supercell.com february-balance-changes-26). It carried 10 ticks.
        // 2026-10-06 balance update: 906 hp per Musketeer (docs/LIVE_SYNC.md).
        // First hit 0.7 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/february-balance-changes/).
        add(troop(80, "Three Musketeers", 9.0f, Archetype::RangedSquad, 906, SPEED_MEDIUM, 6.0f, 218, 13, ']').withFirstHit(7)
            .withOffsets({ {-2.0f, 0.0f}, {0.0f, 0.0f}, {2.0f, 0.0f} })
            .withTargetsAir().withSightRange(6.0f));

        // === Building targeters (expansion) ===
        // Battle Ram: breaks on its first hit on a building and releases its
        // 2 Barbarians, as the real card does (withDieAfterFirstHit; the hit
        // is doubled if it charged). It used to keep swinging until killed.
        // UPSTREAM_REQUESTS.md item 34. Level 11: 967 hp, 286 damage (572 on
        // a charge; the game's is 573 since a March 2025 rounding change),
        // 0.5-tile reach, a 2x-speed charge, 6.5 sight (official September
        // 2026 notes: 5.5 -> 6.5). It had 691 / 192 / 1.0 reach / no charge
        // speed / 5.5 sight. Item 35. The Barbarians keep their 1 s deploy:
        // the game files' death_spawn_deploy_time is 1000 ms.
        add(troop(81, "Battle Ram", 4.0f, Archetype::MeleeBuildingTargeter, 967, SPEED_MEDIUM, 0.5f, 286, 14, '^')
            .withDeathEffect(std::make_shared<SpawnOnDeath>(battleRamBarbarianStats()))
            .withCharge(3.0f, 2.0f).withChargeSpeed(2.0f).withDieAfterFirstHit().withSightRange(6.5f));
        // Royal Hogs jump the river.
        // First hit 0.4 s: RoyaleAPI cr-api-data, hit_speed 1200 - load_time 800 ms (docs/LIVE_SYNC.md).
        add(troop(82, "Royal Hogs", 5.0f, Archetype::MeleeBuildingTargeter, 837, SPEED_VERY_FAST, 1.0f, 74, 12, '_').withFirstHit(4)
            .withOffsets({ {-1.0f, -0.3f}, {-0.3f, 0.3f}, {0.3f, -0.3f}, {1.0f, 0.3f} })
            .withCharge(3.0f, 2.0f).withSightRange(9.5f)
            .withIgnoresRiver());
        // 2026-10-06 balance update: 302 damage per Wall Breaker (docs/LIVE_SYNC.md).
        // First hit 0.2 s: RoyaleAPI cr-api-data, hit_speed 1200 - load_time 1000 ms (docs/LIVE_SYNC.md).
        add(troop(83, "Wall Breakers", 2.0f, Archetype::MeleeBuildingTargeter, 330, SPEED_VERY_FAST, 1.0f, 302, 12, '{').withFirstHit(2)
            .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} })
            .withSplash(1.5f)
            .withDieAfterFirstHit().withSightRange(7.0f));
        // First hit 1.0 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/may-balance-changes-2026/).
        add(troop(84, "Electro Giant", 7.0f, Archetype::MeleeBuildingTargeter, 3952, SPEED_SLOW, 1.0f, 184, 18, '|').withFirstHit(10) // 2026-08-26 final August balance update: damage 163 -> 184 (docs/LIVE_SYNC.md)
            .withSplash(1.5f)
            .withPeriodicEffect(50, std::make_shared<AreaStunEffect>(2.5f, 5)));
        // 2024-10-08 balance update: Bush speed 45 -> 60 = Medium (docs/LIVE_SYNC.md).
        add(troop(85, "Suspicious Bush", 2.0f, Archetype::MeleeBuildingTargeter, 81, SPEED_MEDIUM, 0.25f, 256, 14, '}')
            .withInvisibility(5)
            .withDeathEffect(std::make_shared<SpawnOnDeath>(suspiciousBushGoblinStats())));
        // 2026-08-04 balance update: 2816 hp (was 2662; docs/LIVE_SYNC.md).
        add(troop(86, "Rune Giant", 4.0f, Archetype::MeleeBuildingTargeter, 2816, 0.5f, 1.2f, 153, 15, '~')
            .withAllyBuffAura(3.0f, 3, 1.5f, 50, 2));
        // Ram Rider jumps the river.
        // 2026-10-06 balance update: ram damage 271; the charge hit is 2x (docs/LIVE_SYNC.md).
        add(troop(87, "Ram Rider", 5.0f, Archetype::MeleeBuildingTargeter, 1766, SPEED_MEDIUM, 1.0f, 271, 17, '"')
            .withCharge(2.5f, 2.0f).withChargeSpeed(2.0f) // the ram charges at 2x (item 35); 2.5 tiles to charge (official 24 Oct 2025 post, docs/LIVE_SYNC.md)
            .withSecondaryUnit(ramRiderCrossbowStats()) // the rider's independently targeting crossbow (default 5.5 sight)
            .withSightRange(7.5f) // the ram itself
            .withIgnoresRiver());
        // First hit 0.8 s: RoyaleAPI cr-api-data, hit_speed 1500 - load_time 700 ms (docs/LIVE_SYNC.md).
        add(troop(88, "Goblin Giant", 6.0f, Archetype::MeleeBuildingTargeter, 3110, SPEED_MEDIUM, 1.2f, 176, 15, '`').withFirstHit(8)
            .withSecondaryUnit(goblinGiantSpearGoblinsStats()).withSightRange(7.5f)); // carried Spear Goblins, independently targeting
        // Skeleton Barrel releases 2 Skeletons on death.
        add(troop(89, "Skeleton Barrel", 3.0f, Archetype::MeleeBuildingTargeter, 532, SPEED_FAST, 1.0f, 81, 10, ',')
            .withFlying()
            .withDeathEffect(std::make_shared<SpawnOnDeath>(skeletonBarrelSkeletonStats())).withSightRange(7.7f));
        // The split chain: Golem -> 2 Golemites (each -> 2 Blobs), with elixir
        // to the opponent at each tier (see elixirGolemiteStats,
        // elixirBlobStats).
        // First hit 0.8 s: RoyaleAPI cr-api-data, hit_speed 1100 - load_time 300 ms (docs/LIVE_SYNC.md).
        add(troop(90, "Elixir Golem", 3.0f, Archetype::MeleeBuildingTargeter, 1569, SPEED_SLOW, 1.0f, 253, 11, '(').withFirstHit(8)
            .withDeathEffect(std::make_shared<CompositeDeathEffect>(
                std::vector<std::shared_ptr<IDeathEffect>>{
                    std::make_shared<SpawnOnDeath>(
                        elixirGolemiteStats().withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} })),
                    std::make_shared<EnemyElixirGrantOnDeath>(1.0f)
                }))
            .withSightRange(7.5f));

        // === Ranged building targeter (expansion) ===
        // Lava Hound releases 6 Lava Pups; their spread is an engine choice.
        // 2026-10-06 balance update: damage 53 -> 71, hit speed 1.3 s -> 1.5 s
        // (docs/LIVE_SYNC.md).
        add(troop(91, "Lava Hound", 7.0f, Archetype::RangedBuildingTargeter, 3581, SPEED_SLOW, 3.5f, 71, 15, ')')
            .withFlying()
            .withDeathEffect(std::make_shared<SpawnOnDeath>(lavaHoundPupStats())));

        // Minion Giant (new card, season 87, 2026-09-07). Flying ranged building
        // targeter, like Lava Hound; a building targeter never considers air, so no
        // withTargetsAir. Damage 153 (official: Sep 16 189->168, Oct 19 168->153) and
        // hit speed 1.7 s (official, Oct 6, from 1.5 s).
        // 1817 hp, 4.0 range, 6.5 sight, SPEED_MEDIUM: single source, LDPlayer guide
        // (https://www.ldplayer.net/blog/clash-royale-minion-giant-guide.html, HP also
        // relayed from RoyaleAPI by After Strategy); nothing contradicts them.
        // Deploy time is the engine default and no first-hit delay is set (neither is
        // published); projectile speed is the RangedBuildingTargeter constant.
        // Symbol 'Y' is Royal Giant's (symbols are not unique); 'j' is Lightning's and
        // src/main.cpp hides it from the spawn log.
        add(troop(183, "Minion Giant", 4.0f, Archetype::RangedBuildingTargeter, 1817, SPEED_MEDIUM, 4.0f, 153, 17, 'Y')
            .withFlying()
            .withSightRange(6.5f));

        // === Defensive structures (expansion) ===
        // X-Bow: the default 30 s building lifetime matches the real card;
        // withDeployDelay adds the ~3.5 s lock-on.
        // 2026-08-04 balance update: hit speed 0.3 s -> 0.4 s (4 ticks; docs/LIVE_SYNC.md).
        // 2026-10-06 balance update (chain 43 -> 58 -> 61): 61 damage (docs/LIVE_SYNC.md).
        add(building(92, "X-Bow", 6.0f, 1600, 'P', 11.5f, 61, 4)
            .withDeployDelay(35).withSightRange(11.5f));
        // 2026-08-04 balance update (final August changes, supercell.com): hit speed 5 s -> 4.7 s.
        // First hit 1.0 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/october-balance-changes/).
        add(building(93, "Mortar", 4.0f, 1369, 'R', 11.5f, 266, 47).withFirstHit(10)
            .withMinRange(3.5f).withSightRange(11.5f)); // blind spot

        // Spawner buildings do not attack in the real game: 0-damage structures
        // spawning via PeriodicSpawnEffect. Barbarian Hut: 3 every 15 s, plus
        // one Barbarian on its death.
        add(building(94, "Barbarian Hut", 6.0f, 1164, '2', 0.0f, 0, 100)
            .withPeriodicEffect(150, std::make_shared<PeriodicSpawnEffect>(barbarianHutBarbarianStats()))
            .withDeathEffect(std::make_shared<SpawnOnDeath>(barbarianHutDeathBarbarianStats())));
        // Goblin Hut spawns only while an enemy is within 6 tiles, every 2.2 s
        // (ProximityGatedPeriodicSpawnEffect), and releases one Spear Goblin on
        // death.
        // 1180 hp since the 2025-10-06 -4% (was 1228).
        add(building(95, "Goblin Hut", 4.0f, 1180, '3', 0.0f, 0, 100)
            .withPeriodicEffect(22, std::make_shared<ProximityGatedPeriodicSpawnEffect>(goblinHutSpearGoblinStats(), 6.0f))
            .withDeathEffect(std::make_shared<SpawnOnDeath>(goblinHutSpearGoblinStats())));
        // Tombstone: 2 Skeletons per pulse, 4 on its death.
        add(building(96, "Tombstone", 3.0f, 529, '5', 0.0f, 0, 100)
            .withPeriodicEffect(35, std::make_shared<PeriodicSpawnEffect>(skeletonBarrelSkeletonStats()))
            .withDeathEffect(std::make_shared<SpawnOnDeath>(witchSkeletonStats())));
        // Goblin Cage: its whole function is releasing a Goblin Brawler on
        // death.
        add(building(97, "Goblin Cage", 4.0f, 780, '6', 0.0f, 0, 100)
            .withDeathEffect(std::make_shared<SpawnOnDeath>(goblinCageBrawlerStats())));
        // Goblin Drill: a one-time 360-degree emergence burst (the spawn
        // effect; knockback not modelled), periodic Goblins, and 2 Goblins on
        // death. A Crown Tower takes 20 from the burst (2026-10-06 balance
        // update: "Crown Tower Spawn Damage: 0 -> 20"; docs/LIVE_SYNC.md).
        add(building(98, "Goblin Drill", 4.0f, 1313, '7', 0.0f, 0, 100)
            .withPeriodicEffect(30, std::make_shared<PeriodicSpawnEffect>(goblinDrillGoblinStats()))
            .withDeployAnywhere()
            .withSpawnEffect(2.0f, 84).withSpawnEffectCrownTowerDamage(20)
            .withDeathEffect(std::make_shared<SpawnOnDeath>(goblinDrillDeathGoblinStats())));
        // Elixir Collector: no attack; periodically credits elixir
        // (ElixirGrantEffect). 2026-10-06 balance update: production speed
        // 13 s -> 15 s, i.e. one elixir per 150 ticks (docs/LIVE_SYNC.md). The
        // 1.0 amount per pulse is an engine assumption, not sourced.
        add(building(99, "Elixir Collector", 6.0f, 1070, '9', 0.0f, 0, 100)
            .withPeriodicEffect(150, std::make_shared<ElixirGrantEffect>(1.0f)));

        // === Spells (expansion) ===
        // Slow 30% (factor 0.70): 2025-10-06 balance update lists Giant Snowball
        // among the slowing effects cut from 35% to 30% (docs/LIVE_SYNC.md).
        // Slow lasts 3 s (30 ticks): the 2026-05-04 balance update cut the
        // Evolution's 4 s "standardizing the slow duration with its unevolved
        // form" (supercell.com May balance changes 2026; Fandom wiki lists 3 s
        // for the base). Was 15 ticks, unsourced (docs/LIVE_SYNC.md).
        // A Crown Tower takes 45 (June 2026 balance update: 54 -> 45;
        // docs/LIVE_SYNC.md; UPSTREAM item 29).
        add(spell(100, "Giant Snowball", 2.0f, 2.5f, 179, 8, '!')
            .withCrownTowerDamage(45)
            .withSpellOnHit(std::make_shared<FreezeOnHit>(30, 0.70f))
            .withKnockback(1.0f));
        // Barbarian Barrel: rolls 4.5 tiles at width 2.6 (published), and its
        // Barbarian spawns where the barrel stops. No knockback: the real
        // barrel does not throw what it rolls over. Cast on its own side only,
        // unlike The Log's river allowance (UPSTREAM_REQUESTS.md item 32n).
        // 2026-10-06 balance update: damage 232 -> 215 (docs/LIVE_SYNC.md).
        add(spell(101, "Barbarian Barrel", 2.0f, 2.5f, 215, 8, '#')
            .withGroundOnly()
            .withOwnSideOnly()
            .withRollingSweep(4.5f, 2.6f, 0.5f, 0.0f)
            .withSpellSpawn(std::make_shared<PeriodicSpawnEffect>(barbarianBarrelBarbarianStats())));
        // Goblin Curse: only the damage over time is modelled, not the
        // damage-taken amplification or the Goblin on cursed death.
        // 2025-09-01 September Balance Changes: damage/sec 30 -> 35 and the 20%
        // damage amplification removed (docs/LIVE_SYNC.md). One pulse per second
        // (10 ticks), so 35 per application x 6 = 210. The Crown Tower value
        // (6 -> 10 per second) is the separate spell-vs-tower item.
        // 2026-08-04 balance update: enemies inside move 15% slower (movement
        // only, refreshed by each pulse, as Poison; docs/LIVE_SYNC.md).
        // Crown Tower damage 10 per application (official Sept 2025 notes, "Crown
        // Tower Damage/Sec: 6 -> 10"; the engine applies once a second; item 29).
        add(spell(102, "Goblin Curse", 2.0f, 3.0f, 35, 8, '$').withRepeats(6, 10)
            .withCrownTowerDamage(10)
            .withSpellOnHit(std::make_shared<MoveSlowOnHit>(10, 0.85f)));
        // Earthquake: ground-only damage over time with the 50% movement slow.
        // The real 3.5x damage against buildings is not modelled. A Crown Tower
        // takes 49 per wave (147 in all: June 2026 balance update, 159 -> 147,
        // per wave being 147 / 3; docs/LIVE_SYNC.md; UPSTREAM item 29).
        add(spell(103, "Earthquake", 3.0f, 3.5f, 82, 8, '%').withRepeats(3, 10).withGroundOnly()
            .withCrownTowerDamage(49)
            .withSpellOnHit(std::make_shared<FreezeOnHit>(30, 0.5f)));
        // Void: three damage tiers, once per wave. 2026-08-04 balance update
        // (docs/LIVE_SYNC.md): 5 elixir (was 3), 696 per hit on 1 target
        // (was 340), 294 on 2-4 (was 161), 153 on 5+ (was 76). 2026-10-19
        // balance update: 2-4 target damage 294 -> 307 (docs/LIVE_SYNC.md).
        add(spell(104, "Void", 5.0f, 2.5f, 0, 8, '&').withRepeats(3, 10)
            .withSpellTieredDamage(696, 307, 153));
        // Vines: only the 3 highest-HP enemies in radius. Pulling flyers to the
        // ground is not modelled.
        // 102 per pulse x 3 = 306 total: 2025-10-06 balance update, total damage
        // 335 -> 306 (docs/LIVE_SYNC.md). The 2.5 s -> 2 s duration cut is not
        // modelled (the pulse cadence is unsourced).
        add(spell(105, "Vines", 3.0f, 2.5f, 102, 8, '+').withRepeats(3, 7)
            .withSpellTopHpTargets(3));
        // 2025-04-09 balance update: Damage 84x2 -> 84 (-50%), so one 84 hit in
        // total (supercell.com April balance changes; docs/LIVE_SYNC.md). The old
        // 154 matched neither the pre-patch 168 nor this value.
        add(spell(106, "Tornado", 3.0f, 5.5f, 84, 8, '_')
            .withKnockback(-1.5f)); // negative: pulls toward the centre
        // Freeze: a full stun (3.5 s since the 2026-09-08 balance update, was 4 s)
        // plus one hit of 148 damage (official December
        // 2025 balance notes, 115 -> 148, level 11: supercell.com
        // /clashroyale/blog/release-notes/december-balance-changes/). Crown Tower
        // damage is the separate June 2026 reduction (see withCrownTowerDamage).
        // A Crown Tower takes 37 (June 2026 balance update: 45 -> 37;
        // docs/LIVE_SYNC.md; UPSTREAM item 29).
        add(spell(107, "Freeze", 4.0f, 3.0f, 148, 8, '~')
            .withCrownTowerDamage(37)
            .withSpellOnHit(std::make_shared<FreezeOnHit>(35, 0.0f)));
        // Rage: buffs allied damage in radius. Boost 30% (2025-10-06 balance
        // update, 35% -> 30%; docs/LIVE_SYNC.md); the duration is not sourced.
        add(spell(108, "Rage", 2.0f, 3.0f, 0, 8, '(')
            .withSpellBuff(1.30f, 45));
        // Goblin Barrel: drops 3 Goblins; the barrel itself deals no damage.
        add(spell(109, "Goblin Barrel", 3.0f, 0.5f, 0, 8, '[')
            .withSpellSpawn(std::make_shared<PeriodicSpawnEffect>(goblinBarrelGoblinStats())));
        // Graveyard: 12 Skeletons (official June 2026), the first 2.2 s after
        // the cast (delay 21: an AreaSpell applies on the update after its
        // delay reaches 0), then one every 0.5 s, cycling seven fixed points --
        // the cast point and six on a 3.3-tile ring (AreaSpell::
        // spawnRingRadius) -- rather than all rising on the cast point, where
        // one small splash caught them all. Arrivals outrun a Princess
        // Tower's 0.8 s fire rate 1.6:1. How many connect now depends on the
        // placement, as in the real game: 81 to 810 tower damage over 15 s
        // against a defended Princess Tower, most on its outer side.
        // UPSTREAM_REQUESTS.md items 32i and 32j.
        add(spell(110, "Graveyard", 5.0f, 4.0f, 0, 21, ']')
            .withRepeats(12, 5)
            .withSpellSpawnRing(3.3f)
            .withSpellSpawn(std::make_shared<PeriodicSpawnEffect>(graveyardSkeletonStats())));
        // Royal Delivery: a shielded Royal Recruit plus its own landing damage
        // (2026-08-04 balance update: 384, was 437; docs/LIVE_SYNC.md).
        add(spell(111, "Royal Delivery", 3.0f, 2.0f, 384, 8, '^')
            .withSpellSpawn(std::make_shared<PeriodicSpawnEffect>(royalDeliveryRecruitStats())));

        // === Compound troops ===
        // Goblin Gang and Rascals: both halves spawn via the primary unit plus
        // secondaryUnit.
        // First hit 0.6 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/april-balance-changes/).
        add(troop(112, "Goblin Gang", 3.0f, Archetype::MeleeSquad, 202, SPEED_VERY_FAST, 0.5f, 120, 11, 'g').withFirstHit(6)
            .withOffsets({ {-0.5f, -0.5f}, {0.5f, -0.5f}, {0.0f, 0.5f} })
            .withSecondaryUnit(
                // First hit 0.5 s: official balance notes (inferred: the same Spear Goblin character) (https://supercell.com/en/games/clashroyale/blog/release-notes/august-balance-changes-2/).
                troop(-27, "Spear Goblins", 0.0f, Archetype::RangedSquad, 133, SPEED_VERY_FAST, 5.0f, 81, 16, 'S').withFirstHit(5)
                    .withOffsets({ {-0.5f, 0.5f}, {0.5f, 0.5f}, {0.0f, -0.5f} })));
        // Rascals: only the Boy's stats are sourced; the two Girls reuse Spear
        // Goblins' ranged stats.
        // First hit 0.4 s: RoyaleAPI cr-api-data, hit_speed 1500 - load_time 1100 ms (docs/LIVE_SYNC.md).
        add(troop(113, "Rascals", 5.0f, Archetype::MeleeSquad, 1832, SPEED_MEDIUM, 1.2f, 217, 15, 'X').withFirstHit(4)
            .withSecondaryUnit(
                troop(-28, "Rascals", 0.0f, Archetype::RangedSquad, 133, 0.5f, 5.0f, 81, 17, 'r')
                    .withOffsets({ {-0.5f, 0.0f}, {0.5f, 0.0f} })));

        // Clone: duplicates every allied troop in radius; each clone has a
        // fresh id and 1 hp.
        add(spell(114, "Clone", 3.0f, 3.0f, 0, 8, ')')
            .withSpellClone());

        // === Champions ===
        // Mighty Miner: MeleeSquad like the Miner, but not deploy-anywhere.
        // Damage ramps 43 -> 204 -> 409 on one target (first stage 40 -> 43 in the 2026-08-04 update), 2 s per stage
        // (Liquipedia, 2025-01-08 patch), so mid 20 and full 40 ticks; `damage`
        // is the max. "Explosive Escape" (1 elixir, ~13 s cooldown): a lane
        // swap leaving a bomb that detonates after ~1 s for 332 with 1.8-tile
        // knockback. The bomb radius (2.5) is not sourced. See
        // MightyMinerEscapeEffect.
        add(troop(115, "Mighty Miner", 4.0f, Archetype::MeleeSquad, 2250, SPEED_MEDIUM, 1.6f, 409, 4, '\'')
            // 2026-08-04 balance update (final August changes, supercell.com): base damage 40 -> 43.
            .withDamageRamp(20, 40, 43.0f / 409.0f, 204.0f / 409.0f)
            // 2026-08-04 balance update: single-use per deployment (Supercell, Final August balance changes; docs/LIVE_SYNC.md). Cooldown is moot.
            .withChampionAbility(1.0f, 130, std::make_shared<MightyMinerEscapeEffect>(2.5f, 332, 10, 1.8f), 1));

        // Golden Knight's "Dashing Dash": up to 10 dashes to the nearest enemy
        // within 5.0 tiles, stopping at a Crown Tower (GoldenKnightDashEffect).
        // 2026-10-06 balance update: damage 161 -> 168, dash range 5.5 -> 5
        // tiles (docs/LIVE_SYNC.md); the live range is measured to the target's
        // edge, the engine's to its centre.
        // First hit 0.2 s: RoyaleAPI cr-api-data, hit_speed 900 - load_time 700 ms (docs/LIVE_SYNC.md).
        add(troop(116, "Golden Knight", 4.0f, Archetype::MeleeSquad, 1799, SPEED_MEDIUM, 1.2f, 168, 9, 'k').withFirstHit(2)
            // 2026-08-04 balance update: single-use per deployment (Supercell, Final August balance changes; docs/LIVE_SYNC.md). Cooldown is moot.
            .withChampionAbility(1.0f, 120, std::make_shared<GoldenKnightDashEffect>(335, 5.0f, 10), 1));

        // Skeleton King: collects up to 10 souls from any death within
        // soulCollectionRadius (not sourced); "Soul Summoning" spawns 6 + 1 per
        // soul, consuming them.
        // First hit 0.3 s: RoyaleAPI cr-api-data, hit_speed 1600 - load_time 1300 ms (docs/LIVE_SYNC.md).
        add(troop(117, "Skeleton King", 4.0f, Archetype::MeleeSquad, 2298, SPEED_MEDIUM, 1.2f, 180, 16, 'K').withFirstHit(3)
            .withSplash(1.3f)
            .withSoulCollection(5.0f, 10)
            .withChampionAbility(2.0f, 200,
                std::make_shared<SkeletonKingSoulSummonEffect>(skeletonKingSummonedSkeletonStats(), 6, 3.5f),
                1)); // 2026-08-04 balance update: single-use per deployment (Supercell, Final August balance changes; docs/LIVE_SYNC.md). Cooldown is moot.

        // Archer Queen's "Cloaking Cape": untargetable and ~2.8x attack speed
        // for 3.5 s.
        // First hit 0.3 s: RoyaleAPI cr-api-data, hit_speed 1200 - load_time 900 ms (docs/LIVE_SYNC.md).
        add(troop(118, "Archer Queen", 5.0f, Archetype::RangedSquad, 1000, SPEED_MEDIUM, 5.0f, 232, 12, 'Q').withFirstHit(3) // 2026-08-26 final August balance update: damage 225 -> 232 (docs/LIVE_SYNC.md)
            .withTargetsAir()
            // 2026-08-04 balance update: single-use per deployment (Supercell, Final August balance changes; docs/LIVE_SYNC.md). Cooldown is moot.
            .withChampionAbility(1.0f, 170, std::make_shared<ArcherQueenCloakEffect>(35, 1.0f / 2.8f), 1));

        // Monk: his 3-hit combo is flat damage here. "Pensive Protection" (65%
        // damage reduction for 4 s): see MonkDeflectEffect.
        // Cost 5 elixir (DeckShop; cr-api-data's 4 predates the live value).
        // First hit 0.2 s: RoyaleAPI cr-api-data, hit_speed 800 - load_time 600 ms (docs/LIVE_SYNC.md).
        add(troop(119, "Monk", 5.0f, Archetype::MeleeSquad, 2214, SPEED_MEDIUM, 1.2f, 140, 8, 'M').withFirstHit(2)
            // 2026-08-04 balance update: single-use per deployment (Supercell, Final August balance changes; docs/LIVE_SYNC.md). Cooldown is moot.
            .withChampionAbility(1.0f, 170, std::make_shared<MonkDeflectEffect>(0.35f, 40), 1));

        // Little Prince: hit speed ramps 1.2 s -> 0.8 s -> 0.4 s on one target
        // (3 attacks per stage, as tick thresholds). "Royal Rescue" summons the
        // Guardienne. 2026-08-04 balance update (final August changes, supercell.com): Hit Speed Conservation 0 -> 0.3 s
        // (3 ticks), modelled as a ramp grace period after the last landed hit.
        add(troop(120, "Little Prince", 3.0f, Archetype::RangedSquad, 698, 0.5f, 5.5f, 104, 12, 'p')
            .withTargetsAir()
            .withHitSpeedRamp(36, 60, 8.0f / 12.0f, 4.0f / 12.0f)
            .withRampGracePeriod(3)
            // 2026-08-04 balance update: single-use per deployment (Supercell, Final August balance changes; docs/LIVE_SYNC.md). Cooldown is moot.
            .withChampionAbility(3.0f, 300, std::make_shared<SpawnOnAbility>(guardienneStats()), 1));

        // Goblinstein: the Monster (front, building-only tank, carries the
        // ability) plus the Doctor (secondary, 3 tiles behind at a fixed
        // offset, a brief stun on hit). "Lightning Link": see
        // GoblinsteinLightningLinkEffect. Duration 5 s -> 4 s (2024-10-17), then
        // 4 s -> 3.5 s (2026-09-16 balance update, Supercell release notes;
        // docs/LIVE_SYNC.md): 35 ticks = 7 hits at the 5-tick interval.
        add(troop(121, "Goblinstein", 5.0f, Archetype::MeleeBuildingTargeter, 2240, 0.5f, 1.2f, 128, 15, 'G') // 2026-08-26 final August balance update: Monster hp 2385 -> 2240 (docs/LIVE_SYNC.md)
            // 2026-08-04 balance update: single-use per deployment (Supercell, Final August balance changes; docs/LIVE_SYNC.md). Cooldown is moot.
            .withChampionAbility(2.0f, 170, std::make_shared<GoblinsteinLightningLinkEffect>(2.0f, 94, 5, 35), 1) // 2026-08-04: 214 -> 188 damage per second = 94 per 5-tick pulse (docs/LIVE_SYNC.md)
            .withSecondaryUnit(
                // 2026-08-04 balance update: Doctor damage 92 -> 135 (docs/LIVE_SYNC.md).
                troop(-40, "Goblinstein", 0.0f, Archetype::RangedSquad, 721, 0.5f, 5.5f, 135, 18, 'D')
                    .withOffsets({ {0.0f, 3.0f} })
                    .withTargetsAir()
                    .withOnHit(std::make_shared<FreezeOnHit>(5, 0.0f))));

        // Boss Bandit: the Bandit's charge; dash invulnerability is inferred
        // from the Bandit, not sourced for this card. Always ignores the river,
        // as the Bandit does. "Getaway Grenade": brief invisibility and a
        // 6-tile teleport back toward her own side, limited to 2 uses per
        // deploy.
        add(troop(122, "Boss Bandit", 6.0f, Archetype::MeleeSquad, 2624, 0.7f, 0.8f, 245, 11, 'x')
            .withCharge(3.0f, 2.0f)
            .withChargeInvulnerability()
            .withIgnoresRiver()
            .withChampionAbility(1.0f, 30, std::make_shared<BossBanditGetawayGrenadeEffect>(10, 6.0f), 2));

        // === Evolutions ===
        // A deck holds the Evolution's id instead of the base card's. Unless
        // noted, each unlocks after 2 cycles and then evolves one play in
        // three, repeating (addEvolution).
        //
        // Wall Breakers ("Powder Barrels"): a defeated Wall Breaker's barrel
        // explodes in a 1.5-tile radius (the reduced Crown Tower damage is not
        // modelled), then a Runner rolls on to the nearest building
        // (runnerStats). One Runner per Wall Breaker. Sources disagree on a
        // walking-damage buff, so the stats stay unchanged.
        // 2026-10-06 balance update: 302 damage per Wall Breaker, both halves (docs/LIVE_SYNC.md).
        addEvolution(123,
            // First hit 0.2 s: RoyaleAPI cr-api-data, hit_speed 1200 - load_time 1000 ms (docs/LIVE_SYNC.md).
            troop(83, "Wall Breakers", 2.0f, Archetype::MeleeBuildingTargeter, 330, SPEED_VERY_FAST, 1.0f, 302, 12, '{').withFirstHit(2)
                .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} })
                .withSplash(1.5f)
                .withDieAfterFirstHit().withSightRange(7.0f),
            // First hit 0.2 s: RoyaleAPI cr-api-data, hit_speed 1200 - load_time 1000 ms (docs/LIVE_SYNC.md).
            troop(83, "Wall Breakers", 2.0f, Archetype::MeleeBuildingTargeter, 330, SPEED_VERY_FAST, 1.0f, 302, 12, '{').withFirstHit(2)
                .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} })
                .withSplash(1.5f)
                .withDieAfterFirstHit().withSightRange(7.0f)
                .withDeathEffect(std::make_shared<CompositeDeathEffect>(
                    std::vector<std::shared_ptr<IDeathEffect>>{
                        // 2026-04-06 balance update: area damage to troops 258 -> 192
                        // (-26%), radius unchanged (docs/LIVE_SYNC.md). The engine's
                        // single scalar also hits buildings and towers.
                        std::make_shared<AreaDamageOnDeath>(1.5f, 192),
                        std::make_shared<SpawnOnDeath>(runnerStats())
                    })),
            2, 1);

        // Zap ("Triple Shock"): since the 2024-10-08 balance update the 3rd zap
        // is removed, so two zaps, each at full damage (docs/LIVE_SYNC.md). The
        // growing radius between zaps is not modelled; the 3-tick gap is not
        // sourced ("the spell takes less time" has no number).
        addEvolution(124,
            spell(29, "Zap", 2.0f, 2.5f, 192, 3, 'Z').withCrownTowerDamage(48),
            // 48 per zap, as the base card: the notes do not state the Evolution's separately.
            spell(29, "Zap", 2.0f, 2.5f, 192, 3, 'Z').withRepeats(2, 3).withCrownTowerDamage(48),
            2, 1);

        // Skeletons ("Never-ending Horde"): see evolvedSkeletonChildStats.
        addEvolution(125,
            // Un-evolved half follows base Skeletons: first hit 0.5 s (5 ticks), as above.
            troop(24, "Skeletons", 1.0f, Archetype::MeleeSquad, 81, SPEED_FAST, 0.5f, 81, 11, 'k').withFirstHit(5)
                .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f} }),
            // Evolved half: first hit 0.4 s (4 ticks), official: https://supercell.com/en/games/clashroyale/blog/release-notes/balance-changes-nov-2023
            troop(24, "Skeletons", 1.0f, Archetype::MeleeSquad, 81, SPEED_FAST, 0.5f, 81, 11, 'k').withFirstHit(4)
                .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f} })
                .withOnHitSpawn(std::make_shared<CappedSpawnOnHitEffect>(evolvedSkeletonChildStats(), 8)),
            2, 1);

        // Bats: 122 hp (+50%); every attack heals two pulses of 38, 0.5 s
        // apart, overhealing up to 244 (wiki, level 11; UPSTREAM_REQUESTS.md
        // item 32g).
        addEvolution(126,
            troop(78, "Bats", 2.0f, Archetype::MeleeSquad, 81, SPEED_VERY_FAST, 1.0f, 81, 12, 't')
                .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f}, {0.3f, 0.5f}, {-0.3f, 0.5f} })
                .withFlying().withTargetsAir(),
            troop(78, "Bats", 2.0f, Archetype::MeleeSquad, 122, SPEED_VERY_FAST, 1.0f, 81, 12, 't')
                .withOffsets({ {0.0f, 0.0f}, {0.6f, 0.0f}, {-0.6f, 0.0f}, {0.3f, 0.5f}, {-0.3f, 0.5f} })
                .withFlying().withTargetsAir()
                .withHealOnHit(38, 244, 5),
            2, 1);

        // Bomber: +25% hp (380). "Bouncy Bomb" bounces twice at full damage,
        // approximated as a 3-target full-damage split.
        addEvolution(127,
            // First hit 0.2 s: RoyaleAPI cr-api-data, hit_speed 1800 - load_time 1600 ms (docs/LIVE_SYNC.md).
            troop(9, "Bomber", 2.0f, Archetype::RangedSquad, 304, SPEED_MEDIUM, 4.5f, 212, 18, 'b').withFirstHit(2), // 2026-09-08 balance update (supercell.com September balance changes, docs/LIVE_SYNC.md): damage 212
            // First hit 0.2 s: RoyaleAPI cr-api-data, hit_speed 1800 - load_time 1600 ms (docs/LIVE_SYNC.md).
            troop(9, "Bomber", 2.0f, Archetype::RangedSquad, 380, SPEED_MEDIUM, 4.5f, 212, 18, 'b').withFirstHit(2) // same 212 damage
                .withSplitTargets(3).withSplitTargetsFullDamage(),
            2, 1);

        // Archers: +1 tile range (6.0). "Power Shot": +25% damage at 4-6 tiles
        // (112 -> 140; was +50%, 168: 2026-08-04 balance update, docs/LIVE_SYNC.md).
        addEvolution(128,
            // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 900 - load_time 400 ms (docs/LIVE_SYNC.md).
            troop(1, "Archers", 3.0f, Archetype::RangedSquad, 304, SPEED_MEDIUM, 5.0f, 112, 9, 'A').withFirstHit(5)
                .withOffsets({ {0.0f, 0.0f}, {1.0f, 0.0f} }).withTargetsAir(),
            // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 900 - load_time 400 ms (docs/LIVE_SYNC.md).
            troop(1, "Archers", 3.0f, Archetype::RangedSquad, 304, SPEED_MEDIUM, 6.0f, 112, 9, 'A').withFirstHit(5)
                .withOffsets({ {0.0f, 0.0f}, {1.0f, 0.0f} }).withTargetsAir()
                .withRangeBandBonus(4.0f, 6.0f, 1.25f),
            2, 1);

        // Cannon ("Deploy Barrage"): a one-time burst on deploy: radius 2.0
        // (2025-05-05 balance update, 2.5 -> 2), damage 261 (2026-10-06 balance
        // update, 281 -> 261; docs/LIVE_SYNC.md). Knockback is not modelled.
        addEvolution(129,
            // First hit 1.0 s (10 ticks), the same as its hit speed: https://liquipedia.net/clashroyale/Cannon ; https://clashroyale.fandom.com/wiki/Cannon
            building(25, "Cannon", 3.0f, 824, 'C', 5.5f, 202, 10).withFirstHit(10),
            building(25, "Cannon", 3.0f, 824, 'C', 5.5f, 202, 10).withFirstHit(10)
                .withSpawnEffect(2.0f, 261),
            2, 1);

        // Firecracker: the real center and shrapnel sparks (low damage every
        // 0.25 s for 3 s) and the 15% slow are approximated as a
        // damage-over-time mark on whatever the shot hits. The per-tick damage
        // is not sourced; 3 ticks approximates 0.25 s.
        addEvolution(130,
            // 2025-08-06 balance update: sight range 8.5 -> 8 tiles on both blocks (docs/LIVE_SYNC.md).
            // First hit 0.7 s: RoyaleAPI cr-api-data, hit_speed 3000 - load_time 2350 ms (docs/LIVE_SYNC.md).
            troop(64, "Firecracker", 3.0f, Archetype::RangedSquad, 304, SPEED_FAST, 6.0f, 64, 30, '$').withFirstHit(7)
                .withTargetsAir().withSplash(1.5f).withRecoil(1.0f).withSightRange(8.0f),
            // First hit 0.7 s: RoyaleAPI cr-api-data, hit_speed 3000 - load_time 2350 ms (docs/LIVE_SYNC.md).
            troop(64, "Firecracker", 3.0f, Archetype::RangedSquad, 304, SPEED_FAST, 6.0f, 64, 30, '$').withFirstHit(7)
                .withTargetsAir().withSplash(1.5f).withRecoil(1.0f).withSightRange(8.0f)
                .withOnHit(std::make_shared<PoisonOnHit>(16, 30, 3)),
            2, 1);

        // Dart Goblin: the real escalating poison (51 / 115 / 307 per
        // consecutive dart, 1.25 s delay) is a flat stage-1 value per second
        // for 1 s (4 s until 2025-08-06): 64 since the 2026-06-01 balance update
        // ("Stage 1: 51 -> 64", supercell.com june-balance-changes-2026; stages 2
        // and 3 are not modelled, so stage 2's 115 -> 128 has no effect here). The
        // mark lives on the victim, so it outlasts the Goblin as in the real
        // card. withTargetsAir on both stat blocks: addEvolution inherits
        // nothing from the base registration.
        addEvolution(131,
            troop(20, "Dart Goblin", 3.0f, Archetype::RangedSquad, 261, SPEED_VERY_FAST, 6.5f, 143, 8, 'd')
                .withTargetsAir().withSightRange(7.0f), // 2026-03-02: 7.0 tiles, as the base card.
            troop(20, "Dart Goblin", 3.0f, Archetype::RangedSquad, 261, SPEED_VERY_FAST, 6.5f, 143, 8, 'd')
                .withTargetsAir()
                // 2025-08-06 balance update: poison cloud duration 4 s -> 1 s (40 -> 10 ticks); damage per tick unchanged here (docs/LIVE_SYNC.md).
                .withOnHit(std::make_shared<PoisonOnHit>(64, 10, 10)).withSightRange(7.0f), // 2026-03-02: 7.0 tiles.
            2, 1);

        // Goblin Barrel: the real second barrel on the mirrored lane is
        // approximated as 6 Goblins at one landing point.
        addEvolution(132,
            spell(109, "Goblin Barrel", 3.0f, 0.5f, 0, 8, '[')
                .withSpellSpawn(std::make_shared<PeriodicSpawnEffect>(goblinBarrelGoblinStats())),
            spell(109, "Goblin Barrel", 3.0f, 0.5f, 0, 8, '[')
                .withSpellSpawn(std::make_shared<PeriodicSpawnEffect>(goblinBarrelGoblinStats()
                    .withOffsets({ {-0.4f,0.0f},{0.4f,0.0f},{0.0f,0.4f},{-0.4f,0.4f},{0.4f,0.4f},{0.0f,-0.4f} }))),
            2, 1);

        // Skeleton Army: 16 skeletons (skeletonArmyEvolvedOffsets) and shadows
        // on death. The General's own shield is not modelled (a squad shares
        // one CardStats).
        addEvolution(133,
            troop(12, "Skeleton Army", 3.0f, Archetype::MeleeSquad, 81, SPEED_FAST, 0.5f, 81, 11, 's')
                .withOffsets(skeletonArmyOffsets()),
            troop(12, "Skeleton Army", 3.0f, Archetype::MeleeSquad, 81, SPEED_FAST, 0.5f, 81, 11, 's')
                .withOffsets(skeletonArmyEvolvedOffsets())
                .withDeathEffect(std::make_shared<SpawnOnDeath>(skeletonArmyShadowStats())),
            2, 1);

        // Skeleton Barrel: +25% hp (665). The real card drops 7 skeletons at
        // 75% hp and again on death; only the death drop is modelled, with 7.
        addEvolution(134,
            troop(89, "Skeleton Barrel", 3.0f, Archetype::MeleeBuildingTargeter, 532, SPEED_FAST, 1.0f, 81, 10, ',')
                .withFlying()
                .withDeathEffect(std::make_shared<SpawnOnDeath>(skeletonBarrelSkeletonStats())).withSightRange(7.7f),
            troop(89, "Skeleton Barrel", 3.0f, Archetype::MeleeBuildingTargeter, 665, SPEED_FAST, 1.0f, 81, 10, ',')
                .withFlying()
                .withDeathEffect(std::make_shared<SpawnOnDeath>(skeletonBarrelSkeletonStats().withOffsets({
                    {-0.6f,0.0f},{0.6f,0.0f},{-0.3f,0.3f},{0.3f,0.3f},{-0.3f,-0.3f},{0.3f,-0.3f},{0.0f,0.0f}
                }))).withSightRange(7.7f),
            2, 1);

        // Knight: the real shield (60% less damage while approaching) is a flat
        // permanent 35% reduction (withPassiveDamageReduction).
        addEvolution(135,
            // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1200 - load_time 700 ms (docs/LIVE_SYNC.md).
            troop(0, "Knight", 3.0f, Archetype::MeleeSquad, 1766, SPEED_MEDIUM, 1.2f, 202, 12, 'K').withFirstHit(5),
            // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1200 - load_time 700 ms (docs/LIVE_SYNC.md).
            troop(0, "Knight", 3.0f, Archetype::MeleeSquad, 1766, SPEED_MEDIUM, 1.2f, 202, 12, 'K').withFirstHit(5)
                .withPassiveDamageReduction(0.65f),
            2, 1);

        // Royal Ghost: a longer reveal window (now 2 s on both forms), and every
        // landed hit spawns 2 Souldiers. The real "only while invisible"
        // condition is not checked; she is cloaked for nearly all her attacks
        // anyway.
        // 2026-10-06 balance update: Royal Ghost 1152 hp, 263 damage, both halves (docs/LIVE_SYNC.md).
        addEvolution(136,
            // First hit 0.6 s: RoyaleAPI cr-api-data, hit_speed 1800 - load_time 1200 ms (docs/LIVE_SYNC.md).
            troop(47, "Royal Ghost", 3.0f, Archetype::MeleeSquad, 1152, SPEED_FAST, 1.2f, 263, 18, 'Q').withFirstHit(6)
                .withInvisibility(20).withIgnoresRiver(), // 2026-03-02: 2 s, as the base card.
            // First hit 0.6 s: RoyaleAPI cr-api-data, hit_speed 1800 - load_time 1200 ms (docs/LIVE_SYNC.md).
            troop(47, "Royal Ghost", 3.0f, Archetype::MeleeSquad, 1152, SPEED_FAST, 1.2f, 263, 18, 'Q').withFirstHit(6)
                .withInvisibility(20) // 2026-03-02 balance update: 1.8 s -> 2 s (docs/LIVE_SYNC.md)
                .withOnHitSpawn(std::make_shared<PeriodicSpawnEffect>(royalGhostSouldierStats()
                    .withOffsets({ {-0.4f, 0.0f}, {0.4f, 0.0f} })))
                .withIgnoresRiver(),
            2, 1);

        // Baby Dragon: the real aura (30% enemy slow, 30% ally speed-up) is
        // approximated as an ally damage buff per landed hit; there is no
        // movement-speed buff here.
        // Splash 1.2 -> 1.5 tiles on both forms (the base id 44 already had it):
        // 2025-01-07 balance update (docs/LIVE_SYNC.md).
        addEvolution(137,
            // First hit 0.3 s: RoyaleAPI cr-api-data, hit_speed 1500 - load_time 1200 ms (docs/LIVE_SYNC.md).
            troop(44, "Baby Dragon", 4.0f, Archetype::RangedSquad, 1152, SPEED_FAST, 3.5f, 168, 15, 'y').withFirstHit(3)
                .withFlying().withTargetsAir().withSplash(1.5f),
            // First hit 0.3 s: RoyaleAPI cr-api-data, hit_speed 1500 - load_time 1200 ms (docs/LIVE_SYNC.md).
            troop(44, "Baby Dragon", 4.0f, Archetype::RangedSquad, 1152, SPEED_FAST, 3.5f, 168, 15, 'y').withFirstHit(3)
                .withFlying().withTargetsAir().withSplash(1.5f)
                .withAllyBuffAura(4.0f, 1, 1.3f, 20, 1000000),
            2, 1);

        // Furnace ("Hot Spawning"): while attacking, Fire Spirits every 2.4 s
        // instead of 7, spawned to the side, left then right; otherwise the
        // base card. No damage change is sourced. The 1-tile side offset is not
        // sourced. UPSTREAM_REQUESTS.md item 32f.
        addEvolution(138,
            troop(70, "Furnace", 4.0f, Archetype::RangedSquad, 727, SPEED_MEDIUM, 5.5f, 179, 17, '/')
                .withTargetsAir()
                .withPeriodicEffect(70, std::make_shared<PeriodicSpawnEffect>(furnaceFireSpiritStats())),
            troop(70, "Furnace", 4.0f, Archetype::RangedSquad, 727, SPEED_MEDIUM, 5.5f, 179, 17, '/')
                .withTargetsAir()
                .withPeriodicEffect(70, std::make_shared<PeriodicSpawnEffect>(furnaceFireSpiritStats()))
                .withHotPeriodicInterval(24)
                .withAlternatingSpawnSide(1.0f),
            2, 1);

        // Goblin Cage: the real pull-and-hold is a hook plus a damage-over-time
        // mark; there is no hold mechanism.
        addEvolution(139,
            building(97, "Goblin Cage", 4.0f, 780, '6', 0.0f, 0, 100)
                .withDeathEffect(std::make_shared<SpawnOnDeath>(goblinCageBrawlerStats())),
            building(97, "Goblin Cage", 4.0f, 780, '6', 3.0f, 40, 20)
                .withHook(3.0f)
                .withOnHit(std::make_shared<PoisonOnHit>(30, 40, 10))
                .withDeathEffect(std::make_shared<SpawnOnDeath>(goblinCageBrawlerStats())),
            2, 1);

        // Musketeer ("Sniper Shot"): empowered shots deal 392 (217 x 1.808,
        // truncated), via burst-every-3rd, which repeats rather than stopping
        // after three. 2024-11-14 balance update: Sniper Damage 326 -> 392
        // (supercell.com november-balance-changes; docs/LIVE_SYNC.md). The
        // special range shape is not modelled.
        addEvolution(140,
            // First hit 0.7 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/february-balance-changes/).
            troop(6, "Musketeer", 4.0f, Archetype::RangedSquad, 721, SPEED_MEDIUM, 6.0f, 217, 10, 'U').withFirstHit(7)
                .withTargetsAir().withSightRange(6.0f),
            // First hit 0.7 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/february-balance-changes/).
            troop(6, "Musketeer", 4.0f, Archetype::RangedSquad, 721, SPEED_MEDIUM, 6.0f, 217, 10, 'U').withFirstHit(7)
                .withTargetsAir().withBurstAttack(3, 1.808f).withSightRange(6.0f),
            2, 1);

        // Wizard ("Fire Shield"): a 176 hp shield (2026-09-08 balance update,
        // down from 192; docs/LIVE_SYNC.md), damage 304 as the base card. The
        // shield-break explosion is not modelled.
        addEvolution(141,
            // 2024-08-06 balance update: Wizard hp 720 -> 754 (Supercell august-balance-changes; docs/LIVE_SYNC.md).
            // First hit 0.5 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/september-balance-changes-2027/).
            troop(11, "Wizard", 5.0f, Archetype::RangedSquad, 754, SPEED_MEDIUM, 5.5f, 304, 14, 'W').withFirstHit(5)
                .withTargetsAir().withSplash(1.5f),
            // First hit 0.5 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/september-balance-changes-2027/).
            troop(11, "Wizard", 5.0f, Archetype::RangedSquad, 754, SPEED_MEDIUM, 5.5f, 304, 14, 'W').withFirstHit(5)
                .withTargetsAir().withSplash(1.5f).withShield(176),
            2, 1);

        // Witch: the real heal when a spawned Skeleton dies is approximated as
        // healing on her own hits, capped at 1451 (2026-08-04 balance update:
        // Max Hitpoints 1040 -> 1451; docs/LIVE_SYNC.md).
        // Witch splash 1 -> 1.5 tiles on both forms: 2025-01-07 balance update
        // (docs/LIVE_SYNC.md).
        addEvolution(142,
            // First hit 0.7 s: RoyaleAPI cr-api-data, hit_speed 1100 - load_time 400 ms (docs/LIVE_SYNC.md).
            troop(71, "Witch", 5.0f, Archetype::RangedSquad, 839, SPEED_MEDIUM, 5.5f, 135, 11, ':').withFirstHit(7)
                .withTargetsAir().withSplash(1.5f)
                .withPeriodicEffect(70, std::make_shared<PeriodicSpawnEffect>(witchSkeletonStats())),
            // First hit 0.7 s: RoyaleAPI cr-api-data, hit_speed 1100 - load_time 400 ms (docs/LIVE_SYNC.md).
            troop(71, "Witch", 5.0f, Archetype::RangedSquad, 839, SPEED_MEDIUM, 5.5f, 135, 11, ':').withFirstHit(7)
                .withTargetsAir().withSplash(1.5f)
                .withPeriodicEffect(70, std::make_shared<PeriodicSpawnEffect>(witchSkeletonStats()))
                .withHealOnHit(20, 1451),
            2, 1);

        // Royal Giant: every attack also splashes 2.5 tiles. The knockback is
        // not modelled.
        addEvolution(143,
            troop(18, "Royal Giant", 6.0f, Archetype::RangedBuildingTargeter, 3164, SPEED_SLOW, 5.0f, 307, 18, 'Y')
                .withSightRange(7.5f),
            troop(18, "Royal Giant", 6.0f, Archetype::RangedBuildingTargeter, 3164, SPEED_SLOW, 5.0f, 307, 18, 'Y')
                .withSplash(2.5f).withSightRange(7.5f),
            2, 1);

        // Ice Spirit: +0.5 splash radius, applied as an absolute 1.7 on the
        // evolved form (the base card has no splash here). Both forms carry the
        // base 11-tick stun (official 7 Jul 2025 notes, 1.2 s -> 1.1 s, applied
        // to the Evolution too); the real delayed second pulse is not modelled.
        addEvolution(144,
            troop(72, "Ice Spirit", 1.0f, Archetype::RangedSquad, 215, SPEED_VERY_FAST, 2.5f, 110, 10, ';')
                .withTargetsAir()
                .withOnHit(std::make_shared<FreezeOnHit>(11, 0.0f))
                .withDieAfterFirstHit(),
            troop(72, "Ice Spirit", 1.0f, Archetype::RangedSquad, 215, SPEED_VERY_FAST, 2.5f, 110, 10, ';')
                .withTargetsAir()
                .withSplash(1.7f)
                .withOnHit(std::make_shared<FreezeOnHit>(11, 0.0f))
                .withDieAfterFirstHit(),
            2, 1);

        // Princess: every hit slows (the real card slows every third shot); the
        // lingering slow zone on death is not modelled.
        addEvolution(145,
            // First hit 0.3 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/march-balance-changes/).
            troop(61, "Princess", 3.0f, Archetype::RangedSquad, 261, SPEED_MEDIUM, 9.0f, 168, 30, '9').withFirstHit(3)
                .withTargetsAir().withSplash(1.5f).withSightRange(9.5f)
                .withDeployTime(12), // 2025-03-04: Deploy Time 1.0 s -> 1.2 s, as the base card
            // First hit 0.3 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/march-balance-changes/).
            troop(61, "Princess", 3.0f, Archetype::RangedSquad, 261, SPEED_MEDIUM, 9.0f, 168, 30, '9').withFirstHit(3)
                .withTargetsAir().withSplash(1.5f).withSightRange(9.5f)
                .withDeployTime(12) // 2025-03-04: Deploy Time 1.0 s -> 1.2 s, as the base card
                // 2026-08-04 balance update (final August changes, supercell.com): slow duration 7 s -> 5.5 s.
                .withOnHit(std::make_shared<FreezeOnHit>(55, 0.7f)),
            2, 1);

        // Hunter ("Netting Trap"): nets the nearest enemy for 3 s every 8 s (3
        // s net + 5 s recharge), via PeriodicFreezeNearestEffect.
        addEvolution(146,
            // 2024-08-06 balance update: Hunter hp 838 -> 884 (Supercell august-balance-changes; docs/LIVE_SYNC.md).
            // First hit 0.7 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/june-balance-changes-2/).
            troop(62, "Hunter", 4.0f, Archetype::RangedSquad, 884, SPEED_MEDIUM, 4.0f, 84, 22, '!').withFirstHit(7)
                .withTargetsAir().withSplash(1.5f).withRangeFalloff(0.5f),
            // First hit 0.7 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/june-balance-changes-2/).
            troop(62, "Hunter", 4.0f, Archetype::RangedSquad, 884, SPEED_MEDIUM, 4.0f, 84, 22, '!').withFirstHit(7)
                .withTargetsAir().withSplash(1.5f).withRangeFalloff(0.5f)
                .withPeriodicEffect(80, std::make_shared<PeriodicFreezeNearestEffect>(4.0f, 30)),
            2, 1);

        // Valkyrie ("Whirlwind Axe"): identical stats; every landed hit pulls
        // enemy troops within 5.0 tiles toward her (official December 2025
        // notes, Tornado Radius 5.5 -> 5), plus low damage to everyone
        // in that radius, Crown Towers included. Tornado damage 42 (2026-08-04
        // balance update, was 84; docs/LIVE_SYNC.md). Pull distance is not
        // sourced; the lingering zone is not modelled.
        addEvolution(147,
            // First hit 0.1 s: RoyaleAPI cr-api-data, hit_speed 1500 - load_time 1400 ms (docs/LIVE_SYNC.md).
            troop(10, "Valkyrie", 4.0f, Archetype::MeleeSquad, 1907, SPEED_MEDIUM, 1.2f, 266, 15, 'V').withFirstHit(1)
                .withSplash(1.5f),
            // First hit 0.1 s: RoyaleAPI cr-api-data, hit_speed 1500 - load_time 1400 ms (docs/LIVE_SYNC.md).
            troop(10, "Valkyrie", 4.0f, Archetype::MeleeSquad, 1907, SPEED_MEDIUM, 1.2f, 266, 15, 'V').withFirstHit(1)
                .withSplash(1.5f)
                .withOnHitPull(5.0f, 2.0f, 42),
            2, 1);

        // P.E.K.K.A. ("Butter-Heal"): the real heal on each kill, up to +66% hp
        // (6242), is a smaller heal on every hit; there is no on-kill hook.
        addEvolution(148,
            // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1800 - load_time 1300 ms (docs/LIVE_SYNC.md).
            troop(13, "P.E.K.K.A.", 7.0f, Archetype::MeleeSquad, 3760, SPEED_SLOW, 1.2f, 842, 18, 'E').withFirstHit(5)
                .withSightRange(5.0f),
            // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1800 - load_time 1300 ms (docs/LIVE_SYNC.md).
            troop(13, "P.E.K.K.A.", 7.0f, Archetype::MeleeSquad, 3760, SPEED_SLOW, 1.2f, 842, 18, 'E').withFirstHit(5)
                .withHealOnHit(40, 6242).withSightRange(5.0f),
            2, 1);

        // Minion Horde ("Dark Guard"): a member hit by a troop or spell turns
        // invisible for 3 s (DarkGuardOnDamageEffect). The real card triggers
        // on the first hit only; this refreshes on every hit.
        addEvolution(149,
            troop(42, "Minion Horde", 5.0f, Archetype::MeleeSquad, 230, SPEED_FAST, 2.5f, 107, 12, 'h')
                .withOffsets({ {-0.6f, -0.3f}, {0.0f, -0.3f}, {0.6f, -0.3f},
                               {-0.6f, 0.3f}, {0.0f, 0.3f}, {0.6f, 0.3f} })
                .withFlying().withTargetsAir(),
            troop(42, "Minion Horde", 5.0f, Archetype::MeleeSquad, 230, SPEED_FAST, 2.5f, 107, 12, 'h')
                .withOffsets({ {-0.6f, -0.3f}, {0.0f, -0.3f}, {0.6f, -0.3f},
                               {-0.6f, 0.3f}, {0.0f, 0.3f}, {0.6f, 0.3f} })
                .withFlying().withTargetsAir()
                .withOnDamageTaken(std::make_shared<DarkGuardOnDamageEffect>(30)),
            2, 1);

        // Royal Recruits: the real charge after the shield breaks (2.5 tiles
        // for 2x) is unconditional here.
        addEvolution(150,
            // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1300 - load_time 800 ms (docs/LIVE_SYNC.md).
            troop(77, "Royal Recruits", 7.0f, Archetype::MeleeSquad, 547, SPEED_MEDIUM, 1.0f, 133, 13, '@').withFirstHit(5)
                .withOffsets({ {-2.5f, 0.0f}, {-1.5f, 0.0f}, {-0.5f, 0.0f}, {0.5f, 0.0f}, {1.5f, 0.0f}, {2.5f, 0.0f} })
                .withShield(240),
            // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1300 - load_time 800 ms (docs/LIVE_SYNC.md).
            troop(77, "Royal Recruits", 7.0f, Archetype::MeleeSquad, 547, SPEED_MEDIUM, 1.0f, 133, 13, '@').withFirstHit(5)
                .withOffsets({ {-2.5f, 0.0f}, {-1.5f, 0.0f}, {-0.5f, 0.0f}, {0.5f, 0.0f}, {1.5f, 0.0f}, {2.5f, 0.0f} })
                .withShield(240)
                .withCharge(2.5f, 2.0f),
            2, 1);

        // Electro Dragon ("Infinite Bolts"): the real unbounded chain with
        // degrading effect is a fixed 6 full-damage targets.
        addEvolution(151,
            // First hit 0.5 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/october-balance-changes-2026/).
            troop(57, "Electro Dragon", 5.0f, Archetype::MeleeSquad, 1049, SPEED_MEDIUM, 3.5f, 192, 21, '5').withFirstHit(5)
                .withFlying().withTargetsAir()
                .withSplitTargets(3).withSplitTargetsFullDamage()
                .withOnHit(std::make_shared<FreezeOnHit>(5, 0.0f)),
            // First hit 0.5 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/october-balance-changes-2026/).
            troop(57, "Electro Dragon", 5.0f, Archetype::MeleeSquad, 1049, SPEED_MEDIUM, 3.5f, 192, 21, '5').withFirstHit(5)
                .withFlying().withTargetsAir()
                .withSplitTargets(6).withSplitTargetsFullDamage()
                .withOnHit(std::make_shared<FreezeOnHit>(5, 0.0f)),
            2, 1);

        // Mortar: attack period 4.7 s for both forms (2026-08-04 balance update (final August changes, supercell.com): base 5 s -> 4.7 s,
        // evolved 4 s -> 4.7 s). "Green Siege": a Goblin spawns with
        // every landed shot (onHitSpawnEffect). The launched Goblins deploy in
        // 0.5 s (5 ticks; official 2026-06-01 balance update, "Deploy Time:
        // 0.2s -> 0.5s"). A dedicated copy: Goblin Drill's Goblins share
        // goblinDrillGoblinStats() and are not touched by that post.
        addEvolution(152,
            // First hit 1.0 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/october-balance-changes/).
            building(93, "Mortar", 4.0f, 1369, 'R', 11.5f, 266, 47).withFirstHit(10)
                .withMinRange(3.5f).withSightRange(11.5f),
            // First hit 1.0 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/october-balance-changes/).
            building(93, "Mortar", 4.0f, 1369, 'R', 11.5f, 266, 47).withFirstHit(10)
                .withMinRange(3.5f).withSightRange(11.5f)
                .withOnHitSpawn(std::make_shared<PeriodicSpawnEffect>(goblinDrillGoblinStats().withDeployTime(5))),
            2, 1);

        // Goblin Drill: the real resurfacing at 66% / 33% hp (leaving a Goblin
        // each time) is approximated as more Goblins on death.
        addEvolution(153,
            building(98, "Goblin Drill", 4.0f, 1313, '7', 0.0f, 0, 100)
                .withPeriodicEffect(30, std::make_shared<PeriodicSpawnEffect>(goblinDrillGoblinStats()))
                .withDeployAnywhere()
                .withSpawnEffect(2.0f, 84).withSpawnEffectCrownTowerDamage(20) // as the base card
                .withDeathEffect(std::make_shared<SpawnOnDeath>(goblinDrillDeathGoblinStats())),
            building(98, "Goblin Drill", 4.0f, 1313, '7', 0.0f, 0, 100)
                .withPeriodicEffect(30, std::make_shared<PeriodicSpawnEffect>(goblinDrillGoblinStats()))
                .withDeployAnywhere()
                .withSpawnEffect(2.0f, 84).withSpawnEffectCrownTowerDamage(20) // as the base card
                .withDeathEffect(std::make_shared<SpawnOnDeath>(goblinDrillDeathGoblinStats()
                    .withOffsets({ {-0.4f,0.0f},{0.4f,0.0f},{0.0f,0.4f},{0.0f,-0.4f} }))),
            2, 1);

        // Tesla ("Electro Pulse"): 6-tile radius, 148 damage (2026-05-04 balance
        // update; docs/LIVE_SYNC.md) and a 0.5 s stun. There is no hiding to
        // emerge from here, so it fires once on deploy.
        addEvolution(154,
            // First hit 0.4 s: RoyaleAPI cr-api-data, hit_speed 1100 - load_time 700 ms (docs/LIVE_SYNC.md).
            building(26, "Tesla", 4.0f, 1182, 'T', 5.5f, 220, 11).withFirstHit(4).withTargetsAir(),
            // First hit 0.4 s: RoyaleAPI cr-api-data, hit_speed 1100 - load_time 700 ms (docs/LIVE_SYNC.md).
            building(26, "Tesla", 4.0f, 1182, 'T', 5.5f, 220, 11).withFirstHit(4).withTargetsAir()
                .withSpawnEffect(6.0f, 148, std::make_shared<FreezeOnHit>(5, 0.0f)),
            2, 1);

        // Barbarians: the base 716 hp, the Evolution's hp bonus cut to 0% by
        // the official 4 Aug 2026 notes (was +10%, 760). "Blade Rage": +35%
        // attack speed for 5 s (3 s until the 4 Aug 2026 notes), refreshed on
        // each attack; the official 24 Oct 2025 notes cut the boost to +30%
        // (0.77 ~= 1/1.30; docs/LIVE_SYNC.md). The
        // movement-speed boost is not modelled.
        addEvolution(155,
            troop(8, "Barbarians", 5.0f, Archetype::MeleeSquad, 716, SPEED_MEDIUM, 0.7f, 192, 14, 'B')
                .withOffsets({ {0.0f, 0.0f}, {-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, 0.5f} }),
            troop(8, "Barbarians", 5.0f, Archetype::MeleeSquad, 716, SPEED_MEDIUM, 0.7f, 192, 14, 'B')
                .withOffsets({ {0.0f, 0.0f}, {-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, 0.5f} })
                .withSelfHasteOnHit(50, 0.77f),
            2, 1);

        // Lumberjack: the ghost (lumberjackGhostStats) composed with the Rage
        // drop (30% boost, official 24 Oct 2025 balance post; docs/LIVE_SYNC.md).
        addEvolution(156,
            // First hit 0.4 s: RoyaleAPI cr-api-data, hit_speed 800 - load_time 400 ms (docs/LIVE_SYNC.md).
            troop(21, "Lumberjack", 4.0f, Archetype::MeleeSquad, 1282, SPEED_VERY_FAST, 0.7f, 256, 8, 'l').withFirstHit(4)
                .withDeathEffect(std::make_shared<AreaBuffOnDeath>(2.5f, 1.30f, 55)),
            // First hit 0.4 s: RoyaleAPI cr-api-data, hit_speed 800 - load_time 400 ms (docs/LIVE_SYNC.md).
            troop(21, "Lumberjack", 4.0f, Archetype::MeleeSquad, 1282, SPEED_VERY_FAST, 0.7f, 256, 8, 'l').withFirstHit(4)
                .withDeathEffect(std::make_shared<CompositeDeathEffect>(
                    std::vector<std::shared_ptr<IDeathEffect>>{
                        std::make_shared<AreaBuffOnDeath>(2.5f, 1.30f, 55),
                        std::make_shared<SpawnOnDeath>(lumberjackGhostStats())
                    })),
            2, 1);

        // Executioner ("Axe Smash"): unlocks after 1 cycle. Close Quarters: 240
        // damage within 2.5 tiles (the range-band bonus), 179 beyond. 2026-07-06
        // balance update: 268 -> 240 damage, 3.5 -> 2.5 tiles (docs/LIVE_SYNC.md).
        // 240 is expressed as 1.3414f on the base 179 (static_cast<int> truncation;
        // 1.34f would give 239). The 1.5-tile knockback on the
        // outgoing hit is not modelled.
        addEvolution(157,
            troop(36, "Executioner", 5.0f, Archetype::RangedSquad, 1280, SPEED_MEDIUM, 4.5f, 179, 24, 'x')
                .withTargetsAir().withBoomerang(15),
            troop(36, "Executioner", 5.0f, Archetype::RangedSquad, 1280, SPEED_MEDIUM, 4.5f, 179, 24, 'x')
                .withTargetsAir().withBoomerang(15)
                .withRangeBandBonus(0.0f, 2.5f, 1.3414f),
            1, 1);

        // Giant Snowball ("Snow Roll"): trades the base push for a 4.0-tile
        // pull (4.5 until the 2026-08-04 balance update, docs/LIVE_SYNC.md), gathering enemies in its path. "Untargetable while trapped" is
        // not modelled.
        addEvolution(158,
            spell(100, "Giant Snowball", 2.0f, 2.5f, 179, 8, '!')
                .withCrownTowerDamage(45) // June 2026 balance update, as the base card
                // Base half: 3 s slow, as the base card above (2026-05-04).
                .withSpellOnHit(std::make_shared<FreezeOnHit>(30, 0.70f)) // 2025-10-06: base slow 30%
                .withKnockback(1.0f),
            spell(100, "Giant Snowball", 2.0f, 2.5f, 179, 8, '!')
                .withCrownTowerDamage(45) // June 2026 balance update, as the base card
                // 2026-05-04 balance update: slow 4 s -> 3 s (30 ticks) to
                // match the base form (docs/LIVE_SYNC.md). Factor 0.7 unchanged.
                .withSpellOnHit(std::make_shared<FreezeOnHit>(30, 0.7f))
                .withKnockback(-4.0f),
            2, 1);

        // Goblin Giant ("Sack-trick"): the real knife Goblins every 2.2 s below
        // 50% hp are an unconditional periodic spawn from deploy; a periodic
        // effect cannot be gated on hp.
        addEvolution(159,
            // First hit 0.8 s: RoyaleAPI cr-api-data, hit_speed 1500 - load_time 700 ms (docs/LIVE_SYNC.md).
            troop(88, "Goblin Giant", 6.0f, Archetype::MeleeBuildingTargeter, 3110, SPEED_MEDIUM, 1.2f, 176, 15, '`').withFirstHit(8)
                .withSecondaryUnit(goblinGiantSpearGoblinsStats()).withSightRange(7.5f),
            // First hit 0.8 s: RoyaleAPI cr-api-data, hit_speed 1500 - load_time 700 ms (docs/LIVE_SYNC.md).
            troop(88, "Goblin Giant", 6.0f, Archetype::MeleeBuildingTargeter, 3110, SPEED_MEDIUM, 1.2f, 176, 15, '`').withFirstHit(8)
                .withSecondaryUnit(goblinGiantSpearGoblinsStats()).withSightRange(7.5f)
                .withPeriodicEffect(22, std::make_shared<PeriodicSpawnEffect>(goblinDrillGoblinStats())),
            2, 1);

        // Mega Knight: identical stats. "Mega Uppercut" (hit targets launched 4
        // tiles toward the enemy Crown Tower) is not modelled: there is no
        // "knock the target toward a point" primitive.
        addEvolution(160,
            // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1700 - load_time 1200 ms (docs/LIVE_SYNC.md).
            troop(48, "Mega Knight", 7.0f, Archetype::MeleeSquad, 3993, SPEED_MEDIUM, 1.2f, 268, 17, 'X').withFirstHit(5)
                .withSplash(1.5f).withSpawnEffect(1.3f, 430).withJump(3.5f, 5.0f, 2.0f, 2.2f).withIgnoresRiver(),
            // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1700 - load_time 1200 ms (docs/LIVE_SYNC.md).
            troop(48, "Mega Knight", 7.0f, Archetype::MeleeSquad, 3993, SPEED_MEDIUM, 1.2f, 268, 17, 'X').withFirstHit(5)
                .withSplash(1.5f).withSpawnEffect(1.3f, 430).withJump(3.5f, 5.0f, 2.0f, 2.2f).withIgnoresRiver(),
            2, 1);

        // Battle Ram: the real card's death-spawned Barbarians are themselves
        // evolved; here they get a flat stat buff (the Evolution machinery is
        // keyed to deck cycling). "Head-First Ram": once the charge connects,
        // every later hit keeps double damage (withStickyCharge). The contact
        // damage and knockback to troops along the way are not modelled.
        // Until it evolves it is the ordinary Battle Ram and breaks on its
        // first hit like card 81; the evolved Ram does NOT break, it keeps
        // swinging until destroyed (UPSTREAM_REQUESTS.md item 34).
        // Both forms carry the base card's stats (item 35).
        addEvolution(161,
            troop(81, "Battle Ram", 4.0f, Archetype::MeleeBuildingTargeter, 967, SPEED_MEDIUM, 0.5f, 286, 14, '^')
                .withDeathEffect(std::make_shared<SpawnOnDeath>(battleRamBarbarianStats()))
                .withCharge(3.0f, 2.0f).withChargeSpeed(2.0f).withDieAfterFirstHit().withSightRange(6.5f),
            troop(81, "Battle Ram", 4.0f, Archetype::MeleeBuildingTargeter, 967, SPEED_MEDIUM, 0.5f, 286, 14, '^')
                .withDeathEffect(std::make_shared<SpawnOnDeath>(battleRamEvolvedBarbarianStats()))
                .withCharge(3.0f, 2.0f).withChargeSpeed(2.0f).withStickyCharge().withSightRange(6.5f),
            2, 1);

        // Royal Hogs: identical stats. "Hog Flight" (spawn flying, then fall
        // for 155% damage on the first attack) is not modelled, and how it
        // would stack with the charge bonus is unsourced.
        addEvolution(162,
            // First hit 0.4 s: RoyaleAPI cr-api-data, hit_speed 1200 - load_time 800 ms (docs/LIVE_SYNC.md).
            troop(82, "Royal Hogs", 5.0f, Archetype::MeleeBuildingTargeter, 837, SPEED_VERY_FAST, 1.0f, 74, 12, '_').withFirstHit(4)
                .withOffsets({ {-1.0f, -0.3f}, {-0.3f, 0.3f}, {0.3f, -0.3f}, {1.0f, 0.3f} })
                .withCharge(3.0f, 2.0f).withSightRange(9.5f).withIgnoresRiver(),
            // First hit 0.4 s: RoyaleAPI cr-api-data, hit_speed 1200 - load_time 800 ms (docs/LIVE_SYNC.md).
            troop(82, "Royal Hogs", 5.0f, Archetype::MeleeBuildingTargeter, 837, SPEED_VERY_FAST, 1.0f, 74, 12, '_').withFirstHit(4)
                .withOffsets({ {-1.0f, -0.3f}, {-0.3f, 0.3f}, {0.3f, -0.3f}, {1.0f, 0.3f} })
                .withCharge(3.0f, 2.0f).withSightRange(9.5f).withIgnoresRiver(),
            2, 1);

        // Inferno Dragon ("Damage Charge-up"): identical stats to id 56. Losing
        // or having no target holds the ramp stage for a 7 s grace period (9 s
        // -> 7 s, "Charge Duration", official 2026-06-01 balance update), so a
        // new target inherits it (a stun still resets at once); and a fourth
        // stage after 20 s of continuous attacking deals double the max. The
        // sources disagree (49 ticks vs 20 s); the seconds figure is used.
        addEvolution(163,
            troop(56, "Inferno Dragon", 4.0f, Archetype::MeleeSquad, 1295, SPEED_MEDIUM, 5.0f, 422, 4, '4')
                .withFlying().withTargetsAir()
                .withDamageRamp(15, 30, 0.083f, 0.284f),
            troop(56, "Inferno Dragon", 4.0f, Archetype::MeleeSquad, 1295, SPEED_MEDIUM, 5.0f, 422, 4, '4')
                .withFlying().withTargetsAir()
                .withDamageRamp(15, 30, 0.083f, 0.284f)
                .withRampStage4(200, 2.0f)
                .withRampGracePeriod(70),
            2, 1);

        // Elite Barbarians Evolution (id 182): body identical to id 17
        // (single-source: trophycoach.com/guides/elite-barbarians-evolution-guide,
        // "Identical Stats"). Every 5 s (single-source: same guide; official text
        // only says "every few seconds") each barbarian hurls a spear at the
        // nearest enemy GROUND target (single-source: same guide) whose
        // surface-to-surface distance is 3-4.5 tiles (official 8 Sep 2026:
        // "Spear Range: 3.5-5 -> 3-4.5 tiles"; the guide says the range has a
        // minimum as well as a maximum) for 220 (official 26 Aug 2026:
        // "Spear Damage: 284 -> 220"). Cycles to evolve 1: single-source (same
        // guide); evolvedUses 1 is the engine convention, unsourced. NOT modelled:
        // the Rage path (magnitude, radius, geometry unsourced; official duration
        // 2 s, 8 Sep 2026), spear flight time, the separate 6-tile spear sight
        // (official 6 Oct 2026; sightRange stays 5.5 because test_sight_range.cpp
        // pins an Evolution to its base sight), and any Crown Tower damage
        // difference (unsourced; assumed equal).
        addEvolution(182,
            troop(17, "Elite Barbarians", 6.0f, Archetype::MeleeSquad, 1341, SPEED_FAST, 1.2f, 384, 14, 'e').withFirstHit(5)
                .withOffsets({ {-0.5f, 0.0f}, {0.5f, 0.0f} }),
            troop(17, "Elite Barbarians", 6.0f, Archetype::MeleeSquad, 1341, SPEED_FAST, 1.2f, 384, 14, 'e').withFirstHit(5)
                .withOffsets({ {-0.5f, 0.0f}, {0.5f, 0.0f} })
                .withPeriodicEffect(50, std::make_shared<EliteBarbarianSpearEffect>(3.0f, 4.5f, 220)),
            1, 1);
        // Partly built (no Rage path, spear flight time or separate spear
        // sight), so random decks never deal it.
        markAbilityPending(182);

        // Giant ("Enrage", id 187): stats identical to id 2 (single-source:
        // timesaver.gg/blog/clash-royale-evo-giant says the Evolution matches
        // the base card; Supercell's 2026-10-27 post gives none). Unlocks after
        // 1 cycle (same single source). Once it has taken damage down to 50% of
        // its hp (single-source: timesaver.gg, citing RoyaleTools; Supercell
        // says only "take enough damage") it is enraged for good and leaves a
        // Rage trail: friendly units standing within the trail's radius get
        // +30% movement and attack speed (single-source: timesaver.gg) for 1.5 s
        // (same source), refreshed while they stay in it. The radius is 2 tiles
        // at enrage and grows 0.5 per attack to 3.5 (single-source: timesaver.gg,
        // creator testing; Supercell: "each of his attacks make that Rage trail
        // bigger"). The Giant itself and buildings/towers are buffed too (creator
        // testing, same page, uncontradicted). Not modelled: trail-circle drop
        // interval (the trail is the Giant's last 1.5 s of positions, no invented
        // number), evolved-uses beyond 1 (engine convention, unsourced).
        addEvolution(187,
            troop(2, "Giant", 5.0f, Archetype::MeleeBuildingTargeter, 3968, SPEED_SLOW, 1.2f, 253, 15, 'G').withFirstHit(5)
                .withSightRange(7.5f),
            troop(2, "Giant", 5.0f, Archetype::MeleeBuildingTargeter, 3968, SPEED_SLOW, 1.2f, 253, 15, 'G').withFirstHit(5)
                .withSightRange(7.5f)
                .withEnrageTrail(0.5f, {2.0f, 2.5f, 3.0f, 3.5f}, 15, 1.3f),
            1, 1);

        // Electro Giant Evolution ("Zap Pack 2.0", released 2026-10-05, official:
        // https://supercell.com/en/games/clashroyale/blog/release-notes/new-season-shocktober/).
        // The expanding de-level pulse is NOT modelled: the official post is
        // qualitative and no sourced interval, radius, first-pulse time or
        // de-level magnitude exists (third-party figures conflict: 3.5 s vs 5.5 s
        // first pulse, ~9% vs ~10% per level). Until they are supplied the evolved
        // form is the live base card (id 84) restated, including its periodic
        // AreaStun approximation of the base Zap Pack. Cycles to evolve 2 and
        // evolvedUses 1 are the ENGINE'S evolution defaults, NOT sourced for this
        // card (the one claim of "cycle 1" was a single unverified page).
        addEvolution(186,
            troop(84, "Electro Giant", 7.0f, Archetype::MeleeBuildingTargeter, 3952, SPEED_SLOW, 1.0f, 184, 18, '|').withFirstHit(10)
                .withSplash(1.5f)
                .withPeriodicEffect(50, std::make_shared<AreaStunEffect>(2.5f, 5)),
            troop(84, "Electro Giant", 7.0f, Archetype::MeleeBuildingTargeter, 3952, SPEED_SLOW, 1.0f, 184, 18, '|').withFirstHit(10)
                .withSplash(1.5f)
                .withPeriodicEffect(50, std::make_shared<AreaStunEffect>(2.5f, 5)),
            2, 1);
        // Partly built (no de-level pulse), so random decks never deal it.
        markAbilityPending(186);

        // === Mirror ===
        // Registered so it can occupy a deck slot. GameManager::playCard
        // substitutes the last card played (cost + 1), so this entry's own
        // cost, flags and footprint are never read at play time.
        add(spell(164, "Mirror", 3.0f, 0.0f, 0, 0, '='));

        // === Mirror Evolution ===
        // Supercell 2026-10-27 King vs. King post: "Evolved Mirror copies the
        // last card you played, but evolved, with 1 extra level." GameManager::
        // playCard substitutes the copied card's Evolution, so this stub's own
        // spawn and footprint are never read. The +1 level is a no-op (the
        // engine has no levels; base Mirror models only the +1 elixir).
        // Cycles-to-evolve 2: single source, timesaver.gg/blog/clash-royale-evo-mirror
        // ("It charges in 2 cycles", citing the wiki); not contradicted.
        // Evolved uses 1: NOT a sourced value (no source states it); 1 is the
        // integer's structural minimum. Unresolved.
        addEvolution(188,
            spell(188, "Mirror", 3.0f, 0.0f, 0, 0, '='),
            spell(188, "Mirror", 3.0f, 0.0f, 0, 0, '='),
            2, 1);

        // === Spirit Empress ===
        // Registered at the ground form's cost (3.0) for the hand display and
        // placement; playCard always spawns from SpiritEmpressForms.h instead.
        // Not a Champion: it has no activated ability.
        // 2026-05-04 balance update: 1121 hp on both forms (ground 1152 -> 1121,
        // air 1177 -> 1121; docs/LIVE_SYNC.md). Was 926, a Liquipedia level-9 value.
        // Damage 320 (2026-09-08 balance update, level 11; docs/LIVE_SYNC.md); the
        // forms in SpiritEmpressForms.h carry the same value.
        add(troop(165, "Spirit Empress", 3.0f, Archetype::MeleeSquad, 1121, SPEED_FAST, 1.2f, 320, 12, '<')); // speed matches the ground form (2026-03-02: Fast)

        // === Heroes ===
        // A Hero gives an existing troop a second, ability-carrying form, using
        // the Champion machinery via isHero. Base stats are copied verbatim
        // from the engine's own base-card registration; only the ability is
        // added.

        // Hero Mini P.E.K.K.A. (base: id 5). "Breakfast Boost": a flat one-time
        // hp and damage boost (HeroMiniPekkaBoostEffect).
        // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1600 - load_time 1100 ms (docs/LIVE_SYNC.md).
        add(troop(170, "Hero Mini P.E.K.K.A.", 4.0f, Archetype::MeleeSquad, 1390, SPEED_FAST, 0.8f, 755, 16, 'M').withFirstHit(5)
            .withHeroAbility(1.0f, 0, std::make_shared<HeroMiniPekkaBoostEffect>(210, 1.15f), 1));

        // Hero Musketeer (base: id 6). "Trusty Turret": a short-range turret
        // with a 10 s lifetime (heroMusketeerTurretStats).
        // First hit 0.7 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/february-balance-changes/).
        add(troop(168, "Hero Musketeer", 4.0f, Archetype::RangedSquad, 721, SPEED_MEDIUM, 6.0f, 217, 10, 'U').withFirstHit(7)
            .withTargetsAir().withSightRange(6.0f)
            // 2026-08-04 balance update: single-use per deployment (Supercell, Final August balance changes; docs/LIVE_SYNC.md). Cooldown is moot.
            .withHeroAbility(3.0f, 220, std::make_shared<SpawnOnAbility>(heroMusketeerTurretStats()), 1));

        // Hero Goblins (base: id 4). "Banner Brigade" (1 elixir, one use,
        // within 5 s of the last goblin dying; 7 s before the 2026-02-23
        // balance update, "Banner Duration: 7s -> 5s", docs/LIVE_SYNC.md): a fresh squad at the death
        // position. withPostDeathAbility, since there is no alive-path ability.
        // The new squad uses plain Goblins stats, so it cannot chain.
        // 2026-08-04 balance update: Goblin damage 120 -> 125, and Banner
        // Brigade respawns 2 Goblins instead of 3 (docs/LIVE_SYNC.md; the
        // earlier 4 -> 3 change was 2026-02-23, still 4 here).
        // First hit 0.6 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/april-balance-changes/).
        add(troop(172, "Hero Goblins", 2.0f, Archetype::MeleeSquad, 202, SPEED_VERY_FAST, 0.5f, 125, 11, 'g').withFirstHit(6)
            .withOffsets({ {-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, 0.5f} })
            .withPostDeathAbility(1.0f, 50, std::make_shared<PeriodicSpawnEffect>(
                troop(-48, "Goblins", 0.0f, Archetype::MeleeSquad, 202, SPEED_VERY_FAST, 0.5f, 125, 11, 'g')
                    .withOffsets({ {-0.5f, 0.0f}, {0.5f, 0.0f} }))));

        // Hero Knight (base: id 0). "Triumphant Taunt" (2 elixir, 25 s
        // cooldown): a shield, and enemies within 6.5 tiles must attack him for
        // 5 s (HeroKnightTauntEffect). 2026-07-06 balance update: shield 768 ->
        // 512 hp (docs/LIVE_SYNC.md).
        // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1200 - load_time 700 ms (docs/LIVE_SYNC.md).
        add(troop(166, "Hero Knight", 3.0f, Archetype::MeleeSquad, 1766, SPEED_MEDIUM, 1.2f, 202, 12, 'K').withFirstHit(5)
            // 2026-08-04 balance update: single-use per deployment (Supercell, Final August balance changes; docs/LIVE_SYNC.md). Cooldown is moot.
            .withHeroAbility(2.0f, 250, std::make_shared<HeroKnightTauntEffect>(512, 50, 6.5f), 1));

        // Hero Wizard (base: id 11). "Fiery Flight" (1 elixir, 20 s cooldown):
        // flies for 3.5 s (2026-09-08 balance update, down from 5 s; docs/LIVE_SYNC.md),
        // each landed attack pulsing a damaging, pulling tornado on the target
        // (HeroWizardFieryFlightEffect). The 0.5-tile pull approximates the
        // sourced "~50% pull strength". Damage 304 as the base card.
        // 2024-08-06 balance update: Wizard hp 720 -> 754 (Supercell august-balance-changes; docs/LIVE_SYNC.md).
        // First hit 0.5 s: official balance notes (https://supercell.com/en/games/clashroyale/blog/release-notes/september-balance-changes-2027/).
        add(troop(167, "Hero Wizard", 5.0f, Archetype::RangedSquad, 754, SPEED_MEDIUM, 5.5f, 304, 14, 'W').withFirstHit(5)
            .withTargetsAir().withSplash(1.5f)
            // 2026-08-04 balance update: single-use per deployment (Supercell, Final August balance changes; docs/LIVE_SYNC.md). Cooldown is moot.
            .withHeroAbility(1.0f, 200, std::make_shared<HeroWizardFieryFlightEffect>(35, 4.0f, 20, 0.5f), 1));

        // Hero Giant (base: id 2). "Heroic Hurl" (2 elixir, 14 s cooldown):
        // throws the highest-HP enemy troop within 3.0 tiles (not sourced) to
        // the other lane and stuns it 2 s (HeroGiantHurlEffect).
        // First hit 0.5 s: RoyaleAPI cr-api-data, hit_speed 1500 - load_time 1000 ms (docs/LIVE_SYNC.md).
        add(troop(169, "Hero Giant", 5.0f, Archetype::MeleeBuildingTargeter, 3968, SPEED_SLOW, 1.2f, 253, 15, 'G').withFirstHit(5)
            .withSightRange(7.5f)
            // 2026-08-04 balance update: single-use per deployment (Supercell, Final August balance changes; docs/LIVE_SYNC.md). Cooldown is moot.
            .withHeroAbility(2.0f, 140, std::make_shared<HeroGiantHurlEffect>(3.0f, 20), 1));

        // Hero Mega Minion (base: id 43). "Wounding Warp" (2 elixir, one use,
        // locked for the first 1.5 s): teleports to the lowest-HP enemy
        // anywhere and deals bonus damage (HeroMegaMinionWarpEffect). Warp
        // damage 422 per the 2026-09-08 balance update (399 -> 422; docs/LIVE_SYNC.md).
        // First hit 0.4 s: RoyaleAPI cr-api-data, hit_speed 1500 - load_time 1100 ms (docs/LIVE_SYNC.md).
        add(troop(173, "Hero Mega Minion", 3.0f, Archetype::MeleeSquad, 837, SPEED_MEDIUM, 1.6f, 312, 15, 'F').withFirstHit(4)
            .withFlying().withTargetsAir()
            .withHeroAbility(2.0f, 0, std::make_shared<HeroMegaMinionWarpEffect>(422), 1)
            .withInitialAbilityCooldown(15));

        // Hero Magic Archer (base: id 63). "Triple Threat" (2 elixir, 25 s
        // cooldown): dashes back 3.5 tiles (2026-07-06 balance update: 5 -> 3.5,
        // docs/LIVE_SYNC.md), leaves a decoy, and gains a 7 s multi-shot window
        // (HeroMagicArcherTripleThreatEffect).
        // hp 531 as the base card (2024-10-08 balance update, docs/LIVE_SYNC.md).
        // 2026-08-04 balance update (final August changes, supercell.com): damage 143 -> 135 (follows the base form).
        add(troop(171, "Hero Magic Archer", 4.0f, Archetype::RangedSquad, 531, SPEED_MEDIUM, 7.0f, 135, 11, '#')
            .withTargetsAir()
            .withSplash(0.25f).withLineSplash(11.0f).withSightRange(7.5f)
            .withHeroAbility(2.0f, 250, std::make_shared<HeroMagicArcherTripleThreatEffect>(
                3.5f, heroMagicArcherDecoyStats(), 70), 1)); // 2026-08-04 balance update: single-use per deployment (Supercell, Final August balance changes; docs/LIVE_SYNC.md). Cooldown is moot.

        // Hero Ice Golem (base: id 40, including the slow on its death
        // explosion; 1228 hp and 25-tick slow as of 2026-09-08, docs/LIVE_SYNC.md). "Snowstorm" (2 elixir, 17 s cooldown): three staggered
        // blasts in a 4-tile radius, the third a 1.5 s freeze, with reduced
        // Crown Tower damage (HeroIceGolemSnowstormEffect). Per-blast damage
        // and slow are not sourced. 2026-04-06 balance update: Snowstorm's
        // knockback removed (ctor arg 3 = 0, docs/LIVE_SYNC.md). 2026-08-04
        // balance update: damage per blast 38 -> 69 (the engine's 80 was unsourced).
        // First hit 1.0 s: RoyaleAPI cr-api-data, hit_speed 2500 - load_time 1500 ms (docs/LIVE_SYNC.md).
        add(troop(175, "Hero Ice Golem", 2.0f, Archetype::MeleeBuildingTargeter, 1228, SPEED_SLOW, 0.75f, 84, 25, 'c').withFirstHit(10)
            .withDeathEffect(std::make_shared<AreaDamageOnDeath>(
                2.0f, 84, std::make_shared<FreezeOnHit>(25, 0.70f))).withSightRange(7.0f) // 2025-10-06: slow 30%; 2026-09-08: 1228 hp, 2.5 s
            // 2026-08-04 balance update: single-use per deployment (Supercell, Final August balance changes; docs/LIVE_SYNC.md). Cooldown is moot.
            .withHeroAbility(2.0f, 170, std::make_shared<HeroIceGolemSnowstormEffect>(4.0f, 69, 0.0f, 20, 0.6f, 15), 1));

        // Hero Balloon (base: id 45; body restated from the base registration
        // above, hp 1676 per the 2026-02-02 balance post; the hero's own body
        // values are unpublished, so the base card's are used, first hit
        // included). "Coffin Cadets" (Supercell 2026-04-06: Skeletroopers launch
        // off the Balloon, crash onto the nearest target for landing damage,
        // then fight on foot; landing 230 per 2026-09-08, Skeletrooper damage
        // 204 / hit speed 1.1 s per 2026-07-06; single-use per 2026-08-04) is
        // NOT modelled: no source states the ability's elixir cost (2 on one
        // third-party page, 1 elsewhere), nor the Skeletrooper's hp, count,
        // speed, range or sight. isHero is set by hand since withHeroAbility
        // needs a cost; abilityPending keeps random decks from dealing it.
        {
            CardStats heroBalloon = troop(176, "Hero Balloon", 5.0f, Archetype::MeleeBuildingTargeter, 1676, SPEED_MEDIUM, 0.1f, 640, 20, 'a').withFirstHit(2)
                .withFlying()
                .withDeathEffect(std::make_shared<AreaDamageOnDeath>(1.5f, 240)).withSightRange(7.7f);
            heroBalloon.isHero = true;
            add(heroBalloon);
            markAbilityPending(176);
        }

        // Hero Barbarian Barrel (base: id 101). The ability lives on the
        // spawned Barbarian (heroBarbarianBarrelBarbarianStats), since the
        // rolling spell never has a turn on which activating anything would
        // mean something. isHero is set directly so validateDeckSlots and
        // seedSlotState treat id 174 as Hero-eligible; it has no runtime effect
        // on the spell. "Rowdy Reroll" (1 elixir, one use): see
        // HeroBarbarianBarrelRerollEffect.
        {
            CardStats heroBarbarianBarrelSpellStats = spell(174, "Hero Barbarian Barrel", 2.0f, 2.5f, 215, 8, '#') // 2026-10-06: 232 -> 215
                .withGroundOnly()
                .withOwnSideOnly()
                .withRollingSweep(4.5f, 2.6f, 0.5f, 0.0f)
                .withSpellSpawn(std::make_shared<PeriodicSpawnEffect>(heroBarbarianBarrelBarbarianStats()));
            heroBarbarianBarrelSpellStats.isHero = true;
            add(heroBarbarianBarrelSpellStats);
        }

        // Hero Dark Prince (base: id 46; body restated from the base
        // registration: hp 1200, 266 damage, 1.4 s hit speed, 240 shield, 2x
        // charge damage and speed, ignores the river; the hero's own body
        // values are unpublished, so the base card's are used, with no splash
        // because the base card models none). Hero-specific, official (Supercell
        // June 2026 balance post): deploy 2 s -> 1 s, which is the engine
        // default DEPLOY_TIME_TICKS, and "Tiles Needed for First Charge: 0 ->
        // 2.5 tiles". The engine has one charge threshold, so re-charges after
        // a hit also use 2.5 (the base card's 3.0 is not stated for the Hero).
        // Cost 4: single source, DeckShop (https://deckshop.pro/card/detail/hero-dark-prince); not contradicted.
        // "Destructive Dismount" (Rhino mount) is NOT modelled: Rhino hp,
        // trample damage, speed, landing damage/radius and mount-up timing have
        // no usable source. isHero is set by hand since withHeroAbility needs
        // an effect; abilityPending keeps random decks from dealing it.
        {
            CardStats heroDarkPrince = troop(177, "Hero Dark Prince", 4.0f, Archetype::MeleeSquad, 1200, SPEED_MEDIUM, 1.2f, 266, 14, 'N')
                .withShield(240) // sourced (base Dark Prince; fandom says 256)
                .withCharge(2.5f, 2.0f) // June 2026 official: first charge after 2.5 tiles; +100% on a charging hit
                .withChargeSpeed(2.0f)
                .withIgnoresRiver();
            heroDarkPrince.isHero = true;
            add(heroDarkPrince);
            markAbilityPending(177);
        }

        // Hero Bowler (base: id 22; body restated from the base registration:
        // hp 2081, 289 damage, 25 ticks, range 4.0, first hit 0.5 s, splash 1.8,
        // piercing line 11.0 = 4.0 + 7.0 after the 2026-08-04 7.5 -> 7 tile
        // projectile range cut; the hero's own body values are unpublished and
        // whether the Hero inherits the range cut is unconfirmed). Cost 5:
        // single source, DeckShop (https://deckshop.pro/card/detail/hero-bowler); not contradicted.
        // "Stone Swish" (plants his feet for 7.3 s, official June 2026 post; single
        // use) is NOT modelled: the longer reach, wind-up and per-shot damage have
        // no source, and a bare root with no extra shots would be strictly worse
        // than not activating. isHero is set by hand since withHeroAbility needs
        // an effect; abilityPending keeps random decks from dealing it.
        {
            CardStats heroBowler = troop(178, "Hero Bowler", 5.0f, Archetype::RangedSquad, 2081, SPEED_SLOW, 4.0f, 289, 25, 'w')
                .withFirstHit(5)
                .withSplash(1.8f).withLineSplash(11.0f).withSightRange(4.0f);
            heroBowler.isHero = true;
            add(heroBowler);
            markAbilityPending(178);
        }

        // Hero Tombstone (base: id 96; body restated from the base registration:
        // hp 529, 2 Skeletons per 35-tick pulse, 4 Skeletons on death, 30 s
        // lifetime; the hero's own body values are unpublished, and the base's
        // pulse interval, death count and +-1 hp are themselves unresolved
        // across sources, so they are inherited, not chosen). Cost 3: single
        // source, DeckShop (https://www.deckshop.pro/card/detail/hero-tombstone);
        // not contradicted. "Regal Revival" (5 elixir, single use, destroys the
        // Tombstone and raises Tomb Queen: Supercell 2026-06-01, 2026-07-06,
        // 2026-08-04) is NOT modelled: Tomb Queen's hit speed, movement speed,
        // range and size have no source (hp 4224, damage 422, sight 7 are
        // official but not enough to build her). isHero is set by hand since
        // withHeroAbility needs an effect; abilityPending keeps random decks
        // from dealing it.
        {
            CardStats heroTombstone = building(179, "Hero Tombstone", 3.0f, 529, '5', 0.0f, 0, 100)
                .withPeriodicEffect(35, std::make_shared<PeriodicSpawnEffect>(skeletonBarrelSkeletonStats()))
                .withDeathEffect(std::make_shared<SpawnOnDeath>(witchSkeletonStats()));
            heroTombstone.isHero = true;
            add(heroTombstone);
            markAbilityPending(179);
        }

        // Hero Valkyrie (base: id 10, live level 11; no live-sync row changes base
        // Valkyrie, so the body restates it, first hit included). "Wild Whirlwind"
        // (3 elixir, ONE use per deployment: official 2026-08-04 post): dashes to
        // the nearest enemy non-tower ground unit within 5.5 tiles, then spins for
        // 3.5 s, hitting everything in 2.5 tiles every 0.25 s for 97 (48 to Crown
        // Towers), taking 15% less damage meanwhile (HeroValkyrieWhirlwindEffect).
        // Ability numbers (cost 3, 3.5 s, 2.5 tiles, 0.25 s, 97, -15%, -50% CT,
        // 5.5 dash): single source, https://timesaver.gg/blog/clash-royale-hero-valkyrie
        // (wiki-derived); not contradicted. Range, speed, sight and deploy time of
        // the Hero are unpublished and inherit the base card; sight is the 5.5 default.
        add(troop(180, "Hero Valkyrie", 4.0f, Archetype::MeleeSquad, 1907, SPEED_MEDIUM, 1.2f, 266, 15, 'V').withFirstHit(1)
            .withSplash(1.5f)
            .withHeroAbility(3.0f, 0, std::make_shared<HeroValkyrieWhirlwindEffect>(
                /*dashRange=*/5.5f, /*spinTicks=*/35, /*radius=*/2.5f,
                /*hitIntervalHalfTicks=*/5, /*damage=*/97,
                /*selfDamageTakenMultiplier=*/0.85f, /*towerDamageMultiplier=*/0.5f), 1));

        // Hero Berserker (base: id 51; no live-sync row changes base Berserker, so
        // the body restates it: 896 hp / 102 dmg / 0.6 s / Fast / Melee: Short).
        // "Savage Survival" (ONE use per deployment: official 2026-08-04 post): for
        // 3.5 s she swings every 0.2 s, her hp cannot fall below 1, and she deals
        // 75% less damage to Crown Towers (HeroBerserkerSavageSurvivalEffect).
        // Duration 35 ticks: official, supercell.com/en/games/clashroyale/blog/news/final-august-balance-changes-826/
        //   (Balance Update - August 26th: Ability Duration 4sec -> 3.5sec).
        // Cost 3, hit speed 0.2 s (2 ticks), hp floor 1, x0.25 to Crown Towers:
        //   single source, https://trophycoach.com/heroes/hero-berserker (and its
        //   guide); not contradicted ("increasing her hit speed and becoming
        //   unbeatable" is the official wording).
        // Not modelled: the "Ultra Fast" movement during the ability (no numeric
        //   speed published), the 2026-09-07 elixir refund when she is defeated
        //   before activating (no such mechanic engine-wide, amount unstated), and
        //   any "+64% damage" (one unverified claim). Sight is the 5.5 default.
        add(troop(181, "Hero Berserker", 2.0f, Archetype::MeleeSquad, 896, SPEED_FAST, 0.8f, 102, 6, 'v')
            .withHeroAbility(3.0f, 0, std::make_shared<HeroBerserkerSavageSurvivalEffect>(
                /*durationTicks=*/35, /*hitCooldownTicks=*/2, /*towerDamageMultiplier=*/0.25f), 1));

        // Hero Ice Wizard (base: id 34), released 2026-09-07
        // (https://supercell.com/en/games/clashroyale/blog/news/new-season-k-h-a-o-s/).
        // Restates the base card's live body (hp 689, damage 90, 1.7 s, 5.5 range,
        // Medium, first hit 0.5 s, splash 1.5, slow 0.70 for 3 s). Hero-specific
        // stats are unpublished; Supercell says base changes also affect Hero
        // forms and nothing contradicts, so base figures are used (single source
        // for the base figures: trophycoach.com/heroes/hero-ice-wizard). The 3 s
        // slow duration is the engine's existing base value, not in any official
        // post. "Frosty Fella": 2 elixir (trophycoach, timesaver.gg, ldplayer.net;
        // not official, not contradicted), single use (official 2026-08-04),
        // 46 damage (official 2026-10-06, 89 -> 46), 5 s freeze (official
        // 2026-09-16, 7 s -> 5 s), radius ~2.5 tiles (ldplayer.net and
        // timesaver.gg, community-measured; unofficial, not contradicted).
        // Not modelled: the Snowman (hp contradicted: ~425 timesaver vs 266
        // ldplayer), its "behind the troop" offset, the freeze ending when it
        // dies, a separate Crown Tower damage figure (unpublished, so 100%), and
        // the 2026-10-27 elixir refund when defeated before the projectile lands.
        add(troop(184, "Hero Ice Wizard", 3.0f, Archetype::RangedSquad, 689, SPEED_MEDIUM, 5.5f, 90, 17, 'i').withFirstHit(5)
            .withTargetsAir().withSplash(1.5f).withSightRange(5.5f)
            .withOnHit(std::make_shared<FreezeOnHit>(30, 0.70f))
            .withHeroAbility(2.0f, 0, std::make_shared<HeroIceWizardFrostyFellaEffect>(2.5f, 46, 50), 1));
        // Partly built (no Snowman), so random decks never deal it.
        markAbilityPending(184);

        // Hero Electro Wizard (base: id 35), released 2026-10-05
        // (https://supercell.com/en/games/clashroyale/blog/release-notes/new-season-shocktober/).
        // The body restates the base card's live values: hp 714, damage 116
        // (official 2025-03-04), 1.8 s, range 5.0, Fast, first hit 0.6 s, split
        // to 2 targets, 0.5 s stun per hit, deploy zap radius 3.0 / 118. Hero
        // stats are unpublished, so the base is used (Supercell: base changes
        // also affect Hero forms). Cost 4: DeckShop + TrophyCoach agree.
        // "Surging Strikes" (official: stuns two nearby enemies at once for "a
        // few seconds"; third parties say 2 elixir, one use) is NOT modelled:
        // the stun duration has no unconflicted source (a ~3 s claim conflicts
        // with "while he is alive"), and the engine has no timed stun-on-hit
        // window. Not invented; abilityPending keeps random decks from dealing
        // it, and isHero is set by hand since withHeroAbility needs an effect.
        // The official 2026-10-19 "removed attack delay after ability" needs
        // nothing here: the engine adds no such delay to any ability.
        {
            CardStats heroElectroWizard = troop(185, "Hero Electro Wizard", 4.0f, Archetype::MeleeSquad, 714, SPEED_FAST, 5.0f, 116, 18, 'z').withFirstHit(6)
                .withTargetsAir()
                .withSplitTargets(2)
                .withSightRange(5.5f)
                .withOnHit(std::make_shared<FreezeOnHit>(5, 0.0f))
                .withSpawnEffect(3.0f, 118, std::make_shared<FreezeOnHit>(5, 0.0f));
            heroElectroWizard.isHero = true;
            add(heroElectroWizard);
            markAbilityPending(185);
        }

    }

public:
    static CardRegistry& getInstance() {
        static CardRegistry instance;
        return instance;
    }

    // Evolution id for a base card id, or -1. Never keyed to specific ids: it
    // is filled by addEvolution from baseStats.id.
    int evolutionIdForBase(int baseId) const {
        auto it = evolutionIdByBase.find(baseId);
        return it == evolutionIdByBase.end() ? -1 : it->second;
    }

    const CardDefinition* getCard(int id) const {
        auto it = cards.find(id);
        if (it != cards.end()) {
            return &(it->second);
        }
        return nullptr;
    }

    const std::unordered_map<int, CardDefinition>& getAllCards() const {
        return cards;
    }
};

// How many Champions a deck holds. The real game allows at most one; nothing
// here enforces that. For callers that want to check a deck; unused by the
// engine.
inline int countChampions(const std::vector<int>& deck) {
    int count = 0;
    for (int cardId : deck) {
        const CardDefinition* def = CardRegistry::getInstance().getCard(cardId);
        if (def && def->isChampion) count++;
    }
    return count;
}

// As countChampions, for Heroes.
inline int countHeroes(const std::vector<int>& deck) {
    int count = 0;
    for (int cardId : deck) {
        const CardDefinition* def = CardRegistry::getInstance().getCard(cardId);
        if (def && def->isHero) count++;
    }
    return count;
}

// Deck slot legality: slot 0 (Evolution) takes an Evolution or a plain card;
// slot 1 (Heroic) a Champion, a Hero or a plain card; slot 2 (Wild Card) any of
// these; slots 3-7 plain only. Champions and Heroes share slots 1 and 2, as in
// the real game. A real gate: GameManager::reset and setOpponentDeck throw on a
// non-empty result. Returns "" for a legal deck, else the reason.
inline std::string validateDeckSlots(const std::vector<int>& deck) {
    if (deck.size() != 8) {
        return "deck must have exactly 8 cards (got " + std::to_string(deck.size()) + ")";
    }
    for (int i = 0; i < 8; ++i) {
        const CardDefinition* def = CardRegistry::getInstance().getCard(deck[i]);
        if (!def) {
            return "slot " + std::to_string(i) + ": card id " + std::to_string(deck[i]) + " is not a registered card";
        }
        bool specialUnitAllowed = (i == 1 || i == 2); // Heroic / Wild Card: Champion or Hero
        bool evolutionAllowed = (i == 0 || i == 2);   // Evolution slot / Wild Card
        if ((def->isChampion || def->isHero) && !specialUnitAllowed) {
            return "slot " + std::to_string(i) + ": " + def->name +
                " is a Champion/Hero, only allowed in slot 1 (Heroic) or slot 2 (Wild Card)";
        }
        if (def->isEvolution && !evolutionAllowed) {
            return "slot " + std::to_string(i) + ": " + def->name +
                " is an Evolution, only allowed in slot 0 or slot 2 (Wild Card)";
        }
    }
    return "";
}
