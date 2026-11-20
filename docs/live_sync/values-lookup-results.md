# Values to look up: results

Researched from public pages only (Supercell posts, Liquipedia, the Fandom wiki, RoyaleAPI posts as quoted by search engines, and guide sites). Units follow the ledger: level-11 stats, seconds with engine ticks in brackets (10 ticks = 1 s).

How to use this:
- **Section A**: two or more agreeing sources, or an official post. Apply.
- **Section B**: one source, nothing contradicting it. Apply only as single-source, per DECISIONS.md.
- **Section C**: still unresolved. Update the ledger notes with the new information; don't apply.

Freshness check on Liquipedia: for each first-hit value, I compared the page's *hit speed* with the engine's current hit speed. They matched (so the page is up to date) except where flagged as stale.

## A. Resolved (2+ sources or official)

| Card (ids) | Stat | Value | Sources |
|---|---|---|---|
| Goblin Drill (98, 153) | Building lifetime | 10 s (100 ticks). Also: first spawn 1.0 s, spawn speed 3.0 s | https://liquipedia.net/clashroyale/Goblin_Drill ; https://clashroyale.fandom.com/wiki/Goblin_Drill |
| Goblin Cage (97, 139) | Building lifetime | 20 s (200 ticks) | https://liquipedia.net/clashroyale/Goblin_Cage ; https://clashroyale.fandom.com/wiki/Goblin_Cage |
| Goblin Cage (97, 139) | Goblin Brawler movement speed | Fast (90). Liquipedia's own history mentions a 2019 change to 120, but its current attribute table says Fast (90), matching Fandom | same two URLs |
| Skeleton Barrel (89) | Skeletons spawned on drop | 7 (engine has 2) | https://royaleapi.com/blog/evo-skeleton-barrel-2025-july ("Skeletons per Barrel: 7 (same)") ; https://devtrackers.gg/clash-royale/p/984af3e5-daily-card-discussion-4-9-4-10-skeleton-barrel-a-skele-party-in-the-sky (7 since April 2018) |
| Monk (119) | Third-hit (combo) damage | 422 | https://liquipedia.net/clashroyale/Monk ; https://timesaver.gg/blog/clash-royale-monk |
| Mother Witch (68) | Hitpoints after the −1 HP fix | 529 (engine already 529, so close the row) | https://liquipedia.net/clashroyale/Mother_Witch (shows 529 after a 2025-03-31 "−0.18%" HP entry) |
| Ronin (54) | Damage | 371 (engine already 371, so close the row) | https://liquipedia.net/clashroyale/Ronin ; Noff (already in ledger) |
| Cannon (25, 129) | First hit | 1.0 s (10 ticks), the same as its hit speed, so no load | https://liquipedia.net/clashroyale/Cannon ; https://clashroyale.fandom.com/wiki/Cannon |
| Giant Skeleton (39) | First hit | 0.3 s (3 ticks) | https://liquipedia.net/clashroyale/Giant_Skeleton ; https://clashroyale.fandom.com/wiki/Giant_Skeleton |
| Skeletons (24 + spawned skeletons) | First hit | 0.5 s (5 ticks) | Official 2023 post set Skeletons *Evolution* 0.5 → 0.4 s, implying base 0.5: https://supercell.com/en/games/clashroyale/blog/release-notes/balance-changes-nov-2023 ; Skeleton Army skeletons 0.5 s: https://liquipedia.net/clashroyale/Skeleton_Army ; Tombstone skeletons 0.5 s: https://theriagames.com/guide/clash-royale-tombstone-guide/ |
| Skeletons Evolution (125) | First hit | 0.4 s (4 ticks), official | https://supercell.com/en/games/clashroyale/blog/release-notes/balance-changes-nov-2023 |
| Bats Evolution (126) | First hit | 0.2 s (2 ticks), official | same Nov 2023 post. Both Evolution values predate the ledger's August 2024 start, so check nothing later changed them |
| Hero Ice Wizard (184) | Ability cost | 2 elixir | https://trophycoach.com/heroes/hero-ice-wizard ; https://riggedroyale.com/blog/clash-royale-new-cards-september-2026 ; https://note.com/kabutom/n/n3be95568c055 |
| Hero Ice Wizard (184) | Freeze rules | Freeze lasts while the Snowman is alive, capped at 5 s (official 7 → 5 s on Sept 16). It only triggers on targets already slowed by Hero Ice Wizard's attack | https://supercell.com/en/games/clashroyale/blog/release-notes/september-balance-changes-2027/ ; https://royaleapi.com/blog/hero-ice-wizard-september-2026 |
| Hero Ice Wizard (184) | Snowman spawn damage radius and freeze radius | 2.5 tiles | RoyaleAPI post above ; https://gamelevate.com/hero-ice-wizard-clash-royale-ability-stats-and-how-it-works/ ; https://www.ldplayer.net/blog/clash-royale-hero-ice-wizard-guide.html |
| Hero Ice Wizard (184) | Snowman spawn damage | 46 (89 → 46 on Oct 6) | https://alviran.net/blog/clash-royale-heroes-list-release-dates-2026/ ; https://www.sportsdunia.com/gaming/clash-royale-balance-changes-october-2026 ; launch value 89: note.com post above |
| Hero Electro Wizard (185) | Surging Strikes | Costs 2 elixir. Two beams lock onto the 2 closest enemies, applying a 0.5 s stun every 0.5 s for 3 s. Since Oct 6 there is no attack delay after the ability (official) | https://royaleapi.com/blog/hero-electro-wizard-october-2026 (dev-build stats) ; https://note.com/kabutom/n/nf1741c6d472e ; https://trophycoach.com/guides/clash-royale-shocktober-season-88-guide ; https://supercell.com/en/games/clashroyale/blog/release-notes/october-balance-changes-2026/ |
| Electro Giant Evolution (186) | De-level pulse | Every 6 s, radius 6 tiles, −1 level per pulse, min level 1, never Crown Towers, no damage (≈9% HP and damage per level) | https://royaleapi.com/blog/electro-giant-evolution-october-2026 ; https://gamingonphone.com/news/clash-royale-season-88-shocktober-october-2026-update-and-balance-changes/ ; trophycoach Shocktober guide above |
| Minion Giant (183) | Hitpoints | 1,817 | https://after-strategy.com/en/clash-royale-season-87-minion-academy-minion-giant-2026/ (citing RoyaleAPI) ; ldplayer (already in ledger) |
| Minion Giant (183) | Knockback immunity | Yes | https://trophycoach.com/guides/how-to-counter-minion-giant-deck (citing RoyaleAPI's TV Royale coverage) ; after-strategy post above (citing Rigged Royale) |
| Hero Dark Prince (177) | Ability cost | 3 elixir | https://trophycoach.com/heroes/hero-dark-prince ; https://lootbar.gg/blog/en/clash-royale-season-83-hero-bowler-and-dark-prince.html ; https://www.ldshop.gg/blog/clash-royale-gp/season-83-hero-bowler-and-dark-prince.html |
| Hero Dark Prince (177) | Shield HP | 240 | ldshop post above ; Liquipedia (already in ledger) |
| Hero Dark Prince (177) | 2.5-tile first charge | Applies to the Rhino: it needs 2.5 tiles of running before its first charge, then charges the nearest building. TrophyCoach also lists a June 2026 deploy change of 2 s → 1 s (it doesn't say whether that's the Rhino or the ability) and the Rhino's Crown Tower spawn damage going from 307 to 0 | https://timesaver.gg/blog/clash-royale-hero-dark-prince ; trophycoach hero page above ; official June post (no guaranteed tower damage): https://supercell.com/en/games/clashroyale/blog/release-notes/june-balance-changes-2026 |
| Hero Bowler (178) | Stone Swish | Costs 2 elixir, range 11.5 tiles, 3 shots over 7.3 s (official June 9 → 7.3 s) | https://trophycoach.com/heroes/hero-bowler ; https://trophycoach.com/guides/hero-bowler-guide ; lootbar post above |
| Hero Balloon (176) | Ability cost | 2 elixir (RoyaleAPI's own launch tweet; trophycoach's "1" is the outlier) | https://mobilematters.gg/news/clash-royale/april-2026-update-new-hero-card-evolution-balloon-festival (quotes the RoyaleAPI tweet) ; https://www.sportsdunia.com/gaming/clash-royale-best-hero-balloon-decks ; https://www.u7buy.com/blog/clash-royale-balloon-festival-guide/ |
| Lightning (31) | Stun on each of its three struck targets (applied 2026-11-20) | 0.5 s (5 ticks); a struck Crown Tower is stunned too (no source either way) | https://theriagames.com/?p=78055 ("disables the struck targets for 0.5 seconds") ; https://noff.gg/clash-royale/card/lightning and Fandom's Epic cards page (per the user, 2026-11-20; both refuse automated fetches) ; cr-api-data description "damage and stun" |

## B. Single source (apply flagged, or skip)

| Card (ids) | Stat | Value | Source | Notes |
|---|---|---|---|---|
| Minions (41) | First hit | 0.5 s (5) | https://liquipedia.net/clashroyale/Minions | Hit speed 1.2 matches the engine |
| Minion Horde (42, 149) | First hit | 0.5 s (5), inferred from Minions | as above | The Horde page couldn't be read |
| Bats (78, NW bats −12/−14) | First hit | 0.6 s (6) | https://liquipedia.net/clashroyale/Bats | Hit speed 1.2 matches the engine |
| Skeleton Army (12, 133) | First hit | 0.5 s (5) | https://liquipedia.net/clashroyale/Skeleton_Army | Hit speed 1.1 matches the engine |
| Royal Giant (18, 143) | First hit | 0.9 s (9) | https://liquipedia.net/clashroyale/Royal_Giant | Hit speed 1.8 matches the engine |
| Miner (52) | First hit | 0.5 s (5) | https://liquipedia.net/clashroyale/Miner | Hit speed 1.3 matches the engine |
| Dart Goblin (20, 131) | First hit | 0.35 s | https://liquipedia.net/clashroyale/Dart_Goblin | Between ticks 3 and 4: decide the rounding rule. Hit speed 0.8 matches |
| Dark Prince (46, 177) | First hit | 0.4 s (4) | https://liquipedia.net/clashroyale/Dark_Prince | Hit speed 1.4 matches the May 2026 change |
| Dark Prince (46, 177) | Splash | Yes, area damage with radius 1.1 tiles | same | Conflicts with the "1.5 tiles for all troops" reading of the 2025-01-07 rule. Check that rule's exact wording before applying |
| Cannon Cart (69) | First hit | 0.5 s (5) | https://liquipedia.net/clashroyale/Cannon_Cart | Hit speed 0.9 matches |
| Executioner (36, 157) | First hit | 0.5 s (5) | https://liquipedia.net/clashroyale/Executioner | Liquipedia lists hit speed 0.9 s (the same as cr-api's 900 ms hit time), not the 2.4 s card value. Only apply if the engine models his hit time the same way |
| Goblin Cage Brawler | First hit | 0.2 s (2) | https://liquipedia.net/clashroyale/Goblin_Cage | Bonus value, not in your list |
| Barbarians (8, 155, spawned) | First hit | 0.4 s (4) | https://liquipedia.net/clashroyale/Barbarians | **Page looks stale**: HP 690 vs the official Aug 2026 716, and its hit speed (1.4) disagrees with the Battle Ram page (1.3). Low confidence |
| X-Bow (92) | First hit | 0.3 s (3) | https://liquipedia.net/clashroyale/X-Bow | **Page is stale** (still shows hit speed 0.3, pre-Aug 2026). Low confidence |
| Hero Bowler (178) | Stone Swish details | AoE radius 1.5 tiles; wind-up 2.5 s; +30% hit speed while planted | radius: trophycoach guide ; wind-up: https://twinfinite.net/guides/best-bowler-deck-in-clash-royale/ ; +30%: lootbar post | One source each |
| Hero Balloon (176) | Skeletrooper targeting | Goes to the nearest ground enemy within 6.5 tiles, otherwise lands under the Balloon | u7buy post above | |
| Hero Dark Prince (177) | Rhino stats | About Ram Rider's stats minus 20% HP and damage | https://trophycoach.com/guides/hero-dark-prince-guide | Approximate. The two Rhino figures in the ledger still conflict |
| Champions | Cooldown when a new Champion is placed | Resets instantly (ability ready) | https://clashroyale.fandom.com/wiki/Cards | Moot for single-use abilities |
| Monk (119) | Third-hit knockback | 1.8 tiles | timesaver (already in ledger) | Liquipedia doesn't state it |

## C. Still unresolved (new information only)

| Card (ids) | Stat | What I found |
|---|---|---|
| Hero Ice Wizard (184) | Snowman HP | Still conflicting: 266 (note.com, ldplayer) vs 425 (riggedroyale; timesaver calls 425 "pre-release, provisional"). Snowman lifetime is probably the 5 s freeze cap, but that's my inference |
| Hero Electro Wizard (185) | Beam damage and range | Not published anywhere I could read |
| Electro Giant Evolution (186) | First pulse time; cycles | note.com hints at a first pulse around 5.5 s. Fandom's Evolution table says 1 cycle, but it names a different ability ("Lightning Rod"), so it may be stale |
| Minion Giant (183) | Deploy time, projectile speed, splash, range, speed | Not found in any readable source |
| Hero Tombstone (179) | Tomb Queen hit speed, range, speed | Nothing beyond the official HP 4224 / damage 422 / sight 7 |
| Inferno Tower / Dragon, the Spirits, Lava Hound | First hit | Liquipedia has no first-hit field for these (ramp or kamikaze units), and the Lava Hound page is stale |
| Rascals, Skeleton Barrel, Minion Horde, Skeletons, Bomb Tower, Ram Rider (Liquipedia pages) | various | These pages refused automated reads, so they can't be checked from here |
| Cannon (25) | Level-11 damage (not in your list) | Sources disagree: Liquipedia 202, Fandom 212. Worth checking against the engine and the ledger |

Not attempted: the remaining first-hit rows (Battle Healer, Berserker, Boss Bandit, Flying Machine, Furnace, Goblin Machine, Goblinstein, Little Prince, Magic Archer, Mighty Miner, Rune Giant, Spirit Empress, Suspicious Bush, Runner, Souldier, etc.), Tower Troops, and the prose-only rows (Vines, Graveyard, Three Musketeers, Rune Giant enchant).
