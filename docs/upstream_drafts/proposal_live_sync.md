# Draft issue (not posted): proposal to sync the engine with the live game (2026-11)

**Title:** Proposal: sync card data with the live game (Nov 2026), in reviewable pieces

Hi! I brought the engine's cards up to date with the live game as of 2026-11-18, on a fork
(branch `update/live-sync`, about 60 small commits). I'd like to ask whether you'd want it
upstream, and in what shape. Everything comes from public sources (Supercell's posts first,
then RoyaleAPI, DeckShop and the wiki). Nothing comes from the game client.

**What it changes**

- **Balance values.** Every official balance post from 2024-08 to 2026-10-19: 31 dated
  commits (one per patch), plus one commit of long-standing errors. The engine matched the
  2026-07-06 update for most base cards; later changes and some 2024–25 ones were missing.
  The full ledger, with sources and per-row status, is in `docs/LIVE_SYNC.md` and
  `docs/live_sync/ledger.json`.
- **Spell Crown Tower damage** (UPSTREAM item 29). Fireball 159, Rocket 342, The Log 35, and
  so on.
- **Hero and Champion abilities are single-use** (live since 2026-08-04), except Boss Bandit's.
- **First-hit time**, a new mechanic. The wind-up before a unit's first attack on a target.
  Values come from the official posts, plus a separate commit deriving them from RoyaleAPI's
  `hit_speed - load_time` where that is safe.
- **13 new entries.** Minion Giant, 8 Heroes and 4 Evolutions. Five of the Heroes are
  body-only, because no public source gives their ability numbers. They are flagged so random
  decks never deal them.
- **Building deploy-burst fix** (separate bug report).

**The card-id raise (breaking).** 13 new ids were needed and 9 were free, so `CARD_ID_COUNT`
goes from 185 to 256. The observation grows from 13977 to 14403 floats, and every checkpoint
needs migrating. `python_ai/tools/migrate_checkpoint_card_ids.py` pads a checkpoint forward:
new input columns are zero, new output rows copy a never-registered id, and the Adam moments
are padded. On a test checkpoint the outputs for old ids are bit-identical (heads to float32
round-off). Evolution ids are no longer contiguous, so `get_card_info` gains `is_evolution`.

**Single-source values.** Where no official post gives a number, I used a single non-official
source only when nothing contradicted it, and commented it in the code with its source. They
are all listed in `docs/DECISIONS.md` (2026-11-18 section). Unresolved numbers were never
guessed: they are listed in `docs/live_sync/values_to_look_up.md`.

**A possible split into PRs** (each one green on its own):

1. The building deploy-burst fix (2 commits, independent).
2. Test scaffolding and the ledger docs (no behaviour change).
3. Value patches, by period: 2024–2025, 2026-01 to 07, 2026-08 to 10.
4. Spell Crown Tower damage.
5. Single-use abilities.
6. The first-hit mechanism with official values; the RoyaleAPI-derived values as an optional
   follow-up.
7. `CARD_ID_COUNT` 185 → 256, with the migration tool (breaking).
8. The 13 new entries (one commit each), after 7.
9. The python_ai test pins and perception `card_map` refresh that follow the engine.

Every gameplay change makes existing win rates and the critic's history incomparable, so you
may prefer to land it at a natural lineage break. Happy to adjust anything, or to keep it on the
fork if it doesn't fit your plans.
