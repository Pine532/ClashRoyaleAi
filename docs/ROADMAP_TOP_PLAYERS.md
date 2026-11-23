# Roadmap to top-player level: the read-only audit (2026-11-22)

Five questions from the user, to plan how to get the bot to top-player level:
1. hidden information;
2. network size;
3. exploiters in the league;
4. lookahead speed;
5. an opponent tracker, as a design only.

How the answers were produced:
- Nothing was changed. The phase-2 run (`train_selfplay.py`) kept training throughout, at about phase-2 episode 38,000.
- Every number below was measured with a scratch script, or read from a cited `file:line`.
- Speeds were measured on one core, at `nice -n 19`, while training shared the CPU, so they are pessimistic. The machine is an Apple M5 Max with 18 cores; the load average was 8-13.
- Each investigation was checked by an independent reviewer. The parameter count, the training device and the exploiter's deck refusal were checked again by hand.
- One error was caught and fixed: the channel offsets in section 1 were off by one channel from channel 8 on.

---

## Summary

| Question | Answer |
|---|---|
| 1. Hidden information | No engine-only opponent information reaches the policy. The observation has no opponent hand, no next card and no current elixir. Probes that rewrote them changed 0 of 14,403 floats. Its opponent sections come only from plays a viewer sees: the plays, their recency, and the cumulative card spend. The real gaps run the other way. (a) A player sees things the policy does not: its own next card, shields, freezes and slows, ability spend. (b) Live play does not yet feed the opponent sections at all: they are zeros live and filled in training. |
| 2. Network size | 1,923,155 parameters. The LSTM holds 1,804,288 (93.8%). The encoders hold 24,929 (1.3%) and the heads 93,938 (4.9%), most of that in the auxiliary next-card head. A forward pass costs 0.50 ms at batch 1. Time goes to the convolutional encoder and the placement head, not the LSTM. |
| 3. Exploiters | None. Phase 2 samples about 456 past copies (97.2% of the sampling weight), 4 scripted bots and 2 C++ heuristic anchors. Exploiter code exists, but it has been off since 2026-08-11 and refuses this deck (no Hero-ability support). The past copies and the anchors play the Minion Giant mirror. |
| 4. Lookahead | Copying a game state takes 2-4 µs. A 5-second rollout of the engine alone takes 0.03-0.2 ms, so about 6,000-35,000 per second per core, depending on how crowded the board is. With the policy choosing both sides' moves, it is about 100 per second at batch 1, or about 180-195 batched by 16. The network is the whole cost. Search code exists, but it assumes the opponent's true hand. |
| 5. Opponent tracking | The engine's cycle is a strict first-in-first-out queue. An event-fed tracker reproduced the opponent's hand exactly (226/226) and its elixir exactly (max error 0.0000 over 1,105 decisions) in the simulator. Proposed: one pure-Python tracker fed by play events, shared by the simulator and vision. It feeds a new zero-initialised branch that is exact at any input, on any backend. A migration keeps the resume clean, and an A/B runs after Monday's decision. |

---

## 1. Hidden information

### 1.1 The observation, section by section

`observation_size()` is 14,403. It is built by `ClashEnv::extractObservationForTeam` (`include/core/ClashEnv.h:156-324`), contiguous, with no gaps. Spatial channel c covers floats 612c to 612c + 611.

| Floats | Contents | A real player |
|---|---|---|
| 0-2447 (channels 0-3) | Own units by class: melee, ranged, building-targeter, buildings and towers. Each cell holds HP / 4256 (troops) or HP / 4008 (buildings). | Sees the units. Troop HP shows only as a bar, and only once damaged, so the exact value needs the card's maximum: card knowledge. |
| 2448-4895 (4-7) | Enemy units, the same four classes | The same |
| 4896-5507 (8) | River row: +1 on bridge columns, -1 elsewhere (static) | Visible |
| 5508-6731 (9-10) | Own / enemy units per cell, +0.2 each, capped at 1 | Visible |
| 6732-7955 (11-12) | Own / enemy flyer present | Visible |
| 7956-9179 (13-14) | Own / enemy can hit air | Card knowledge |
| 9180-10403 (15-16) | Own / enemy damage per tick, the maximum in the cell, / 60. Includes ramp, charge and buffs (`CombatEntity.h:1201-1215`). | Card knowledge. The ramp stage is not on screen. |
| 10404-11627 (17-18) | Own / enemy attack range / 12 | Card knowledge |
| 11628-12851 (19-20) | Own / enemy BASE move speed / 1.5. Slows and freezes are not applied (`Troop.h:13`). | Card knowledge |
| 12852 | Own elixir / 10 | Visible |
| 12853-12856 | Own hand costs / 10 | Visible |
| 12857-13880 | Own hand, 4 x 256 one-hot. There is no next card. | Visible |
| 13881-13890 | Extra scalars, in order: time / 3600; own card spend / 280; OPPONENT card spend / 280; own King, left and right tower HP / 4008; enemy King, left and right tower HP / 4008; elixir phase / 3 | Time, tower HP and phase are visible. The spends are derivable by counting the card costs of visible plays. |
| 13891-14146 | Opponent `seen[c]`: 1.0 once the opponent has played card c | Derivable: every play is on screen |
| 14147-14402 | Opponent `recency[c]` = exp(-ticks since its last play / 200) | Derivable |

The two opponent blocks read the table that `GameManager::playCard` writes (`GameManager.h:575-578`). `inject` does not record a play; `note_played_card` does.

### 1.2 What the policy gets that a player can't

No engine-only opponent information reaches the policy. Probes on `env.snapshot()` copies:

| Change | Floats of team 0's observation changed |
|---|---|
| Opponent elixir 0 vs 10 | 0 |
| Opponent hand swapped | 0 |
| Opponent elixir multiplier 3.0 | 0 |
| Own elixir changed (control) | 1 |
| Own hand swapped (control) | 3 |

Inputs beyond the vector:
- **Affordability mask.** From the observation's own elixir and hand costs: visible.
- **Hand-slot lock.** The 20 ticks after a card cycles in. On screen the card slides in, but the timer is not shown.
- **Ability readiness.** The button state: visible.
- **Placement mask.** The deploy zone: visible.

Nothing privileged feeds the critic: the value head reads the LSTM state and the detached cycle features only (`net.py:509,694`).

### 1.3 What a player sees that the policy doesn't

- **Its own next card.** The hand block has 4 slots. `perception/contracts.py`'s `CycleState.next_card` exists but is not wired to the encoder.
- **Shields.** `shieldHp` is not encoded.
- **Freeze, stun and slow states.** Speed is the base speed. Cooldowns, wind-ups and target locks are also absent, though a player sees those only partly.
- **Invisible units.**
  - The builder skips untargetable entities (`ClashEnv.h:182`). That drops invisible units for both sides, including the policy's own: Royal Ghost, Lumberjack Ghost, the Archer Queen cloak, Dark Guard, the Boss Bandit getaway.
  - A Royal Ghost probe changed 0 floats.
  - Whether the real game shows an owner its own invisible unit is not checked against a source.
- **Ability spend, on both sides.**
  - The spend scalars sum card plays only. Abilities go to a separate collector.
  - Probe: a Hero Ice Wizard ability moved elixir 10 → 8.035, with the spend scalar unchanged.

### 1.4 Training-only signals that read engine truth

These are acceptable in training; none is an input to acting.
- **Next-card and CycleId auxiliary heads.** Their label is the opponent's next realised play, from `get_last_played_tick` over the opponent's true deck (`engine_stats.py:93-120`). It is a public event, not the hidden hand.
- **`info["opp_elixir"]`.** Diagnostics only (`gym_wrapper.py:416`).
- **Reward shaping.** It reads engine statistics.
- **The UtilityTeacher (phase 1 opponent).** It reads the TRAINEE's true hand and elixir inside its counter rollouts (`teacher.py:1434-1437`). This makes the teacher stronger than a real opponent could be; it does not leak into the trainee's observation.
- **The opponent elixir multiplier.** The 1.35x and 1.50x heuristic anchors get a hidden handicap (`GameManager.h:59`) that no observation shows.

### 1.5 The live path today: the largest gap

`perception_encoder.encode` (used by `perception/live/mvp_loop.py`) leaves these differences from training:
- **Opponent blocks.** Both opponent blocks are zeros, and the opponent spend is 0.0 (`perception_encoder.py:65,250-255,322`). Live, the policy sees "the opponent has played nothing and spent nothing" all match.
- **Hand-lock mask.** It is not applied (`mvp_loop.py:423`).
- **Ability heads.** They are discarded (`mvp_loop.py:441`), so the Hero is never activated.
- **Opponent play detection.** The detector is a stub (`perception/detect/placements.py` raises `NotImplementedError`; "blocked on data", `perception/README.md:25`). This blocks every opponent-side input live, including the tracker in section 5.
- **Overtime.** The time scalar past tick 3600 is not checked.

---

## 2. Network size

`MicroRoyaleNet` as phase 2 builds it for this deck: one ability slot, for the Hero Ice Wizard. **1,923,155 parameters**, all trainable. This matches the live checkpoint (`minigiant_p2_selfplay.pth`) key for key. The checkpoint's 256 extra numbers are the saved `spell_flags` buffer.

### 2.1 By role

| Role | Modules (`net.py`) | Parameters | Share |
|---|---|---|---|
| LSTM | `LSTMCell(1504, 256)` (:454). The input is the CNN's pooled 32x9x5 = 1440, plus 64 scalar features. | 1,804,288 | 93.8% |
| CNN trunk, including the hi-res path | `cnn_trunk` (:393-407). The hi-res path reuses `cnn_trunk[:2]`. | 14,464 | 0.75% |
| Scalar encoder | econ 8 + hand 20 + extra 12 + cycle 24 = 64 features (:213-287) | 6,353 | 0.33% |
| Card embeddings | `card_id_embed`, `noop_embed` (:445-449) | 4,112 | 0.21% |
| Card head | `Linear(280, 5)` (:459) | 1,405 | 0.07% |
| Placement heads | `place_ctx`, `place_up`, `place_ctx_hi`, `place_hires` (:470-497) | 19,546 | 1.02% |
| Value head | `Linear(280, 1)` (:509) | 281 | 0.01% |
| Ability head | `Linear(256, 2)` (:514) | 514 | 0.03% |
| Auxiliary heads | next card `Linear(256, 256)` 65,792; cycle id 6,400 | 72,192 | 3.75% |

- The input side is tiny: about 25k parameters read the 14,403-float observation.
- Almost all of the capacity is the LSTM's input and recurrent weights.
- Any capacity change kills checkpoint comparability and needs an A/B.

### 2.2 Forward cost

Median of 300 calls, CPU, one thread, under load:

| Batch | `extract_features_hires` | `step_lstm_and_card` | `placement_given_card` | Total |
|---|---|---|---|---|
| 1 | 0.201 ms | 0.046 ms | 0.250 ms | 0.497 ms |
| 16 | 2.011 ms | 0.234 ms | 1.477 ms | 3.723 ms |

- The LSTM is 94% of the parameters but under 10% of the time, because it is one matrix multiply.
- One full decision costs about 1 ms at batch 1 when it includes fetching the observation, the masks and sampling (section 4).

### 2.3 Checkpoint sizes

| File | Size | What it holds |
|---|---|---|
| Live checkpoint | 23,196,677 B | Weights 7.69 MB (1,923,411 x 4 B); the Adam moments 15.39 MB (two per parameter); about 118 KB of metadata and pickle overhead |
| Historical snapshots | 7.7 MB | Weights only |

---

## 3. The phase-2 league

### 3.1 What it samples today

Counts at the checkpoint of phase-2 episode 38,018:
- The pool is `historical + 4 scripted + 2 builtin` (`train_selfplay.py:144-145`): 462 entries.
- The weights come from `max(floor, (1 - s)^2)`, where s is each opponent's win-rate estimate (`selfplay_env.py:223-237`).

| Kind | Count | Deck it plays | Share of sampling weight |
|---|---|---|---|
| Past copies of this lineage | 456 (441 from phase 1, 15 from phase 2) | The Minion Giant MIRROR (`selfplay_env.py:137`) | 97.2% |
| Scripted Counter and Defender (floor 0.8) | 2 | A uniform random engine deck (`selfplay_env.py:148`), not the meta pool | 0.77% each |
| Scripted Rusher and Cycler | 2 | The same | about 0.15% each |
| C++ heuristic at 1.35x and 1.50x elixir (floor 0.5) | 2 | The mirror | 0.48% each |

- 2.7% of counted games (712 of 26,499) were against the scripted and builtin opponents.
- The past copies' estimates sit at 0.456-0.621, and none is at the 0.05 floor.
- **The scripted floor did not scale.** The Counter and Defender floor was sized for about 7.7% of samples at about 98 members (`.claude/CLAUDE.full.md` about line 2250). At 462 members it gives 0.77% each.

### 3.2 Exploiters: in the code, switched off

Status:
- `python_ai/trainers/exploiter.py` exists. `EXPLOITER_ENABLED = False` (line 93).
- The live checkpoint shows no burst ever ran.
- No exploiter snapshot is in `historical_checkpoints/`.
- It would also refuse this deck: its rollout never samples Hero or Champion abilities and raises `NotImplementedError` for a deck with ability slots (`exploiter.py:129-134`; TODO 00.1(b)).

What it does when on:
- Every 20,000 main episodes (first at 10,000), it trains a copy of the main agent for 1,000 episodes on 4 envs. The copy is re-seeded every 2 bursts.
- Its only opponent is a frozen copy of the main agent, on the mirror deck, with the same reward.
- It runs synchronously in the main process, so the main agent pauses.
- Its snapshot (`*_exploiter_b<i>_ep<N>.pth`) joins the pool at once, at the 0.5 prior.

History:
- Off on 2026-07-31 after 3 flat bursts (0.536 / 0.585 / 0.577). The null was then 0.598, from a team-1 observation bug, since fixed.
- On again 2026-08-09.
- Off on 2026-08-11 after bursts scored 0.28 and 0.18.
- The stated reason was that the main agent was reward-hacking, and an exploiter cannot punish a strategy that pays through the reward (`exploiter.py:88-93`). The docs disagree on that diagnosis: `DECISIONS.md:4910` records the elixir-dump reading as measured and refuted.
- No document says the re-enable condition ("a parked building is no longer free") has been re-measured. `TODO.md:1647` lists re-enabling as not pursued.

### 3.3 What adding exploiters would take

1. **Port the ability sampling** (`rl/abilities.py`) into the exploiter's rollout (TODO 00.1(b)). Without it, this deck cannot have exploiters.
2. **Re-measure the re-enable condition** on today's reward and engine before trusting a burst's verdict.
3. **Give exploiter snapshots their own sampling floor** (`selfplay_env.py:223-228`). At the prior, one exploiter in 460 entries gets about 0.2% of the weight, so it would almost never be met. Also add a retirement rule, since an exploiter goes stale when its hole is patched, not with age.
4. **Measure the cost on today's engine.** The old figure was a 17-20% throughput tax at a 20,000-episode cycle (`.claude/CLAUDE.full.md:7539`), on a slower engine. A second process would avoid the pause but compete for cores, and the code has no support for it.
5. **Keep its state across a restart.** Today its weights are lost at a restart (`train_selfplay.py:107-109`).
6. **Judge it like any arm**: the fixed benchmark and the head-to-head.

The repo's docs describe only one kind of exploiter (trained against the current main agent). There is no separate main-exploiter or league-exploiter split.

### 3.4 The deck gap

On the ladder, every opponent plays its own deck. Phase 2 trains about 97% of games on the mirror:
- Its only non-mirror decks are the scripted bots' uniform random decks.
- The 16 meta decks and the teacher are phase 1 only.
- The benchmark plays the 16 meta decks plus the mirror, so phase 2's training mix differs from what it is measured on.

Options the code already allows:
- Draw an opponent deck per episode from `deck_pool` for snapshot opponents (`set_opponent_deck`), and keep win-rate estimates per (opponent, deck).
- Add teacher games, as in Monday's decision rule.

A snapshot trained on Minion Giant would play another deck out of its distribution. Neither option is tested.

---

## 4. Lookahead

### 4.1 Measured speeds

The states come from real policy games (the start copy on both sides): early (tick 200), mid (900), double elixir (1500), and busy (2270, the most crowded of 5 games). One core, under load, using `env.snapshot()` (never `clone()`, which implements the Clone card).

| Per 5-second rollout (snapshot + 50 ticks) | early | mid | double | busy |
|---|---|---|---|---|
| Snapshot alone | 0.002 ms | 0.003 ms | 0.003 ms | 0.004 ms |
| Both sides idle, `step_self_play(..., 50)` | 0.032 ms (31.5k/s) | 0.054 ms (18.4k/s) | 0.074 ms (13.5k/s) | 0.142 ms (7.0k/s) |
| Both sides idle, `step_self_play_fast` | 35.1k/s | 20.5k/s | 14.1k/s | 7.3k/s |
| Each side deploys a card first | 18.4k/s | 15.8k/s | 11.4k/s | 6.6k/s |
| Policy picks both sides, 5 decisions, batch 1 | 107/s | 99/s | 98/s | 96/s |
| The same, 16 rollouts batched | 195/s | 186/s | 176/s | 178/s |

- **The reviewer's own re-measure** at a different mid state: snapshot 0.0022 ms; idle rollout 0.037 ms (about 27k/s); policy batch 1 10.0 ms (about 100/s).
- **State matters.** Engine cost varies at least 30% between states, and it grows with the unit count.
- **The fast step** skips only building the observation (0.085 ms a call), saving 5-13%.
- **The network is the whole cost.** With the policy in the loop, the engine is under 1% of the time. Lookahead speed is a network-latency problem, not a snapshot problem.
- **Not measured:** spreading search over several cores.

### 4.2 Search code and past results

The code (`python_ai/search/`):
- `search.py` tries greedy plus the top 3 cards x 2 cells. It rolls each out on its own snapshot, with our side idle and the C++ heuristic as the opponent, then scores all of them in one batched critic call.
- `realtime_search.py` adds a 120 ms deadline, with greedy as the fallback.
- `eval/bench_search_latency.py` measures the tail latency. It was not run (it plays full matches).

Recorded results:
- **Best:** at 1.5x opponent elixir against the C++ heuristic, greedy 0.625 vs search 0.944, +0.319 [+0.237, +0.401] (2026-08-11).
- **Horizon:** 12 decisions was best (0.563 → 0.963).
- **Null or negative:**
  - Search against the UtilityTeacher was negative, -0.3 to -0.5, because the rollout opponent is the heuristic, not the real opponent.
  - Search on both sides erased the advantage.
  - Distilling search back into the policy gave +0.016 (not significant); a later DAgger variant gave +0.045.
- `DECISIONS.md:7567` calls the +0.319 stale against today's deck, policy and engine.
- No search A/B exists against self-play opponents with the Minion Giant deck.

### 4.3 Hidden information in search

- A snapshot copies the opponent's TRUE hand, queue, elixir and cooldowns.
- `realtime_search.py:56-57` requires the true state.
- No code samples hidden information. The past results were measured with perfect information.
- Live search needs to sample possible opponent hands, queues and elixir from the section-5 tracker's beliefs: fork one snapshot per sample, and average the critic over the samples. The engine has the setters for it (`set_hand_for_team`, which can refuse; `set_elixir_for_team`; `set_opponent_deck`).
- With the speeds above, the samples multiply the network scoring (about 8.4 ms per 16 states batched), not the engine time.

---

## 5. Opponent tracking

### 5.1 What the observation has about the opponent today

| Item | In the observation? | From |
|---|---|---|
| Hand | No (`ClashEnv.h:246,253` read the observer's own hand) | |
| Next card | No | |
| Current elixir | No (`ClashEnv.h:244` is the observer's own; the header calls the opponent's bar "hidden on a real screen", `:430-433`) | |
| `seen` / `recency` | Yes | The engine's played-card table. It equals what a viewer saw, because only real plays write it. |
| Cumulative card spend | Yes | The engine's statistics (`ClashEnv.h:267-271`). It equals the sum of visible card costs, Mirror at cost + 1 included, but leaves out ability spend, Elixir Collector income and the cap. |

So nothing is read straight from hidden engine state. The tracker adds information the policy does not have today: the hand, the next card, and the current elixir.

### 5.2 The mechanics the tracker must reproduce

Read from the engine source and checked by probes.

**The cycle** (`PlayerState.h:146-164`):
- The hand is 4 cards and the queue 4.
- A played card goes to the back of the queue. The front card fills the vacated slot and is locked for 20 ticks.
- After plays p1..pn, with n ≥ 4:
  - the queue is exactly the last 4 plays, oldest first;
  - **the next card is p(n-3)**, known even before all 8 cards are seen;
  - **the hand is the deck minus the last 4 plays.**
- Before all 8 cards are seen, the hand is the seen cards minus the last 4, plus u = 8 - (distinct cards seen) unknown cards.
- With n < 4, the played cards are out and the next card is unknown.
- Measured: 0 violations in 34 plays, a Hero included; hand exact 226 of 226 times once 8 cards were seen.

**Engine details:**
- Heroes and Champions are NOT held out of the cycle in the engine. Whether the real game holds Champions out needs a source before the live version.
- The opening hand never holds an Elixir Collector (99), Mirror (164) or Mirror Evolution (188); `reset()` swaps them into the queue.
- Mirror records id 164 and costs the last non-Mirror play's cost + 1.
- Spirit Empress (165) costs 3, or 6 when its owner holds 6 elixir or more. Its form (ground or flying) shows which.

**Elixir** (`GameManager.h:207-225,720-750`):

| Item | Engine value |
|---|---|
| Start | 5.0 |
| Regeneration | 0.035 per tick, so 2.857 s per elixir. The real game is 2.8 s (`perception/timebase.py`). |
| Phases | 1x; 2x from tick 1200; 3x from tick 1800 |
| Cap | 10 |
| Hidden handicap | `oppElixirMultiplier`: 1.0 for every opponent except the 1.35x and 1.50x heuristic anchors |

| Event | Effect on the opponent's elixir | Visible? |
|---|---|---|
| Card play | minus its cost (Mirror: cost + 1; Spirit Empress: 3 or 6) | Yes |
| Hero or Champion ability | minus 1-3 at the press. In our deck, only the Hero Ice Wizard: 2.0, single use. | The effect is visible; no vision code detects it |
| Armed ability refunded on death | plus 2.0 (Hero Ice Wizard only) | Inferable |
| Elixir Collector | plus 1 per 150 ticks, the first at play + 160. The engine's default 300-tick lifetime gives a SINGLE pulse; the real game is about 1100 ticks, a known unapplied gap (`docs/LIVE_SYNC.md:1043,1223,2059`). | The building and the pump are visible |
| OUR Elixir Golem family dying | plus 1.0 / 0.5 / 0.5 to the opponent. Our deck has none. | Visible |

**Order within a tick:** regenerate, then plays and updates, then grants.

Measured:
- Debiting each play at its own tick reproduced the opponent's true elixir exactly: max error 0.0000 over 1,105 decisions (9 matches, 3 decks, against the C++ heuristic).
- Debiting at the 10-tick decision boundary was off by up to 0.42.
- These probes had no Mirror, Spirit Empress, Collector or abilities.

### 5.3 The tracker

**One pure-Python class** (no torch, no engine import), so the training env and `perception/` run the same code:

```text
OpponentTracker(rules)                 # rules: regen rate, phase schedule, collector lifetime,
                                       # champion-cycle rule; "engine" and "real game" presets
  .on_play(tick, card_id)              # cost from the card table; Mirror and Empress rules inside
  .on_ability(tick, cost)              # optional
  .on_elixir_grant(tick, amount)       # Collector pulse, Golem-family death, refund
  .features(tick) -> fixed-width vector
```

**State:**
- the distinct cards seen (up to 8);
- the play log with ticks;
- the last non-Mirror play;
- the elixir estimate, advanced event by event (regenerate up to the event's tick, then debit or credit, capped at 10);
- a floor correction. A play of cost c proves the opponent had at least c. If the estimate is lower, raise it to c and count the correction. Corrections show a hidden multiplier or a missed event, and widen the uncertainty.

**Features** (proposed; about 520 floats, laid out like the existing per-card-id opponent blocks):
- `in_hand[256]`: 1.0 for each seen card known to be in hand;
- `next[256]`: a one-hot of p(n-3) once n ≥ 4;
- `unknown_in_hand / 4`;
- a next-known flag;
- `distinct_seen / 8`;
- `elixir_estimate / 10`;
- the corrections so far, scaled;
- the ticks since the opponent's last play, scaled.

No feature reads `get_elixir_for_team`, the hand, or the engine's spend counter.

**Simulator feed** (in `gym_wrapper` and `selfplay_env`):
- **Plays.** Each step, scan `get_last_played_tick(opponent, c)` over all 256 ids (24.6 µs) and emit each new play with its tick.
  - At most one opponent play lands per step (self-play opponents act at the step's first tick; the heuristic has a 25-tick cooldown), so no play is lost.
  - Do not reuse `selfplay_env._poll_opponent_play`. It scans only the TRAINEE's deck (`selfplay_env.py:372`), so it misses scripted opponents' other cards.
- **Abilities.** The env generates team 1's ability flags itself, so it can call `on_ability`.
- **Collector pulses.** Modelled from the play tick under the engine preset.
- **A second tracker for team 1.** A neural opponent with the new layout needs one, following the trainee's plays.

**Vision feed:**
- **Prerequisite.** The opponent placement detector, today a stub (section 1.5).
- **Reuse.** `perception/track/cycle.py`, `opp_deck.py` and `opp_elixir.py` already implement parts of this, with the real 2.8 s rate. They are wired only into `tools/divergence_report.py`. They should become adapters over the shared class, not a second copy of the rules.
- **The same feed fixes section 1.5.** The encoder should fill the existing `seen`/`recency` blocks and the opponent spend from this tracker.

**Search** (section 4.3): the tracker state lives in Python, so search must copy it alongside every `env.snapshot()` fork.

**Where the code goes.** A module under `python_ai/`, importable by `perception/`, which already loads the net from there. The env wrappers and `net.py` count as policy code, so implementing this needs the user's yes.

### 5.4 Adding it with zero-initialised weights

**Recommended: a new branch added onto existing features.**
- The branch is `Linear(~520, 32)` (default random init) → ReLU → `Linear(32, 40)` with weights and bias ZERO, about 18k new parameters.
- Its output is added to the first 40 of the 64 scalar features (econ + hand + extra). Leave the last 24 alone: `_split_cycle` reads them as the cycle skip.
- Register it last in `MicroRoyaleNet.__init__`.

Measured:
- **Exact.** Card, placement, ability, value, auxiliary and LSTM outputs were `torch.equal` to the current net on 42 real observations, for tails of zeros, U[0,1] and randn x 1000.
- **Learns.** After one Adam step at 3e-4, the last layer moved, and the first layer then got gradient (max 2.5e-3).
- **An all-zero init is dead.** Only the last bias ever moves.

Exactness holds on any backend: the branch adds an exact 0.0 (for a finite input). This matters because the live run trains on MPS (the GPU).

**Alternatives considered:**

| Alternative | Problems |
|---|---|
| Widening the LSTM input with zero columns | Exact on CPU at batch sizes 1-640, but exactness depends on summation order, and that is untested on MPS. It also moves `_split_cycle`'s last-24-columns window unless that is rewritten as a forward offset. |
| Widening the head layers | Four call sites change, and the placement context's columns shift. |

### 5.5 Keeping old checkpoints and the league exact

**Migrate; don't rely on the flexible loader.**
- `load_state_dict_flexible` returns False for ANY non-strict load, even a lossless one (`policy_io.py:38-83`). On False:
  - the optimizer is not restored (`base_trainer.py:835-849`);
  - phase 2 forces an entropy re-boost.
- A migration script avoids both. It:
  - adds the branch's keys at their init;
  - extends the optimizer's parameter list (new parameters need no moments; `setup_ab_arm._widen_optimizer` remaps by name);
  - writes a NEW file under `runs/ab_*/`;
  - proves `torch.equal` outputs before anything trains, as `migrate_checkpoint_elixir_phase.py:verify` does.
- Precedent: on 2026-08-28 the loader discarded widened heads, and the win rate fell 0.80 → 0.15 (about 1,500 episodes lost; `DECISIONS.md:7648`).

**The reused opponent net has a stale-weight hazard.**
- `selfplay_env` reuses one `opponent_net` per env (`:76,128-135`).
- Measured: after a new-layout snapshot is loaded, loading an old snapshot leaves the trained tracker weights in place. The old snapshot would then play with another copy's tracker.
- Fix: the loader resets the branch to its init when its keys are absent, and the env passes a zero tail to tracker-less snapshots. Then all 456 old snapshots play exactly as before, with no migration of the pool. Copying the pool (3.5 GB) would be the alternative; overwriting it needs the user's yes.

**Tools:**
- `bench.py` loads flexibly and appends the tail through `gym_wrapper`: fine for evaluation.
- `h2h.py` loads strictly and builds one observation for both nets. It needs per-net layouts, plus one tracker per game and team.
- `perception/live/mvp_loop.py:309-317` allows only two missing key prefixes, so the new keys must be added there.

**Code that assumes the observation width:**

| Code | Change needed |
|---|---|
| `observation_space` in both envs | Widen it |
| `perception_encoder.OBSERVATION_SIZE` | Keep it as the engine width, plus a separate tail constant |
| `net.scalar_size` | Keep it as the engine width; add `tracker_start` / `tracker_size` |
| Expert-iteration and behaviour-cloning datasets (`expert_iteration.py:417`, `bc_pretrain.py:146`) | Check the width |
| `rl/buffer.py` | None: it is key-based. Memory grows about 3.6%. |

Decide deliberately whether the net zero-fills a missing tail (keeping engine-width callers and 50+ tests working) or requires it. The project's rule is "never a silent zero".

### 5.6 Tests (each fails before and passes after)

1. **Tracker conformance in the simulator:** hand, next card and elixir against engine truth. This is the control that must fire.
2. **Mirror, Spirit Empress, Collector, ability and refund events** each move the elixir estimate exactly.
3. **One event stream gives identical features** through the simulator adapter and the perception adapter.
4. **The zero-init branch is `torch.equal`** for any finite tail (template: `test_placement_hires.py::test_zero_init_is_bit_identical`).
5. **A migrated checkpoint is exact.** It is also clean: the optimizer is restored and there is no re-boost.
6. **An old snapshot loaded after a new-layout one plays exactly as before** (the stale-weight case).

Then:
- Layout tests: `test_opponent_cycle_observation.py:57`, `test_scalar_encoder_branches.py`, and the perception encoder tests update or stay engine-width per the decision above.
- The side null: a policy against a bit-exact copy scores about 0.50. CLAUDE.md requires it after an observation change.

### 5.7 The A/B, after Monday's decision

**Timing.**
- Start after the 2026-11-23 23:41 benchmark's decision has been applied. One change per A/B: if that decision adds teacher games, that change goes first, and the tracker A/B starts from the resulting main line.
- The restart can also carry the KL logging (TODO L22). It is logging only, in both arms.

**Arms**, from the same migrated checkpoint, at the same minute, with the same settings, as in the floor A/B (`DECISIONS.md:8390-8503`):
- **Control: the live run with the tail forced to zero.** The code is the same, so the only difference is the information.
- **T: the tail live**, with `CLASH_ARTIFACT_DIR=runs/ab_tracker/armT`.
- **The pool needs seeding.** `discover_historical_checkpoints` reads the arm's OWN folder once `CLASH_ARTIFACT_DIR` is set (`league.py:68`, `checkpointing.py:216`). An arm left alone starts phase 2 with no past copies. Seed `runs/ab_tracker/armT/historical_checkpoints/` with symlinks to the shared pool; check that the lineage filter reads each target's mtime. Nothing shared is written.

**Budget:** fixed in advance, for example 600 PPO updates per arm, like the floor A/B. Each arm's frozen copy is its first save at or after the budget.

**Judgement** (the user fixes the rule before launch; proposed):
- **Primary:** the head-to-head of T's copy against the control's copy. 200 games, deals paired with sides swapped, each net with its own observation. It is the more sensitive instrument, since both nets play the same deals. Equal nets reach 0.55 about 9% of the time at 200 games.
- **Guard:** `bench.py --rung 10` on both copies.
- **Reported, not deciding:** rung 3, the tracker's accuracy in play, and plays per card.

**Checks:**
- At launch, the two arms' outputs are `torch.equal`.
- After the budget, a probe that must fire: shuffling T's tail changes its actions. If it does not, T never used the tracker, and a null result means nothing.

---

## 6. Findings to act on, in order

1. **The live path is not yet the trained path** (section 1.5): opponent blocks and spend zeroed, no hand-lock mask, the Hero never activated. All of it needs the opponent placement detector, which is blocked on data. This comes before any live strength claim.
2. **The opponent tracker** (section 5): new information for the policy, and the same code fixes item 1's opponent inputs once vision emits play events.
3. **League breadth** (sections 3.3-3.4):
   - opponents on decks other than the mirror;
   - the scripted floor scaled to the pool size;
   - exploiters only after the ability port and a re-measure of the reward-hole condition.
4. **Search with sampled opponent hands** (section 4.3), on top of the tracker. The engine is fast enough. The cost is the network, and the old search wins were measured with perfect information.
5. **Smaller gaps found on the way:**
   - the spend scalars leave out ability spend (both sides);
   - shields, freezes and slows, and the own next card, are absent from the observation;
   - own invisible units are dropped from the policy's own view;
   - the Elixir Collector's engine lifetime (known, `LIVE_SYNC.md`);
   - `selfplay_env._poll_opponent_play` scans the trainee's deck, so the next-card labels miss scripted opponents' other cards (about 2.7% of games);
   - `advisors/tactics.opp_elixir_estimate` applies no cap during the match (it feeds the solvency gate, which ships off).
6. **Network capacity** (section 2) is LSTM-dominated, with a 25k-parameter input side. A capacity change is an A/B, and it resets comparability.
