# Why the first live-sync run stalled at rung 0 (2026-11-18)

**Summary.** A from-scratch phase-1 run on the synced engine stalled at rung 0. The live sync
did not cause it: an A/B against `main`, same seed and config, stalls the same way. The cause is
this project's documented cold start, a fresh network against the 16-deck meta pool. A
pre-existing mask bug (the hand-slot lock) was found on the way.

The raw outputs lived in scratch and are not kept. The numbers below are copied from them.

## The symptom

From-scratch run: the 2.6 Hog Cycle deck, CPU, phase 1, rung 0, `CLASH_PHASE1_DECK_POOL=1` (the
default).

| Updates | Episodes | `Win_Rate_100` |
|---|---|---|
| 0-35 | 0-1,000 | 0.01 |
| about 50 | about 1,500 | 0.07 |
| about 80 | about 2,500 | 0.09 |
| 112 | 3,387 | 0.06 |

- Average reward went from -2.4 to between -1.4 and -1.8.
- The rung gate is 0.65.
- `[FLOOR]` fired at 1,167, 2,170 and 3,171 episodes.
- The run was healthy: no non-finite updates, `Ratio_Dev_First_Minibatch` at or below 1.1e-5.

## Four checks (read-only; synced arm against `main`'s engine, built separately)

**1. Opponent difficulty: not harder.** Rung 0 is identical in code and config on both arms (an
empty `git diff` for `opponents/`, `envs/`, `rl/curriculum.py` and the deck pool). Trainee
driven by fixed policies, full matches, scenarios off, 95% Wilson CIs:

| Trainee policy | Synced | `main` |
|---|---|---|
| always pass (control) | 0/300 | 0/300 |
| uniform-random legal | 1.0% [0.3, 2.9] | 0.0% [0, 1.3] |
| UtilityTeacher, stage 0 | 9.7% [6.8, 13.5] | 12.3% [9.1, 16.5] |
| UtilityTeacher, stage 4 (n=150) | 31.3% [24.5, 39.1] | 27.3% [20.8, 35.0] |
| UtilityTeacher, stage 10 (n=150) | 77.3% [70.0, 83.3] | 48.7% [40.8, 56.6] |

- A competent policy beats rung 0 more often on the synced engine than on `main`.
- The pool decks that a weak policy never beats are the same on both arms: P.E.K.K.A. bridge
  spam, Three Musketeers bridge, Miner poison, Giant double dragon, Royal Hogs furnace.
- Regulation ends at tick 1,800, so about 160-200 decisions is a normal episode, not a symptom.

**2. Reward: unchanged.**
- The reward code (`rewards/`, `envs/gym_wrapper.py`, `rl/engine_stats.py`) is byte-identical
  to `main`.
- Under the same random policy (n=100 per arm), every shaping term is within about 1.1x of
  `main`, with the same share of nonzero episodes.
- The only term tied to a changed engine value is the lethal-spell window, which follows
  Fireball's Crown Tower damage (159 vs 689). It is worth about +0.003 per episode.
- The shaping is loss-heavy by design: for the trained policy, -0.75 shaping against -0.67
  terminal per episode. That is the same on `main`.

**3. Action masks: one gap, and it is the same on `main`.** About 2M forked actions, judged by
elixir actually spent:
- 0 lost options: nothing the engine accepts is masked.
- The unaffordable control behaves.
- About 0.8% of placements are refused next to existing buildings.
- **The gap:** after a card is played, the slot it cycles into is locked for 20 ticks
  (`PlayerState::playCard`). Nothing in the observation carries the lock, so the mask called
  the slot legal. 10-16% of card attempts were silently refused: 15.9% for the trained policy
  and 10.7% for random play on synced, 9.8% for random play on `main`.
- Fixed afterwards (below).

**4. Observation layout: correct.**
- Every training-path consumer reads the 14403 layout at forward offsets derived from the
  bindings.
- No observation normalisation exists.
- A known state built with the estimator setters reads back exactly through `net.py`, the
  teacher and the shaping.
- Control: the same state repacked in the old 185-wide layout fails 8 checks.
- The checkpoint's input layers match.

## The A/B: `main` vs synced, 25 minutes each, fresh, `CLASH_SEED=20261118`

Both arms ran from exported copies in scratch with identical config, one after the other, each
with the machine to itself. The seed fixes the torch and worker seeds only; the two networks
differ in input width.

| Update | `main` episode | `main` win rate | synced episode | synced win rate | stalled run episode | stalled run win rate |
|---|---|---|---|---|---|---|
| 40 | 1,384 | 0.04 | 1,295 | 0.00 | 1,306 | 0.01 |
| 60 | 2,050 | 0.05 | 1,874 | 0.04 | 1,894 | 0.04 |
| 83 | 2,825 | 0.09 | 2,553 | 0.05 | 2,570 | 0.06 |

- Both arms stayed at rung 0 and fired `[FLOOR]` at 1,000 and 2,000 episodes.
- With 100 games per reading (SE about 0.03), the arms are within noise of each other.
- Both are far from 0.65.

## Cause

The documented cold start (`.claude/CLAUDE.full.md`: "THE COLD START IS REAL AND IS NOT
SOLVED"; `docs/TODO.md` 0a). Since 2026-09 the phase-1 opponent plays a pool of 16 meta decks on
an 11-rung ladder. TODO 0a records that setup as "not yet run at length", meant to be *resumed*
from a trained checkpoint. A fresh net against that pool barely wins. No trained checkpoint
exists on this machine, so the run began cold, against the pool.

The earlier comparison with an August run that "passed its first gate at episode 1,615" was
wrong: that run played the mirror on the old 6-rung ladder.

## Decisions taken (user, 2026-11-18)

- Test the documented remedy first: `CLASH_PHASE1_DECK_POOL=0` (the mirror) on a scratch copy of
  the stalled checkpoint, 25 minutes. If the win rate climbs, run long in mirror mode and turn
  the pool on later. If it stalls, report before trying the fallback (disabling the decks a
  fresh net never beats).
- Fix the hand-slot lock regardless: mask a slot until the engine reports it ready again,
  however close together decisions are. Draft an upstream report, since `main` has it too.
- The engine stays as it is until the cause is known.

## Mirror test: it climbs

Setup:
- A scratch copy of the stalled checkpoint (episode 3,387), resumed with
  `CLASH_PHASE1_DECK_POOL=0` for 25 minutes.
- Code at `59fbd9c`, before the mask fix.
- Criteria fixed before the run: "climbs" means a rung advance, or a final win rate of at least
  0.25 with the last third of readings averaging at least 0.10 above the first third.

| Episode | 3,250 | 3,500 | 3,750 | 4,000 | 4,250 | 4,500 | 4,750 | 5,000 |
|---|---|---|---|---|---|---|---|---|
| `Win_Rate_100` | 0.05 | 0.09 | 0.19 | 0.18 | 0.24 | 0.20 | 0.36 | 0.48 |

- 78 updates, episodes 3,409 to 5,176.
- First third of readings: 0.10. Last third: 0.32. Final: 0.48 (best 0.49).
- Average reward went from -1.3 to +1.0.
- Still rung 0 (gate 0.65). No plateau advances, and every update was finite.
- Verdict: **climbs**.

The same network that sat at 0.06 against the pool reaches 0.48 against its own deck in about
1,800 episodes. That confirms the opponent pool, not the network or the engine, is what held it.

The hand-slot lock fix was made afterwards (`393c497`). On this checkpoint's policy, engine-refused
card attempts fell from 22.6% to 3.0%.

## After the mask fix and the values lookup: the 10-minute check does not climb (2026-11-19)

Setup:
- The same start as the mirror test: a scratch copy of the episode-3,387 checkpoint, mirror mode.
- Code at `a26a041`: the hand-slot mask fix plus the values-lookup changes.
- Criteria fixed beforehand: a final `Win_Rate_100` of at least 0.15, and the last third at
  least 0.05 above the first third.

Result after 36 updates (episodes 3,410 to 4,315):
- First third 0.091, last third 0.106, final 0.100.
- The run was healthy: ratio deviation at most 7.6e-6, no non-finite updates.
- The mirror test had reached 0.170 at the same update count.
- Verdict: **does not climb**, so the long run was not started.

Not yet separated: both changes touch this deck. The Cannon and the Skeletons gained first-hit
wind-ups, and the mask removes about 20% of this policy's card attempts, which it was trained
with. Either could account for it, and so could noise: with 100 games per reading the standard
error is about 0.03-0.04.
