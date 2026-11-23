"""Head-to-head of two checkpoints where each net gets its OWN observation layout.

    python -m python_ai.eval.arm_h2h --a A.pth --a-mode on --b B.pth --b-mode zero \\
        --deck 129,184,126,183,0,31,24,101 --work DIR [--n 400] [--out-md ROW.md]

The tracker A/B's main measure (docs/DECISIONS.md, "2026-11-22 (late)"): the
tracker arm (A) against the control arm (B), 400 games, sides swapped. Based on
the logic of runs/tools_scratch/h2h.py (the phase-2 benchmark; that file is not
touched), which gives both nets one engine-width observation and so cannot play
a tracker net.

Per side
--------
mode  off   an old-layout net: the engine's observation, no tail, no feed.
      zero  a tracker-layout net shown an all-zero tail (the control arm's net).
      on    a tracker-layout net shown the tracker tail, from its OWN TrackerTail
            per game with that side as the observer (tracking/tail.py).
The layout comes from the checkpoint's keys (the tracker_branch's input width;
the ability-head count) and the load is STRICT. A mode that does not match the
layout (off with a tracker net, zero/on without a branch) is refused: no silent
zero tail, no unfed branch.

Per decision (as envs/selfplay_env does it)
-------------------------------------------
Both sides' observations are built and both tails POLLED first; each side then
decides; every ability flag is noted, BEFORE the step, on each side's tail as
its own press and as the other's opponent press (readiness as the deciding side
used it); then the step. A press noted at the current tick stays pending in
EngineFeed.poll until the step has executed, so no side sees the other's
same-step press.

Games, sampling, scoring
------------------------
Games come in pairs: pair p deals with env.seed(seed0 + p); A is team 0 in the
first game of the pair and team 1 in the second. Both sides SAMPLE card and cell
(and abilities) as training does; no scenario injection, no C++ heuristic
(step_self_play). Scored by the engine's verdict (eval/match_outcome): win 1,
draw 0.5, loss 0. The 95% interval is +-1.96 standard errors over the pair
means. torch.manual_seed(seed1) once: the same two checkpoints replay the same
games on this machine (a seed is per-build, never compared across machines).
Writes WORK/result.json and, with --out-md, appends a markdown row.
"""
import argparse
import json
import math
import os
import sys
import time

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import torch
from torch.distributions import Categorical

import python_ai  # noqa: E402,F401  -- finds the engine

import clash_royale_env as E  # noqa: E402
from python_ai.eval.match_outcome import score_from_towers  # noqa: E402
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import LSTM_HIDDEN  # noqa: E402
from python_ai.rl import abilities as ability_mod  # noqa: E402
from python_ai.rl import hand_slots  # noqa: E402
from python_ai.tracking import config as tracker_config  # noqa: E402

SEED0 = 515000          # deal seeds: SEED0 + pair index (the old h2h's)
SEED1 = 4243            # torch's sampling generator (the old h2h's)
WAVE = 20               # games stepped together (10 pairs); batches the forward passes
MAX_STEPS = 400         # a full match is 360 decisions; the guard is never expected to fire
DEFAULT_N = 400
BRANCH = "tracker_branch."


def _layout(state):
    """(tracker tail width, ability head count) of a state dict, from its keys."""
    w = state.get(BRANCH + "0.weight")
    tail = int(w.shape[1]) if w is not None else 0
    if tail == 0 and any(k.startswith(BRANCH) for k in state):
        raise SystemExit("!! a partial tracker_branch in the checkpoint")
    slots = 2 if "ability_slot2_head.weight" in state else (
        1 if "ability_slot1_head.weight" in state else 0)
    return tail, slots


class Side:
    """One checkpoint, loaded strictly into a net of its own layout."""

    def __init__(self, path, mode, deck):
        if mode not in tracker_config.MODES:
            raise SystemExit(f"!! mode {mode!r}: expected one of {tracker_config.MODES}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        meta = ck if isinstance(ck, dict) and "model" in ck else {}
        if meta.get("deck") is not None and [int(c) for c in meta["deck"]] != list(deck):
            raise SystemExit(f"!! {path} records deck {meta['deck']}, not {list(deck)}; refusing to play")
        tail, ability_slots = _layout(state)
        if len(ability_mod.ability_engine_slots(list(deck))) != ability_slots:
            raise SystemExit(f"!! {path} has {ability_slots} ability head(s); the deck needs "
                             f"{len(ability_mod.ability_engine_slots(list(deck)))}")
        if mode == "off" and tail:
            raise SystemExit(f"!! {path} is a tracker-layout net (tail {tail}); mode 'off' would "
                             "play it without its tail. Use 'zero' or 'on'.")
        if mode != "off":
            if not tail:
                raise SystemExit(f"!! {path} has no tracker_branch; mode {mode!r} needs a tracker "
                                 "net. Migrate it (tools/migrate_checkpoint_tracker.py) or use 'off'.")
            if tail != tracker_config.tail_width("on"):
                raise SystemExit(f"!! {path}: tail width {tail} is not this build's "
                                 f"{tracker_config.tail_width('on')}")
        self.path, self.mode, self.deck = os.path.abspath(path), mode, list(deck)
        self.tail_width = tail
        self.net = MicroRoyaleNet(num_ability_slots=ability_slots, tracker_size=tail)
        self.net.load_state_dict(state, strict=True)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        ep = meta.get("episodes_completed")
        self.episode = int(ep) if ep is not None else None
        #: measured ability costs, keyed by card id (shared by this side's tails)
        self.ability_costs = {}

    def make_tail(self, game, observer_team):
        """This side's tail for one game, or None for an old-layout net."""
        if self.mode == "off":
            return None
        from python_ai.tracking.tail import TrackerTail
        return TrackerTail(game, observer_team, self.mode, decks=(self.deck, self.deck),
                           ability_costs=self.ability_costs)


def observation(game, team, tail):
    """The engine's observation for `team`, with its tail (polled here) if any."""
    obs = np.asarray(game.get_observation_for_team(team), dtype=np.float32)
    return obs if tail is None else tail.append(obs)


@torch.no_grad()
def act(net, games, teams, hx, cx, slots, tails=None):
    """One decision for each (game, team), batched: phase 2's opponent path
    (envs/selfplay_env._opponent_action), as the old h2h's act, with each
    game's tail (None for an old-layout net) appended to its observation."""
    tails = [None] * len(games) if tails is None else tails
    obs = torch.tensor(np.stack([observation(g, t, tl) for g, t, tl in zip(games, teams, tails)]))
    if net.expects_tail:
        net.require_tail(obs)
    hand_ready = torch.cat([hand_slots.ready_from_engine(g, t, net.hand_size) for g, t in zip(games, teams)])
    card_mask = net.affordability_mask(obs, hand_ready)
    feats, emb, spatial, hires = net.extract_features_hires(obs)
    logits, ab1, ab2, _, (hx, cx) = net.step_lstm_and_card(feats, (hx, cx), card_mask)
    card = Categorical(logits=logits).sample()
    place = net.placement_given_card(hx, emb, card, obs, spatial, hires_map=hires)
    cell = Categorical(logits=place).sample()
    x, y = net.cell_to_xy(cell)
    flags = np.zeros((len(games), 3), dtype=bool)          # [:, slot] for engine slots 1 and 2
    ab_logits = [l for l in (ab1, ab2) if l is not None]
    if slots and ab_logits:
        ready = torch.tensor([[bool(g.is_champion_ability_ready(t, s)) for s in slots]
                              for g, t in zip(games, teams)])
        acts, _, _ = ability_mod.sample(ab_logits, ready)
        for k, s in enumerate(slots):
            flags[:, s] = acts[:, k].numpy().astype(bool)
    return (card.numpy().reshape(-1), x.numpy().reshape(-1), y.numpy().reshape(-1), flags, hx, cx)


def play_wave(side_a, side_b, game_ids, slots, deck, seed0=SEED0):
    """Plays the given game indices together.

    Returns ({game index: (A's score, A's team, decisions, finished)}, [tails of A], [tails of B]).
    """
    games, a_team = [], []
    for i in game_ids:
        g = E.ClashRoyaleEnv(deck, deck, 3600, E.TowerTroopType.NONE, E.TowerTroopType.NONE)
        g.seed(seed0 + i // 2)
        games.append(g)
        a_team.append(i % 2)                               # even: A is team 0; odd: A is team 1
    n = len(games)
    tails_a = [side_a.make_tail(g, a_team[j]) for j, g in enumerate(games)]
    tails_b = [side_b.make_tail(g, 1 - a_team[j]) for j, g in enumerate(games)]
    for j in range(n):
        for tl in (tails_a[j], tails_b[j]):
            if tl is not None:
                tl.reset((deck, deck))
    z = lambda: torch.zeros(n, LSTM_HIDDEN)
    ha, ca, hb, cb = z(), z(), z(), z()
    done, steps = [False] * n, [0] * n
    for _ in range(MAX_STEPS):
        live = [j for j in range(n) if not done[j]]
        if not live:
            break
        idx = torch.tensor(live)
        g_live = [games[j] for j in live]
        ta = [a_team[j] for j in live]
        tb = [1 - t for t in ta]
        # Both observations (and so both tails' polls) before either side's flags are noted.
        card_a, xa, ya, fa, ha_n, ca_n = act(side_a.net, g_live, ta, ha[idx], ca[idx], slots,
                                             [tails_a[j] for j in live])
        card_b, xb, yb, fb, hb_n, cb_n = act(side_b.net, g_live, tb, hb[idx], cb[idx], slots,
                                             [tails_b[j] for j in live])
        ha[idx], ca[idx], hb[idx], cb[idx] = ha_n, ca_n, hb_n, cb_n
        for k, j in enumerate(live):
            a = (int(card_a[k]), float(xa[k]), float(ya[k]), bool(fa[k, 1]), bool(fa[k, 2]))
            b = (int(card_b[k]), float(xb[k]), float(yb[k]), bool(fb[k, 1]), bool(fb[k, 2]))
            t0, t1 = (a, b) if a_team[j] == 0 else (b, a)
            # BEFORE the step: each side's press is its own on its tail and the
            # opponent's on the other's (readiness as the deciding side used it).
            for tl in (tails_a[j], tails_b[j]):
                if tl is not None:
                    tl.note_flags(0, {1: t0[3], 2: t0[4]})
                    tl.note_flags(1, {1: t1[3], 2: t1[4]})
            r = games[j].step_self_play(t0[0], t0[1], t0[2], t1[0], t1[1], t1[2], 10,
                                        t0[3], t0[4], t1[3], t1[4])
            steps[j] += 1
            if r.done:
                done[j] = True
    res = {i: (score_from_towers(games[j], a_team[j]), a_team[j], steps[j], done[j])
           for j, i in enumerate(game_ids)}
    return res, tails_a, tails_b


def _feed_stats(tails, acc):
    """Sum the feeds' diagnostic counters (logging only; never an input)."""
    for tl in tails:
        if tl is not None and tl.feed is not None:
            for k, v in tl.feed.stats.items():
                acc[k] = acc.get(k, 0) + int(v)


def run(a_path, a_mode, b_path, b_mode, deck, n, work, out_md=None, seed0=SEED0, seed1=SEED1,
        wave=WAVE):
    """Play the head-to-head; write WORK/result.json; return the result dict."""
    if n % 2 or n <= 0:
        raise SystemExit("!! N must be even and positive (games come in side-swapped pairs)")
    deck = [int(c) for c in deck]
    work = os.path.abspath(work)
    os.makedirs(work, exist_ok=True)
    side_a, side_b = Side(a_path, a_mode, deck), Side(b_path, b_mode, deck)
    slots = ability_mod.ability_engine_slots(deck)
    torch.manual_seed(seed1)
    t0 = time.time()
    res, stats_a, stats_b = {}, {}, {}
    for start in range(0, n, wave):
        part, ta, tb = play_wave(side_a, side_b, list(range(start, min(n, start + wave))), slots, deck, seed0)
        res.update(part)
        _feed_stats(ta, stats_a)
        _feed_stats(tb, stats_b)
        print(f"  {len(res)}/{n} games: A score {np.mean([res[i][0] for i in res]):.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)

    s = np.array([res[i][0] for i in range(n)])
    side = np.array([res[i][1] for i in range(n)])
    pairs = s.reshape(-1, 2).mean(axis=1)
    se = float(pairs.std(ddof=1) / math.sqrt(len(pairs))) if len(pairs) > 1 else float("nan")
    w, d, l = int((s == 1.0).sum()), int((s == 0.5).sum()), int((s == 0.0).sum())
    out = dict(a=side_a.path, a_mode=a_mode, a_tail=side_a.tail_width, a_episode=side_a.episode,
               b=side_b.path, b_mode=b_mode, b_tail=side_b.tail_width, b_episode=side_b.episode,
               deck=deck, n=n, wins=w, draws=d, losses=l,
               score=float(s.mean()), a_score=float(s.mean()), b_score=float(1.0 - s.mean()),
               ci95=[float(s.mean() - 1.96 * se), float(s.mean() + 1.96 * se)],
               score_as_team0=float(s[side == 0].mean()), score_as_team1=float(s[side == 1].mean()),
               mean_decisions=float(np.mean([res[i][2] for i in range(n)])),
               unfinished=sum(1 for i in range(n) if not res[i][3]),
               a_tracker_stats=stats_a, b_tracker_stats=stats_b,
               seeds=dict(deal0=seed0, torch=seed1), policy="stochastic (sampled), both sides",
               seconds=time.time() - t0, at=time.strftime("%Y-%m-%d %H:%M"))
    with open(os.path.join(work, "result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))
    if out_md:
        _append_row(out_md, out)
    return out


def _append_row(path, out):
    new = not os.path.exists(path)
    lo, hi = out["ci95"]
    with open(path, "a") as f:
        if new:
            f.write("# Head-to-head, each net with its own observation layout\n\n"
                    "A against B, seeded deal pairs with the sides swapped, both sampling as in training, "
                    "scored by the engine's verdict (python_ai/eval/arm_h2h.py). Score = A's wins plus half "
                    "its draws over N. 95% interval over the pair means.\n\n"
                    "| Time | A (mode, episode) | B (mode, episode) | W / D / L | A score | 95% | "
                    "A as team 0 | A as team 1 | Decisions per game |\n|---|---|---|---|---|---|---|---|---|\n")
        f.write(f"| {out['at']} | {out['a_mode']}, {out['a_episode']} | {out['b_mode']}, {out['b_episode']} "
                f"| {out['wins']} / {out['draws']} / {out['losses']} | {out['a_score']:.3f} "
                f"| {lo:.3f}-{hi:.3f} | {out['score_as_team0']:.3f} | {out['score_as_team1']:.3f} "
                f"| {out['mean_decisions']:.0f} |\n")


def main(argv=None):
    from python_ai.deck import parse_deck
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="checkpoint A (the tracker arm)")
    ap.add_argument("--a-mode", required=True, choices=tracker_config.MODES)
    ap.add_argument("--b", required=True, help="checkpoint B (the control arm)")
    ap.add_argument("--b-mode", required=True, choices=tracker_config.MODES)
    ap.add_argument("--deck", default=os.environ.get("CLASH_DECK"), help="ids or names (default CLASH_DECK)")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help=f"games, even (default {DEFAULT_N})")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--seed1", type=int, default=SEED1)
    ap.add_argument("--wave", type=int, default=WAVE)
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    run(a.a, a.a_mode, a.b, a.b_mode, parse_deck(a.deck), a.n, a.work, a.out_md,
        a.seed0, a.seed1, a.wave)


if __name__ == "__main__":
    main()
