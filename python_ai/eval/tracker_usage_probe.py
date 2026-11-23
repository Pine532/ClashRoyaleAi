"""Does a tracker-arm policy actually USE its tracker tail?

    python -m python_ai.eval.tracker_usage_probe CKPT [--deck IDS] [--games 6]
        [--every 3] [--seed 1]

Records observations (tracker tail included) by playing the checkpoint in a
tracker-on env, then runs the policy twice on each recorded observation: with
its REAL tail, and with the tail of a sample from a DIFFERENT game (a seeded
permutation with no same-game pairs). The engine part of the observation, the
hidden state (zeros: each sample is a stateless one-step read) and the card mask
are identical in both runs, so any difference is the tail's doing.

Reported, over the samples:
  card_tv_mean / placement_tv_mean   mean total-variation distance between the
        two runs' card distributions and placement distributions. Placement is
        read for the card the REAL-tail run's argmax chooses, in both runs.
  card_argmax_changed / placement_argmax_changed   share of samples whose argmax
        differs.
  tails_changed_share   share of samples whose shuffled tail differs from the
        real one (the control that the comparison compared something).

Control: at init, or on a migrated checkpoint, the branch's last layer is zero
and adds exactly 0, so every distance is exactly 0 (tests/test_tracker_usage_probe).
After the A/B budget a nonzero distance shows the arm uses the tracker. The
distance says the output moves with the tail, not that the move helps; the
A/B's verdict is the head-to-head.

Approximation: no recurrent history (hidden state zero per sample) and the
affordability mask without the post-cycle hand lock.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import torch
from torch.distributions import Categorical

import python_ai  # noqa: E402,F401  -- finds the engine

from python_ai.models.policy_io import LSTM_HIDDEN  # noqa: E402
from python_ai.rl import hand_slots  # noqa: E402

BATCH = 64


def record_observations(net, deck, games=6, every=3, max_ticks=3600, seed=1):
    """(obs (N, net.obs_size) float32, game ids (N,)): every `every`-th decision
    of `games` tracker-on games the net plays as team 0 (sampled actions,
    abilities unused). The env is the training env (C++ heuristic opponent)."""
    from python_ai.envs.gym_wrapper import MicroRoyaleEnv
    saved = os.environ.get("CLASH_OPP_TRACKER")
    os.environ["CLASH_OPP_TRACKER"] = "on"
    try:
        rows, gids = [], []
        torch.manual_seed(seed)
        for g in range(games):
            env = MicroRoyaleEnv({"ai_deck": list(deck), "opp_deck": list(deck),
                                  "max_ticks": max_ticks})
            obs, _ = env.reset(seed=seed + g)
            hx = torch.zeros(1, LSTM_HIDDEN)
            cx = torch.zeros(1, LSTM_HIDDEN)
            step = 0
            while True:
                if step % every == 0:
                    rows.append(obs.copy())
                    gids.append(g)
                with torch.no_grad():
                    t = torch.tensor(obs[None])
                    ready = hand_slots.ready_from_engine(env.game, 0, net.hand_size)
                    mask = net.affordability_mask(t, ready)
                    feats, emb, spatial, hires = net.extract_features_hires(t)
                    logits, _, _, _, (hx, cx) = net.step_lstm_and_card(feats, (hx, cx), mask)
                    card = Categorical(logits=logits).sample()
                    place = net.placement_given_card(hx, emb, card, t, spatial, hires_map=hires)
                    cell = Categorical(logits=place).sample()
                    x, y = net.cell_to_xy(cell)
                obs, _, term, trunc, _ = env.step({
                    "card_index": int(card), "target_x": float(x), "target_y": float(y)})
                step += 1
                if term or trunc:
                    break
        return np.stack(rows).astype(np.float32), np.asarray(gids, dtype=np.int64)
    finally:
        if saved is None:
            os.environ.pop("CLASH_OPP_TRACKER", None)
        else:
            os.environ["CLASH_OPP_TRACKER"] = saved


def other_game_permutation(gids, seed=0):
    """perm with gids[perm[i]] != gids[i] for every i (needs >= 2 games)."""
    gids = np.asarray(gids)
    if len(set(gids.tolist())) < 2:
        raise ValueError("the shuffle needs samples from at least two games")
    rng = np.random.default_rng(seed)
    perm = np.empty(len(gids), dtype=np.int64)
    for i, g in enumerate(gids):
        perm[i] = rng.choice(np.flatnonzero(gids != g))
    return perm


@torch.no_grad()
def _dists(net, obs, card=None):
    """(card probs, chosen card, placement probs) for a batch, stateless."""
    hx = torch.zeros(len(obs), LSTM_HIDDEN)
    cx = torch.zeros(len(obs), LSTM_HIDDEN)
    mask = net.affordability_mask(obs)
    feats, emb, spatial, hires = net.extract_features_hires(obs)
    logits, _, _, _, (hx, cx) = net.step_lstm_and_card(feats, (hx, cx), mask)
    probs = torch.softmax(logits, dim=-1)
    chosen = probs.argmax(dim=-1) if card is None else card
    place = net.placement_given_card(hx, emb, chosen, obs, spatial, hires_map=hires)
    return probs, chosen, torch.softmax(place, dim=-1)


def _tv(p, q):
    return 0.5 * (p - q).abs().sum(dim=-1)


def tail_usage(net, obs, gids, seed=0):
    """The probe's numbers (see the module docstring) for `net` on `obs`."""
    if not net.expects_tail:
        raise ValueError("the net has no tracker branch")
    obs = torch.as_tensor(obs, dtype=torch.float32)
    perm = other_game_permutation(gids, seed)
    shuffled = obs.clone()
    s = net.tracker_start
    shuffled[:, s:] = obs[perm, s:]
    net.eval()
    ctv, ptv, cch, pch = [], [], [], []
    for i in range(0, len(obs), BATCH):
        a, b = obs[i:i + BATCH], shuffled[i:i + BATCH]
        pa, ca, qa = _dists(net, a)
        pb, _, qb = _dists(net, b, card=ca)
        ctv.append(_tv(pa, pb))
        ptv.append(_tv(qa, qb))
        cch.append((pa.argmax(-1) != pb.argmax(-1)).float())
        pch.append((qa.argmax(-1) != qb.argmax(-1)).float())
    cat = lambda xs: torch.cat(xs)  # noqa: E731
    return {
        "samples": int(len(obs)),
        "games": int(len(set(np.asarray(gids).tolist()))),
        "card_tv_mean": float(cat(ctv).mean()),
        "placement_tv_mean": float(cat(ptv).mean()),
        "card_argmax_changed": float(cat(cch).mean()),
        "placement_argmax_changed": float(cat(pch).mean()),
        "tails_changed_share": float((obs[:, s:] != shuffled[:, s:]).any(dim=1).float().mean()),
    }


def main(argv=None):
    from python_ai.deck import parse_deck
    from python_ai.eval.arm_h2h import Side
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--deck", default=os.environ.get("CLASH_DECK"))
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    side = Side(a.ckpt, "on", parse_deck(a.deck))
    obs, gids = record_observations(side.net, side.deck, a.games, a.every, seed=a.seed)
    print(json.dumps(tail_usage(side.net, obs, gids, a.seed), indent=2))


if __name__ == "__main__":
    main()
