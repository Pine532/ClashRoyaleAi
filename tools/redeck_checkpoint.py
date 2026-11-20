"""Start a new lineage on another deck from a trained phase-1 checkpoint.

    venv/bin/python tools/redeck_checkpoint.py SRC.pth DST.pth --deck 129,184,126,183,0,31,24,101

Kept: every trained weight, and Adam's moments for every parameter both nets
have, matched by NAME (a new ability head shifts the positions of the
parameters after it, so positional state would land on the wrong tensors).
New parameters, e.g. a Hero/Champion ability head when the new deck has one
at deck index 1 or 2, start from the net's own initialisation with no Adam
state.

Reset, because they describe the old deck or the old schedule:
- the curriculum to rung 0 (teacher_table_size stamped, so the index is read
  against the current table), its demotion / plateau counts and plateau
  tracker, and the mirror phase;
- the outcome window and the per-deck pool win-rate estimates;
- both entropy coefficients to their initial values, and the episode counter
  to 0. That counter is the placement-entropy anneal clock, so the anneal
  restarts wide: most of the new deck's cards have never been placed.

Recorded: the new deck, a new lineage time, and a `redeck` provenance block.
SRC is only read. DST is written with atomic_save. The CLASH_* settings stamp
is dropped (the trainer then notes it is absent, harmlessly), since it
describes the old run.
"""
import argparse
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from python_ai.models.net import MicroRoyaleNet          # noqa: E402
from python_ai.rl.abilities import ability_engine_slots  # noqa: E402
from python_ai.rl.checkpointing import atomic_save        # noqa: E402
from python_ai.rl.config import PHASE1_ENTROPY            # noqa: E402
from python_ai.rl.curriculum import CURRICULUM_STAGES     # noqa: E402

#: Keys the new lineage must not inherit; absent keys take the trainer's
#: defaults on load.
DROPPED = ("clash_settings", "deck_pool_stats", "deck_pool_counts", "best_rung_mean",
           "best_rung_episode", "rung_entry_progress", "progress_signal")


def redeck(src, dst, new_deck):
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    old_deck = [int(c) for c in ckpt.get("deck", [])]
    old_net = MicroRoyaleNet(num_ability_slots=len(ability_engine_slots(old_deck)) if old_deck else 0)
    old_net.load_state_dict(ckpt["model"])
    new_net = MicroRoyaleNet(num_ability_slots=len(ability_engine_slots(new_deck)))

    # Weights: every tensor present in both with the same shape.
    old_sd, new_sd = old_net.state_dict(), new_net.state_dict()
    fresh = []
    for name, tensor in new_sd.items():
        if name in old_sd and old_sd[name].shape == tensor.shape:
            new_sd[name] = old_sd[name].clone()
        else:
            fresh.append(name)
    new_net.load_state_dict(new_sd)

    # Adam: state by parameter name.
    old_names = [n for n, _ in old_net.named_parameters()]
    new_params = list(new_net.named_parameters())
    old_opt = ckpt["optimizer"]
    old_state = old_opt["state"]
    by_name = {old_names[i]: s for i, s in old_state.items() if i < len(old_names)}
    new_state = {}
    for j, (name, p) in enumerate(new_params):
        s = by_name.get(name)
        if s is None:
            continue
        if any(torch.is_tensor(v) and v.dim() > 0 and v.shape != p.shape for v in s.values()):
            continue
        new_state[j] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items()}
    group = {k: v for k, v in old_opt["param_groups"][0].items() if k != "params"}
    group["params"] = list(range(len(new_params)))
    new_opt = {"state": new_state, "param_groups": [group]}
    # Must load into a real Adam over the new net, or the trainer would refuse it.
    torch.optim.Adam(new_net.parameters(), lr=group.get("lr", 3e-4)).load_state_dict(new_opt)

    out = {k: v for k, v in ckpt.items() if k not in DROPPED}
    out.update(
        model=new_net.state_dict(),
        optimizer=new_opt,
        deck=list(new_deck),
        episodes_completed=0,
        outcome_history=[],
        lineage_started_at=time.time(),
        ent_coef_card=PHASE1_ENTROPY.initial_coef_card,
        ent_coef_place=PHASE1_ENTROPY.initial_coef_placement,
        curriculum_stage=0,
        teacher_table_size=len(CURRICULUM_STAGES),
        curriculum_demotions=0,
        curriculum_plateau_advances=0,
        stage_start_episode=0,
        phase="mirror",
        deck_curriculum_stage=0,
        phase_deck_episode_start=0,
        random_phase_episode_start=0,
        current_random_deck=None,
        rung_history=[],
        redeck={"from": os.path.abspath(src), "from_deck": old_deck,
                "from_episodes": int(ckpt.get("episodes_completed", 0)),
                "from_rung": int(ckpt.get("curriculum_stage", 0)),
                "fresh_params": fresh, "adam_states_kept": len(new_state),
                "at": time.strftime("%Y-%m-%d %H:%M:%S")},
    )
    atomic_save(out, dst)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--deck", required=True, help="comma-separated card ids, Hero/Champion at index 1 or 2")
    args = ap.parse_args()
    deck = [int(c) for c in args.deck.split(",")]
    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        sys.exit("refusing to overwrite the source checkpoint")
    out = redeck(args.src, args.dst, deck)
    r = out["redeck"]
    print(f"re-decked {r['from_deck']} (episode {r['from_episodes']}, rung {r['from_rung']}) -> {deck}: "
          f"{len(out['model'])} tensors, fresh {r['fresh_params']}, Adam states kept {r['adam_states_kept']}; "
          f"rung 0, episode 0, entropy {out['ent_coef_card']}/{out['ent_coef_place']} -> {args.dst}")


if __name__ == "__main__":
    main()
