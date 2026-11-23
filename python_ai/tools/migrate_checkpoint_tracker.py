"""Migrate a checkpoint to the opponent-tracker layout, keeping the optimizer.

    python -m python_ai.tools.migrate_checkpoint_tracker \\
        --in OLD.pth --out NEW.pth --mode on|zero [--seed S]

(docs/ROADMAP_TOP_PLAYERS.md section 5; the A/B rule is in docs/DECISIONS.md,
"2026-11-22 (late)".)

What changes
------------
The tracker net is today's net plus `tracker_branch` (Linear -> ReLU -> Linear),
registered LAST. The new file holds the input's weights untouched, plus the
branch at its init: a first layer seeded by --seed and a ZERO last layer
(weight and bias), so the migrated net's outputs equal the old net's exactly
for any finite tail. verify() proves that on real engine observations.

The optimizer
-------------
Adam's state is keyed by parameter POSITION. The branch is registered last, so
its four parameters take positions n_old .. n_old+3: they are appended to
`param_groups[0]['params']` and get NO state entry (Adam initialises it lazily
on the first step). Every old parameter keeps its position, so its moments are
carried bit-for-bit. A migration that dropped the optimizer would restart Adam
cold on a trained policy; one that kept it unextended would fail
`optimizer.load_state_dict` (BaseTrainer.restore_common) on a parameter-count
mismatch, which a trainer treats as fatal.

Every other field is copied unchanged: episodes, PFSP stats and counts, the
entropy controller, the lineage, the deck, the reference roster, curriculum.

The recorded settings
---------------------
CLASH_OPP_TRACKER is a training setting (tracking/config.py), so a resume
under a different value than the one recorded in `clash_settings` prints a
"DIFFERENT CLASH_* SETTINGS" warning (BaseTrainer._report_settings_drift). That
is ALL it does: it re-boosts nothing and resets nothing, in phase 1
(trainers/train.py load_checkpoint -> restore_common, curriculum restored
as saved) and phase 2 (trainers/train_selfplay.py: only an UNCLEAN model load
forces the entropy re-boost). The warning would still be a false alarm, because
this migration is what makes the file a tracker checkpoint. So the recorded
`clash_settings["CLASH_OPP_TRACKER"]` is stamped with --mode, and a resume
under that mode is silent. A file with no `clash_settings` stamp (legacy) is
left without one, as it already was.
"""
import argparse
import os
import sys

# Put the repo root on sys.path, for script and -m invocation alike.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

import python_ai  # noqa: F401  -- the sys.path append that finds the .pyd
from python_ai.models.net import MicroRoyaleNet
from python_ai.models.policy_io import LSTM_HIDDEN
# atomic_save, never torch.save: a torn write would truncate the file a run
# resumes from.
from python_ai.rl.checkpointing import atomic_save
from python_ai.tracking import config as tracker_config

BRANCH_PREFIX = "tracker_branch."
SETTING = tracker_config.ENV_VAR


def ability_slots_of(state):
    """The ability-head count the checkpoint was trained with (from its keys)."""
    if "ability_slot2_head.weight" in state:
        return 2
    if "ability_slot1_head.weight" in state:
        return 1
    return 0


def build_tracker_net(state, mode, seed):
    """A tracker net holding `state`, with the branch at its init.

    The branch's first layer is drawn under `seed` (the global RNG is restored
    afterwards); the last layer is zeroed by reset_tracker_branch_to_init.
    """
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        net = MicroRoyaleNet(num_ability_slots=ability_slots_of(state),
                             tracker_size=tracker_config.tail_width(mode))
    res = net.load_state_dict(state, strict=False)
    bad = [k for k in res.missing_keys if not k.startswith(BRANCH_PREFIX)]
    if bad or res.unexpected_keys:
        raise SystemExit(
            f"!! the input does not fit the tracker net: missing {bad}, "
            f"unexpected {list(res.unexpected_keys)}. This tool only appends "
            "the tracker branch; another architecture change needs its own "
            "migration first.")
    net.reset_tracker_branch_to_init()
    return net


def widen_optimizer(opt, net):
    """Append the branch's parameter positions to `opt`'s only group. In place.

    The branch must be the LAST parameters of the net, and the saved group must
    be exactly the leading positions: otherwise the append would hand moments
    to the wrong tensors.
    """
    names = [n for n, _ in net.named_parameters()]
    n_branch = sum(1 for n in names if n.startswith(BRANCH_PREFIX))
    n_old = len(names) - n_branch
    if not all(n.startswith(BRANCH_PREFIX) for n in names[n_old:]):
        raise SystemExit("!! the tracker branch is not registered last: appending "
                         "optimizer positions would be wrong. Remap by hand.")
    groups = opt["param_groups"]
    if len(groups) != 1:
        raise SystemExit(f"!! expected one optimizer param group, found {len(groups)}")
    params = list(groups[0]["params"])
    if params != list(range(n_old)):
        raise SystemExit(
            f"!! the optimizer describes parameters {params[:3]}...({len(params)}) "
            f"but the old net has {n_old} (positions 0..{n_old - 1}): the saved "
            "optimizer does not match the saved model.")
    groups[0]["params"] = params + list(range(n_old, n_old + n_branch))
    return n_old, n_branch


def migrate(ck, mode, seed):
    """Return the migrated checkpoint dict (a shallow copy of `ck`)."""
    if not (isinstance(ck, dict) and "model" in ck):
        raise SystemExit("!! the input is a bare state_dict, not a training "
                         "checkpoint: there is no optimizer or run state to keep.")
    state = ck["model"]
    if any(k.startswith(BRANCH_PREFIX) for k in state):
        raise SystemExit("!! the input already has tracker_branch keys; nothing "
                         "to migrate.")
    net = build_tracker_net(state, mode, seed)
    out = dict(ck)
    out["model"] = {k: v.detach().clone() for k, v in net.state_dict().items()}
    if ck.get("optimizer") is None:
        print("  no optimizer state in the input (a weights-only file)")
    else:
        import copy
        out["optimizer"] = copy.deepcopy(ck["optimizer"])
        n_old, n_branch = widen_optimizer(out["optimizer"], net)
        print(f"  optimizer: {n_old} old parameters keep their state, "
              f"{n_branch} branch parameters appended with none (lazy init)")
    saved = ck.get("clash_settings")
    if saved is None:
        print("  no clash_settings stamp in the input; none added")
    else:
        out["clash_settings"] = dict(saved, **{SETTING: mode})
        print(f"  clash_settings[{SETTING}]: {saved.get(SETTING)!r} -> {mode!r}")
    return out


# -- verification ------------------------------------------------------------
def _real_observations(n, deck=None):
    """Engine-width observations from a real engine, at varied ticks and both
    teams' views. `n` must be even (the update path views them as L=2 x B)."""
    import clash_royale_env as E
    from python_ai.envs import gym_wrapper
    deck = list(deck) if deck else list(gym_wrapper.DEFAULT_DECK)
    env = E.ClashRoyaleEnv(deck, deck, 3600)
    env.reset()
    rows = []
    for i in range(n):
        env.set_current_tick(40 * i)
        rows.append(torch.tensor(env.get_observation_for_team(i % 2), dtype=torch.float32))
    return torch.stack(rows)


def _flatten(x):
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return [x]
    out = []
    for item in x:
        out += _flatten(item)
    return out


def _rollout_outputs(net, obs):
    """The rollout path (BaseTrainer.collect_rollout's calls)."""
    feats, embeds, spatial, _hires = net.extract_features_hires(obs)
    n = obs.shape[0]
    hidden = (torch.zeros(n, LSTM_HIDDEN), torch.zeros(n, LSTM_HIDDEN))
    mask = net.affordability_mask(obs)
    card, ab1, ab2, value, (hx, cx) = net.step_lstm_and_card(feats, hidden, mask)
    place = net.placement_given_card(hx, embeds, torch.zeros(n, dtype=torch.long),
                                     obs, spatial)
    aux = net.predict_opp_next_card(hx)
    cyc = net.predict_cycle_card(net.cycle_features(obs))
    return _flatten([feats, card, ab1, ab2, value, place, aux, cyc, hx, cx])


def _update_outputs(net, obs):
    """The update path (rl/ppo.py's forward_sequence)."""
    n = obs.shape[0]
    L, B = 2, n // 2
    feats, embeds, spatial = net.extract_features(obs)
    mask = net.affordability_mask(obs).view(L, B, -1)
    hidden = (torch.zeros(B, LSTM_HIDDEN), torch.zeros(B, LSTM_HIDDEN))
    out = net.forward_sequence(
        feats.view(L, B, -1), embeds.view(L, B, *embeds.shape[1:]),
        spatial.view(L, B, *spatial.shape[1:]), obs.view(L, B, -1), mask,
        torch.zeros(L, B, dtype=torch.long), torch.ones(L, B), hidden,
        with_ability=True)
    return _flatten(out)


def verify(old_path, new_path, mode, n_obs=8):
    """Prove the migrated file is what it claims. Returns a dict of numbers.

    1. A STRICT load into a fresh tracker net succeeds, and the optimizer state
       loads into an Adam over that net.
    2. On real engine observations, for several finite tails (zeros, uniform,
       huge), the migrated net's outputs equal the old net's under torch.equal,
       on the rollout path and the update path.
    3. Every old parameter's Adam state is bit-equal to the input's.
    """
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    new = torch.load(new_path, map_location="cpu", weights_only=False)
    slots = ability_slots_of(old["model"])
    width = tracker_config.tail_width(mode)

    old_net = MicroRoyaleNet(num_ability_slots=slots).eval()
    old_net.load_state_dict(old["model"])
    new_net = MicroRoyaleNet(num_ability_slots=slots, tracker_size=width).eval()
    new_net.load_state_dict(new["model"])          # strict
    stats = {"tail_width": width, "tensors_compared": 0, "max_abs_diff": 0.0}

    if new.get("optimizer") is not None:
        import torch.optim as optim
        opt = optim.Adam(new_net.parameters(), lr=1e-3)
        opt.load_state_dict(new["optimizer"])
        oo = old["optimizer"]["state"]
        nn_ = new["optimizer"]["state"]
        if set(nn_) != set(oo):
            raise SystemExit("!! the migrated optimizer state has other positions "
                             "than the input's")
        for i, entry in oo.items():
            for k, v in entry.items():
                if not torch.equal(torch.as_tensor(v), torch.as_tensor(nn_[i][k])):
                    raise SystemExit(f"!! optimizer state {i}.{k} differs")
        stats["optimizer_entries_bit_equal"] = len(oo)

    obs = _real_observations(n_obs, old.get("deck"))
    g = torch.Generator().manual_seed(1)
    tails = {"zeros": torch.zeros(n_obs, width),
             "uniform": torch.rand(n_obs, width, generator=g),
             "huge": torch.randn(n_obs, width, generator=g) * 1e3}
    with torch.no_grad():
        for kind, tail in tails.items():
            full = torch.cat([obs, tail], dim=1)
            for fn in (_rollout_outputs, _update_outputs):
                a, b = fn(old_net, obs), fn(new_net, full)
                if len(a) != len(b):
                    raise SystemExit(f"!! {fn.__name__} output arity differs")
                for x, y in zip(a, b):
                    if not torch.equal(x, y):
                        raise SystemExit(f"!! {fn.__name__} differs at tail '{kind}' "
                                         f"(max abs {(x - y).abs().max().item():.3e})")
                    stats["tensors_compared"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, help="checkpoint to read")
    ap.add_argument("--out", dest="dst", required=True,
                    help="file to write (must not exist)")
    ap.add_argument("--mode", required=True, choices=("on", "zero"),
                    help="CLASH_OPP_TRACKER the resumed run will use; stamped "
                         "into the recorded settings")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the branch's first layer (default 0)")
    a = ap.parse_args()

    src, dst = os.path.abspath(a.src), os.path.abspath(a.dst)
    if src == dst or (os.path.exists(dst) and os.path.samefile(src, dst)):
        raise SystemExit("!! --out must differ from --in; the input stays untouched.")
    if os.path.exists(dst):
        raise SystemExit(f"!! {dst} exists; refusing to overwrite.")
    if not os.path.exists(src):
        raise SystemExit(f"!! no such checkpoint: {src}")

    ck = torch.load(src, map_location="cpu", weights_only=False)
    print(f"read {src} (episode {ck.get('episodes_completed', '?')})")
    print("migrating:")
    out = migrate(ck, a.mode, a.seed)
    atomic_save(out, dst)
    print(f"wrote {dst}")

    print("verifying:")
    stats = verify(src, dst, a.mode)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"OK -- resume with {SETTING}={a.mode}.")


if __name__ == "__main__":
    main()
