"""Widen a checkpoint for a raised NUM_CARD_IDS (CARD_ID_COUNT 185 -> 256, 2026-11 live sync).

Card ids index five parameters, every one of them along a card-id axis:

    input layers, one COLUMN per id:
        scalar_mlp.hand_slot.weight   [5,  N]   hand one-hot, shared per slot
        scalar_mlp.cycle_card.weight  [16, N]   opponent seen[] / recency[]
        card_id_embed.weight          [16, N]   the chosen card's identity
    output heads, one ROW per id:
        aux_card_head.weight [N, 256], aux_card_head.bias [N]
        cycle_id_head.weight [N, 24],  cycle_id_head.bias [N]

plus `spell_flags` [N], a buffer derived from the engine, which is rebuilt
rather than padded. The network slices the observation by offsets derived
from the engine, so nothing else depends on N.

- New input columns are zero, so the migrated net's output is bit-identical to
  the original's on every observation it could already see (no new id present).
- New output rows copy the row of the highest old id (184), which was never
  registered: a new id starts exactly as likely as an id the net has never
  seen. A zero row would instead give each new id logit 0, far above the
  learned "never" logits, and spike the auxiliary losses.
- Adam's exp_avg / exp_avg_sq are padded with zeros along the same axis (no
  history, which is true); migrating the weights alone produces a file that
  fails on optimizer.load_state_dict.

The target width is read from the engine; a stale .pyd is refused.

    python -m python_ai.tools.migrate_checkpoint_card_ids --in model_weights.pth --out model_weights.ids256.pth
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

import python_ai  # noqa: F401  -- the sys.path append that finds the .pyd
import clash_royale_env
from python_ai.envs import gym_wrapper
from python_ai.models.net import MicroRoyaleNet
from python_ai.rl.checkpointing import atomic_save

INPUT_COLUMNS = ("scalar_mlp.hand_slot.weight", "scalar_mlp.cycle_card.weight", "card_id_embed.weight")
OUTPUT_ROWS = ("aux_card_head.weight", "aux_card_head.bias", "cycle_id_head.weight", "cycle_id_head.bias")
DERIVED_BUFFERS = ("spell_flags",)


def _pad_cols(t, new):
    out = torch.zeros(t.shape[0], new, dtype=t.dtype, device=t.device)
    out[:, : t.shape[1]] = t
    return out


def _pad_rows_copy_last(t, new):
    old = t.shape[0]
    out = torch.empty((new,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
    out[:old] = t
    out[old:] = t[old - 1]
    return out


def _pad_rows_zero(t, new):
    out = torch.zeros((new,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
    out[: t.shape[0]] = t
    return out


def _param_index(net, name):
    for i, (n, _p) in enumerate(net.named_parameters()):
        if n == name:
            return i
    raise KeyError(name)


def migrate(ckpt, target, verbose=True):
    """Widen every card-id axis in-place. Returns (old_width, changed)."""
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    old = sd["card_id_embed.weight"].shape[1]
    if old == target:
        if verbose:
            print(f"  already {target} card ids -- nothing to do.")
        return old, False
    if old > target:
        raise ValueError(f"cannot shrink {old} card ids to {target}: this script only widens.")

    for k in INPUT_COLUMNS:
        before = sd[k].clone()
        sd[k] = _pad_cols(sd[k], target)
        assert torch.equal(sd[k][:, :old], before) and torch.count_nonzero(sd[k][:, old:]) == 0
        if verbose:
            print(f"  {k}  {tuple(before.shape)} -> {tuple(sd[k].shape)}  (new columns zero)")
    for k in OUTPUT_ROWS:
        before = sd[k].clone()
        sd[k] = _pad_rows_copy_last(sd[k], target)
        assert torch.equal(sd[k][:old], before)
        if verbose:
            print(f"  {k}  {tuple(before.shape)} -> {tuple(sd[k].shape)}  (new rows copy id {old - 1})")

    fresh = MicroRoyaleNet(num_ability_slots=gym_wrapper.DEFAULT_DECK_ABILITY_SLOTS)
    for k in DERIVED_BUFFERS:
        if k in sd:
            sd[k] = fresh.state_dict()[k].clone()
            if verbose:
                print(f"  {k}  rebuilt from the engine ({tuple(sd[k].shape)})")

    opt = ckpt.get("optimizer") if isinstance(ckpt, dict) else None
    if opt is None:
        if verbose:
            print("  no optimizer state in this file (inference-only checkpoint)")
        return old, True
    state = opt.get("state", {})
    for k in INPUT_COLUMNS + OUTPUT_ROWS:
        idx = _param_index(fresh, k)
        entry = state.get(idx, state.get(str(idx)))
        if entry is None:
            continue
        for moment in ("exp_avg", "exp_avg_sq"):
            if moment in entry and torch.is_tensor(entry[moment]):
                t = entry[moment]
                entry[moment] = _pad_cols(t, target) if k in INPUT_COLUMNS else _pad_rows_zero(t, target)
                if verbose:
                    print(f"  optimizer[{idx}].{moment}  {tuple(t.shape)} -> {tuple(entry[moment].shape)}")
    return old, True


def _old_to_new_scalars(x_old, hand_size, old_ids, new_ids, num_extra):
    """Re-lay an old-layout scalar tail out for the new width: zeros for the new ids."""
    b = x_old.shape[0]
    parts = [x_old[:, : 1 + hand_size]]
    pos = 1 + hand_size
    for _ in range(hand_size):
        parts.append(torch.cat([x_old[:, pos:pos + old_ids], torch.zeros(b, new_ids - old_ids)], dim=1))
        pos += old_ids
    parts.append(x_old[:, pos:pos + num_extra])
    pos += num_extra
    for _ in range(2):
        parts.append(torch.cat([x_old[:, pos:pos + old_ids], torch.zeros(b, new_ids - old_ids)], dim=1))
        pos += old_ids
    return torch.cat(parts, dim=1)


def verify(old_ckpt, new_ckpt, verbose=True):
    """Prove the migration is inert on everything the old net could see.

    Each card-id-dependent piece is driven, old against new: the scalar encoder
    on a re-laid-out observation tail, the card embedding on an old id, and both
    heads' logits on the old ids. Then the migrated file must load strictly and
    its optimizer state must load into a fresh Adam.
    """
    sd_old = old_ckpt["model"]
    sd_new = new_ckpt["model"]
    old = sd_old["card_id_embed.weight"].shape[1]
    new = sd_new["card_id_embed.weight"].shape[1]
    slots = gym_wrapper.DEFAULT_DECK_ABILITY_SLOTS

    net = MicroRoyaleNet(num_ability_slots=slots)
    net.load_state_dict(sd_new, strict=True)
    net.eval()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    if new_ckpt.get("optimizer") is not None:
        opt.load_state_dict(new_ckpt["optimizer"])

    # Old-width scalar encoder, built at the old id count and given the old tensors.
    ref = MicroRoyaleNet(num_card_ids=old, num_ability_slots=slots).scalar_mlp
    ref.load_state_dict({k[len("scalar_mlp."):]: v for k, v in sd_old.items() if k.startswith("scalar_mlp.")})
    ref.eval()
    hand = net.hand_size
    torch.manual_seed(0)
    x_old = torch.rand(8, 1 + hand + hand * old + net.num_extra_scalars + 2 * old)
    x_new = _old_to_new_scalars(x_old, hand, old, new, net.num_extra_scalars)
    with torch.no_grad():
        d_enc = (ref(x_old) - net.scalar_mlp(x_new)).abs().max().item()
        onehot_old = torch.eye(old)[:7]
        onehot_new = torch.cat([onehot_old, torch.zeros(7, new - old)], dim=1)
        d_emb = (onehot_old @ sd_old["card_id_embed.weight"].T - net.card_id_embed(onehot_new)).abs().max().item()
        h = torch.randn(5, sd_old["aux_card_head.weight"].shape[1])
        d_aux = ((h @ sd_old["aux_card_head.weight"].T + sd_old["aux_card_head.bias"])
                 - net.aux_card_head(h)[:, :old]).abs().max().item()
        c = torch.randn(5, sd_old["cycle_id_head.weight"].shape[1])
        d_cyc = ((c @ sd_old["cycle_id_head.weight"].T + sd_old["cycle_id_head.bias"])
                 - net.cycle_id_head(c)[:, :old]).abs().max().item()
    # The heads' old-id rows must be the very same numbers; their logits then
    # agree to float32 round-off only, because a 256-row matmul blocks its
    # arithmetic differently from a 185-row one.
    rows_equal = all(torch.equal(sd_old[k], sd_new[k][:old]) for k in OUTPUT_ROWS)
    if verbose:
        print(f"  max |old - new|: scalar encoder {d_enc:.3e}, card embedding {d_emb:.3e} (must be 0); "
              f"aux head {d_aux:.3e}, cycle head {d_cyc:.3e} (round-off); old-id head rows identical: {rows_equal}")
    if d_enc != 0.0 or d_emb != 0.0 or not rows_equal or max(d_aux, d_cyc) > 1e-5:
        raise SystemExit("!! the migration is not inert on old ids")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, help="checkpoint to read (relative to python_ai/)")
    ap.add_argument("--out", dest="dst", default=None, help="where to write (default: <in stem>.ids<N>.pth)")
    ap.add_argument("--force", action="store_true", help="allow overwriting an existing --out")
    a = ap.parse_args()

    src = a.src if os.path.isabs(a.src) else os.path.join(python_ai.PACKAGE_DIR, a.src)
    if not os.path.exists(src):
        raise SystemExit(f"!! no such checkpoint: {src}")
    target = clash_royale_env.ClashRoyaleEnv.NUM_CARD_IDS
    if a.dst is None:
        stem, ext = os.path.splitext(src)
        dst = f"{stem}.ids{target}{ext}"
    else:
        dst = a.dst if os.path.isabs(a.dst) else os.path.join(python_ai.PACKAGE_DIR, a.dst)
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("!! --out must differ from --in, so the original stays resumable.")
    if os.path.exists(dst) and not a.force:
        raise SystemExit(f"!! {dst} exists; pass --force to overwrite.")

    original = torch.load(src, map_location="cpu", weights_only=False)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    print(f"migrating {src} to {target} card ids")
    old, changed = migrate(ckpt, target)
    if old == target:
        raise SystemExit("the engine still has the checkpoint's id count: rebuild the .pyd first")
    if not changed:
        return
    atomic_save(ckpt, dst)
    print(f"wrote {dst}")
    verify(original, torch.load(dst, map_location="cpu", weights_only=False))
    print("verified: encoder and card embedding bit-identical on old ids, heads equal to float32 round-off.")


if __name__ == "__main__":
    main()
