"""migrate_checkpoint_card_ids widens every card-id axis without changing what
the net computes on the ids it already knew (CARD_ID_COUNT 185 -> 256).

The old-layout checkpoint is built here by cutting a fresh net's tensors back
to 185 ids, so the test needs no old .pyd and writes nothing outside tmp.
"""
import copy

import torch

import python_ai  # noqa: F401
import clash_royale_env
from python_ai.envs import gym_wrapper
from python_ai.models.net import MicroRoyaleNet
from python_ai.tools import migrate_checkpoint_card_ids as M

OLD = 185


def _old_layout_checkpoint():
    torch.manual_seed(1)
    net = MicroRoyaleNet(num_ability_slots=gym_wrapper.DEFAULT_DECK_ABILITY_SLOTS)
    sd = {}
    for k, v in net.state_dict().items():
        if k in M.INPUT_COLUMNS:
            sd[k] = v[:, :OLD].clone()
        elif k in M.OUTPUT_ROWS or k in M.DERIVED_BUFFERS:
            sd[k] = v[:OLD].clone()
        else:
            sd[k] = v.clone()
    return {"model": sd}


def test_migration_widens_every_card_id_axis_to_the_engine_width():
    new = clash_royale_env.ClashRoyaleEnv.NUM_CARD_IDS
    assert new > OLD
    ckpt = _old_layout_checkpoint()
    migrated = copy.deepcopy(ckpt)
    old, changed = M.migrate(migrated, new, verbose=False)
    assert (old, changed) == (OLD, True)
    for k in M.INPUT_COLUMNS:
        assert migrated["model"][k].shape[1] == new
        assert torch.count_nonzero(migrated["model"][k][:, OLD:]) == 0
    for k in M.OUTPUT_ROWS:
        assert migrated["model"][k].shape[0] == new
        assert torch.equal(migrated["model"][k][OLD:], ckpt["model"][k][OLD - 1].expand_as(migrated["model"][k][OLD:]))


def test_migrated_net_is_inert_on_the_old_ids():
    new = clash_royale_env.ClashRoyaleEnv.NUM_CARD_IDS
    ckpt = _old_layout_checkpoint()
    migrated = copy.deepcopy(ckpt)
    M.migrate(migrated, new, verbose=False)
    assert M.verify(ckpt, migrated, verbose=False)


def test_a_checkpoint_already_at_the_engine_width_is_left_alone():
    net = MicroRoyaleNet(num_ability_slots=gym_wrapper.DEFAULT_DECK_ABILITY_SLOTS)
    ckpt = {"model": net.state_dict()}
    old, changed = M.migrate(ckpt, clash_royale_env.ClashRoyaleEnv.NUM_CARD_IDS, verbose=False)
    assert not changed
