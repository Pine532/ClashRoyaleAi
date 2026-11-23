"""eval/tracker_usage_probe: does the policy's output depend on the tracker tail?

Control: a migrated (zero-init) branch adds exactly 0, so swapping the tail for
another game's changes nothing -- distance exactly 0. Once the branch's last
layer is nonzero the distance is > 0.
"""
import numpy as np
import pytest
import torch

import clash_royale_env as E

from python_ai.eval import tracker_usage_probe as probe
from python_ai.models.net import MicroRoyaleNet
from python_ai.tracking import config as tcfg

HOG = [15, 6, 25, 40, 24, 72, 33, 7]


@pytest.fixture(scope="module")
def recorded():
    net = MicroRoyaleNet(num_ability_slots=0, tracker_size=tcfg.tail_width("on"))
    net.eval()
    obs, gids = probe.record_observations(net, HOG, games=3, every=3, max_ticks=600, seed=11)
    return obs, gids


def fresh_net():
    torch.manual_seed(0)
    net = MicroRoyaleNet(num_ability_slots=0, tracker_size=tcfg.tail_width("on"))
    net.eval()
    return net


def test_recorded_observations_carry_a_tail_from_several_games(recorded):
    obs, gids = recorded
    net = fresh_net()
    assert obs.shape[1] == net.obs_size and len(set(gids.tolist())) == 3 and len(obs) >= 30
    assert obs[:, net.tracker_start:].any()


def test_shuffle_takes_the_tail_from_a_different_game(recorded):
    obs, gids = recorded
    perm = probe.other_game_permutation(gids, seed=1)
    assert (gids[perm] != gids).all()


def test_zero_init_branch_gives_exactly_zero_distance(recorded):
    obs, gids = recorded
    net = fresh_net()
    r = probe.tail_usage(net, obs, gids)
    assert r["card_tv_mean"] == 0.0 and r["placement_tv_mean"] == 0.0
    assert r["card_argmax_changed"] == 0.0 and r["placement_argmax_changed"] == 0.0
    # the control only means something if the tails really did differ
    assert r["tails_changed_share"] > 0.5


def test_a_nonzero_last_layer_gives_a_positive_distance(recorded):
    obs, gids = recorded
    net = fresh_net()
    with torch.no_grad():
        net.tracker_branch[-1].weight.normal_(0, 1.0)
        net.tracker_branch[-1].bias.normal_(0, 1.0)
    r = probe.tail_usage(net, obs, gids)
    assert r["card_tv_mean"] > 0.0 and r["placement_tv_mean"] > 0.0
