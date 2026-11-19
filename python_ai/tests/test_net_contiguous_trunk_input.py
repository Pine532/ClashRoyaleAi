"""The CNN trunk is handed a contiguous board, not a strided view of the observation.

The board is a slice of each flat observation row, so `obs[:, :SPATIAL_SIZE].view(...)` has a
row stride that skips the scalar tail. torch 2.13's MPS `Conv2d` mis-computes such an input
(measured 2026-11-19: off by 1.13 on a synthetic conv, and a live checkpoint's critic loss read
4.58 on MPS against 0.220 on CPU on the same batch). `.contiguous()` fixes it and changes
nothing on CPU, where the values are copied as they are.
"""
import pytest
import torch

from python_ai.models.net import MicroRoyaleNet


def _real_obs(n=4):
    import numpy as np
    import clash_royale_env
    deck = [15, 6, 25, 40, 24, 72, 33, 7]
    rows = []
    for i in range(n):
        env = clash_royale_env.ClashRoyaleEnv(deck, deck, 3600)
        env.seed(i)
        for k in range(30 + 5 * i):
            r = env.step((k % 9) if k % 5 == 0 else 4, 4.0 + k % 10, 20.0 + k % 8)
        rows.append(np.asarray(r.observation, dtype=np.float32))
    return torch.from_numpy(np.stack(rows))


def test_extract_features_hands_the_trunk_a_contiguous_board():
    net = MicroRoyaleNet(num_ability_slots=0).eval()
    seen = []
    net.cnn_trunk[0].register_forward_pre_hook(
        lambda mod, args: seen.append(args[0].is_contiguous()))
    with torch.no_grad():
        net.extract_features(_real_obs())
    assert seen == [True]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_extract_features_on_mps_matches_cpu():
    torch.manual_seed(0)
    net = MicroRoyaleNet(num_ability_slots=0).eval()
    obs = _real_obs()
    with torch.no_grad():
        ref = net.extract_features(obs)[0]
        got = net.to("mps").extract_features(obs.to("mps"))[0].cpu()
    assert (ref - got).abs().max().item() < 1e-4


def test_hires_features_hands_the_trunk_a_contiguous_board():
    # The same slice as extract_features_hires; search/search.py calls it.
    net = MicroRoyaleNet(num_ability_slots=0).eval()
    seen = []
    net.cnn_trunk[0].register_forward_pre_hook(
        lambda mod, args: seen.append(args[0].is_contiguous()))
    with torch.no_grad():
        net.hires_features(_real_obs())
    assert seen == [True]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_hires_features_on_mps_matches_cpu():
    torch.manual_seed(0)
    net = MicroRoyaleNet(num_ability_slots=0).eval()
    obs = _real_obs()
    with torch.no_grad():
        ref = net.hires_features(obs)
        got = net.to("mps").hires_features(obs.to("mps")).cpu()
    assert (ref - got).abs().max().item() < 1e-4
