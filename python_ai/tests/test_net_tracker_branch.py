"""The tracker tail's network branch is exact at init, learnable, and cannot
carry stale weights between checkpoints.

Design: docs/ROADMAP_TOP_PLAYERS.md 5.4-5.5. The branch is
Linear(tracker_size, 32) -> ReLU -> Linear(32, 40) with the last layer ZERO,
added to the first 40 of the 64 scalar features. An observation of exactly the
engine width carries no tail and skips the branch.
"""
import copy

import pytest
import torch

import clash_royale_env
from python_ai.envs import gym_wrapper
from python_ai.models.net import MicroRoyaleNet
from python_ai.models.policy_io import LSTM_HIDDEN, load_state_dict_flexible

CE = clash_royale_env.ClashRoyaleEnv
TRACKER = 24  # any width; the real one is the tracker module's


def _real_obs(n=6):
    deck = list(gym_wrapper.DEFAULT_DECK)
    env = CE(deck, deck, 3600)
    env.reset()
    rows = []
    for i in range(n):
        env.set_current_tick(40 * i)
        rows.append(torch.tensor(env.get_observation_for_team(i % 2), dtype=torch.float32))
    return torch.stack(rows)


def _pair(slots=2, tracker=TRACKER):
    torch.manual_seed(0)
    old = MicroRoyaleNet(num_ability_slots=slots).eval()
    new = MicroRoyaleNet(num_ability_slots=slots, tracker_size=tracker).eval()
    res = new.load_state_dict(old.state_dict(), strict=False)
    assert not res.unexpected_keys
    assert res.missing_keys and all(k.startswith("tracker_branch.") for k in res.missing_keys)
    return old, new


def _tail(kind, n, width):
    g = torch.Generator().manual_seed(3)
    if kind == "zeros":
        return torch.zeros(n, width)
    if kind == "uniform":
        return torch.rand(n, width, generator=g)
    return torch.randn(n, width, generator=g) * 1e3


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
    feats, embeds, spatial, hires = net.extract_features_hires(obs)
    hidden = (torch.zeros(obs.shape[0], LSTM_HIDDEN), torch.zeros(obs.shape[0], LSTM_HIDDEN))
    mask = net.affordability_mask(obs)
    card, ab1, ab2, value, (hx, cx) = net.step_lstm_and_card(feats, hidden, mask)
    place = net.placement_given_card(hx, embeds, torch.zeros(obs.shape[0], dtype=torch.long),
                                     obs, spatial)
    aux = net.predict_opp_next_card(hx)
    cyc = net.predict_cycle_card(net.cycle_features(obs))
    return [feats, card, ab1, ab2, value, place, aux, cyc, hx, cx]


def _update_outputs(net, obs):
    n = obs.shape[0]
    L, B = 2, n // 2
    obs_seq = obs.view(L, B, -1)
    feats, embeds, spatial = net.extract_features(obs)
    mask = net.affordability_mask(obs).view(L, B, -1)
    idx = torch.zeros(L, B, dtype=torch.long)
    hidden = (torch.zeros(B, LSTM_HIDDEN), torch.zeros(B, LSTM_HIDDEN))
    out = net.forward_sequence(
        feats.view(L, B, -1), embeds.view(L, B, *embeds.shape[1:]),
        spatial.view(L, B, *spatial.shape[1:]), obs_seq, mask, idx,
        torch.ones(L, B), hidden, with_ability=True)
    return _flatten(out)


@pytest.mark.parametrize("kind", ["zeros", "uniform", "huge"])
def test_a_zero_initialised_branch_is_bit_exact_at_any_tail(kind):
    old, new = _pair()
    obs = _real_obs()
    assert obs.shape[1] == old.spatial_size + old.scalar_size
    full = torch.cat([obs, _tail(kind, obs.shape[0], TRACKER)], dim=1)
    assert full.shape[1] == new.obs_size
    with torch.no_grad():
        for fn in (_rollout_outputs, _update_outputs):
            a, b = fn(old, obs), fn(new, full)
            assert len(a) == len(b) and len(a) >= 8
            for x, y in zip(a, b):
                assert torch.equal(x, y), fn.__name__


def test_an_engine_width_observation_skips_the_branch():
    old, new = _pair()
    obs = _real_obs()
    with torch.no_grad():
        for x, y in zip(_rollout_outputs(old, obs), _rollout_outputs(new, obs)):
            assert torch.equal(x, y)


def test_the_guard_names_a_missing_or_wrong_tail():
    _, new = _pair()
    assert new.expects_tail and new.tracker_start == new.spatial_size + new.scalar_size
    obs = _real_obs()
    new.require_tail(torch.cat([obs, torch.zeros(obs.shape[0], TRACKER)], 1))
    with pytest.raises(ValueError, match="tracker"):
        new.require_tail(obs)
    with pytest.raises(ValueError, match="width"):
        new.extract_features(obs[:, :-3])
    plain = MicroRoyaleNet(num_ability_slots=0)
    assert not plain.expects_tail and plain.obs_size == plain.tracker_start


def test_the_branch_learns_while_all_zero_init_would_be_dead():
    _, new = _pair()
    new.train()
    obs = _real_obs()
    tail = torch.rand(obs.shape[0], TRACKER)
    full = torch.cat([obs, tail], 1)
    last = new.tracker_branch[-1]
    first = new.tracker_branch[0]
    assert not last.weight.any() and not last.bias.any()
    opt = torch.optim.Adam(new.parameters(), lr=3e-4)

    def loss_fn():
        feats, *_ = new.extract_features_hires(full)
        return feats.pow(2).sum()

    opt.zero_grad(); loss_fn().backward()
    # Step 1: the first layer gets exactly zero gradient (last layer is zero),
    # but the last layer already has one.
    assert not first.weight.grad.any()
    assert last.weight.grad.abs().max() > 0
    opt.step()
    assert last.weight.abs().max() > 0
    opt.zero_grad(); loss_fn().backward()
    assert first.weight.grad.abs().max() > 0

    # The dead alternative: with BOTH layers zero, the hidden is 0, so the last
    # weight gets no gradient and the first none either; only a bias moves.
    dead = copy.deepcopy(new)
    for p in dead.tracker_branch.parameters():
        torch.nn.init.zeros_(p)
    dead.zero_grad()
    feats, *_ = dead.extract_features_hires(full)
    feats.pow(2).sum().backward()
    assert not dead.tracker_branch[0].weight.grad.any()
    assert not dead.tracker_branch[-1].weight.grad.any()


def test_an_old_state_dict_warm_starts_with_equal_outputs():
    old, new = _pair()
    fresh = MicroRoyaleNet(num_ability_slots=2, tracker_size=TRACKER).eval()
    clean = load_state_dict_flexible(fresh, old.state_dict(), "old layout")
    assert clean is False  # tracker keys are absent
    obs = _real_obs()
    full = torch.cat([obs, _tail("huge", obs.shape[0], TRACKER)], 1)
    with torch.no_grad():
        for x, y in zip(_rollout_outputs(old, obs), _rollout_outputs(fresh, full)):
            assert torch.equal(x, y)


def test_a_stale_branch_is_reset_when_an_old_snapshot_is_loaded():
    old, new = _pair()
    trained = new.state_dict()
    for k in trained:
        if k.startswith("tracker_branch."):
            trained[k] = torch.randn_like(trained[k])
    reused = MicroRoyaleNet(num_ability_slots=2, tracker_size=TRACKER).eval()
    reused.load_state_dict(trained)
    obs = _real_obs()
    full = torch.cat([obs, torch.rand(obs.shape[0], TRACKER)], 1)
    with torch.no_grad():
        assert not torch.equal(_rollout_outputs(reused, full)[0],
                               _rollout_outputs(old, obs)[0])
    load_state_dict_flexible(reused, old.state_dict(), "old snapshot")
    assert not reused.tracker_branch[-1].weight.any()
    assert not reused.tracker_branch[-1].bias.any()
    with torch.no_grad():
        for x, y in zip(_rollout_outputs(old, obs), _rollout_outputs(reused, full)):
            assert torch.equal(x, y)
    # A snapshot that HAS the keys keeps them.
    load_state_dict_flexible(reused, trained, "new snapshot")
    assert reused.tracker_branch[-1].weight.abs().max() > 0


def test_the_historical_opponent_loader_resets_the_reused_net(tmp_path):
    old, _ = _pair()
    path = tmp_path / "old.pth"
    torch.save({"model": old.state_dict()}, path)
    net = MicroRoyaleNet(num_ability_slots=2, tracker_size=TRACKER)
    with torch.no_grad():
        for p in net.tracker_branch.parameters():
            p.fill_(0.5)
    ckpt = torch.load(path, weights_only=False)
    load_state_dict_flexible(net, ckpt["model"], str(path))
    assert not net.tracker_branch[-1].weight.any()


def test_tracker_size_zero_adds_no_parameters_and_no_keys():
    torch.manual_seed(0)
    a = MicroRoyaleNet(num_ability_slots=2)
    b = MicroRoyaleNet(num_ability_slots=2, tracker_size=0)
    assert list(a.state_dict()) == list(b.state_dict())
    assert not any("tracker" in k for k in b.state_dict())
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    c = MicroRoyaleNet(num_ability_slots=2, tracker_size=TRACKER)
    keys = list(c.state_dict())
    assert keys[:len(list(a.state_dict()))] == list(a.state_dict())
    assert all(k.startswith("tracker_branch.") for k in keys[len(list(a.state_dict())):])
