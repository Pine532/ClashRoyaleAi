"""The card mask follows the engine's post-cycle hand lock (rl/hand_slots.py).

After a card is played, `PlayerState::playCard` refuses the slot it cycled into
for 20 ticks. The observation does not carry that, so the mask used to call the
slot legal: measured 2026-11-18, 10-16% of a rung-0 policy's card attempts were
silently refused, on main too. These tests pin:
- the env reports the lock for exactly as long as the engine refuses, at several
  decision spacings (judged by elixir actually spent, not by the reported flag);
- the mask drops a locked slot and keeps the no-op;
- the update applies the mask the rollout used. Without the stored readiness the
  PPO ratio drifts, which is the control that must fire.
"""
import copy
import math

import numpy as np
import pytest
import torch
import torch.optim as optim

clash_royale_env = pytest.importorskip("clash_royale_env")

from python_ai.envs import gym_wrapper  # noqa: E402
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import LSTM_HIDDEN  # noqa: E402
from python_ai.rl import hand_slots  # noqa: E402
from python_ai.rl.buffer import CORE_FIELDS, HAND_FIELDS, RolloutBuffer  # noqa: E402
from python_ai.rl.config import PPOConfig  # noqa: E402
from python_ai.rl.ppo import PPOUpdater  # noqa: E402

DECK = [15, 6, 25, 40, 24, 72, 33, 7]
HAND = [24, 72, 6, 15]  # Skeletons, Ice Spirit, Musketeer, Hog Rider
HAND_SIZE = clash_royale_env.ClashRoyaleEnv.HAND_SIZE


def _play(env, slot, x, y, skip):
    action = {"card_index": slot, "target_x": [x], "target_y": [y]}
    return env.step(action, skip_frames=skip)


@pytest.mark.parametrize("skip", [10, 5, 1])
def test_a_played_slot_reads_not_ready_exactly_while_the_engine_refuses_it(skip):
    env = gym_wrapper.MicroRoyaleEnv({"opponent": "builtin"})
    env.reset(seed=3)
    g = env.game
    assert g.set_hand_for_team(0, HAND)
    x, y = 9.0, g.get_own_half_max_y() - 3.0

    g.set_elixir_for_team(0, 10.0)
    *_, info = _play(env, 0, x, y, skip)
    spent = g.get_elixir_spent(0)
    assert spent == pytest.approx(1.0)  # Skeletons went down
    assert all(info[hand_slots.INFO_KEY.format(i)] for i in range(1, HAND_SIZE))

    locked = 0
    while not info[hand_slots.INFO_KEY.format(0)]:
        g.set_elixir_for_team(0, 10.0)
        *_, info = _play(env, 0, x, y, skip)
        # The env said "not ready" and the engine agrees: nothing was spent.
        assert g.get_elixir_spent(0) == pytest.approx(spent)
        locked += 1
        assert locked < 40
    # 20 ticks of lock, counted down `skip` per decision, after the play's own step.
    assert locked == math.ceil(20 / skip) - 1

    # The env said "ready" and the engine agrees: the play goes through.
    g.set_elixir_for_team(0, 10.0)
    _play(env, 0, x, y, skip)
    assert g.get_elixir_spent(0) > spent


def test_the_card_mask_drops_a_locked_slot_and_keeps_the_noop():
    torch.manual_seed(0)
    net = MicroRoyaleNet(num_ability_slots=0)
    env = clash_royale_env.ClashRoyaleEnv(DECK, DECK, 3600)
    assert env.set_hand_for_team(0, HAND)
    env.set_elixir_for_team(0, 10.0)
    obs = torch.tensor(np.asarray([env.get_observation_for_team(0)], dtype=np.float32))
    assert net.affordability_mask(obs)[0].tolist() == [True] * (HAND_SIZE + 1)
    locked = torch.tensor([[False, True, True, True]])
    assert net.affordability_mask(obs, locked)[0].tolist() == [False, True, True, True, True]
    # Ready does not make an unaffordable card legal.
    env.set_elixir_for_team(0, 0.0)
    obs0 = torch.tensor(np.asarray([env.get_observation_for_team(0)], dtype=np.float32))
    all_ready = torch.ones((1, HAND_SIZE), dtype=torch.bool)
    assert net.affordability_mask(obs0, all_ready)[0].tolist() == [False] * HAND_SIZE + [True]


def test_missing_readiness_reads_ready_and_present_flags_are_kept():
    infos = {"hand_slot0_ready": np.array([False, True]),
             "_hand_slot0_ready": np.array([True, False])}
    r = hand_slots.ready_from_infos(infos, 2, HAND_SIZE)
    # env 0 reported slot 0 locked; env 1 did not report it (phantom step), so ready.
    assert r.tolist() == [[False, True, True, True], [True, True, True, True]]


def _locked_rollout(cfg, seed=11):
    """A real-engine rollout sampled under a mask that locks slot 0 whenever it
    is affordable, with that readiness stored like the trainer stores it."""
    torch.manual_seed(seed)
    net = MicroRoyaleNet(num_ability_slots=0)
    envs = [clash_royale_env.ClashRoyaleEnv(DECK, DECK, 3600) for _ in range(cfg.num_envs)]
    for e in envs:
        e.seed(7)
        e.reset()
    buf = RolloutBuffer(tuple(CORE_FIELDS) + tuple(HAND_FIELDS))
    hx = torch.zeros(cfg.num_envs, LSTM_HIDDEN)
    cx = torch.zeros(cfg.num_envs, LSTM_HIDDEN)
    n_locked = 0
    for _ in range(cfg.update_timestep):
        for e in envs:
            e.set_elixir_for_team(0, 10.0)
        obs = torch.tensor(np.asarray([e.get_observation_for_team(0) for e in envs],
                                      dtype=np.float32))
        ready = torch.ones((cfg.num_envs, HAND_SIZE), dtype=torch.bool)
        ready[:, 0] = False
        mask = net.affordability_mask(obs, ready)
        n_locked += int(net.affordability_mask(obs)[:, 0].sum())
        with torch.no_grad():
            feats, embeds, spatial = net.extract_features(obs)
            logits, _, _, value, (hx2, cx2) = net.step_lstm_and_card(feats, (hx, cx), mask)
            card = torch.distributions.Categorical(logits=logits).sample()
            pl = net.placement_given_card(hx2, embeds, card, obs, spatial)
            cell = torch.distributions.Categorical(logits=pl).sample()
            lp = (torch.distributions.Categorical(logits=logits).log_prob(card)
                  + torch.distributions.Categorical(logits=pl).log_prob(cell))
        buf.add(obs=obs, card_actions=card, placement_actions=cell,
                decision=(mask.sum(dim=1) > 1).float(), hx_in=hx, cx_in=cx,
                logprobs=lp, values=value.squeeze(-1),
                rewards=torch.zeros(cfg.num_envs), masks=torch.ones(cfg.num_envs),
                valid=torch.ones(cfg.num_envs),
                aux_opp_played=torch.full((cfg.num_envs,), -1, dtype=torch.long),
                coverage_slot=torch.zeros(cfg.num_envs, dtype=torch.long),
                hand_ready=ready)
        hx, cx = hx2, cx2
        for i, e in enumerate(envs):
            e.step(int(card[i]), 0.0, 0.0, 10)
    assert n_locked > 0  # the case under test occurred
    return net, buf.stack()


def _first_minibatch_ratio_dev(net, batch, cfg):
    torch.manual_seed(1234)
    np.random.seed(1234)
    net = copy.deepcopy(net)
    updater = PPOUpdater(net, optim.Adam(net.parameters(), lr=1e-4), cfg)
    T, N = batch["rewards"].shape
    stats = updater.update(batch, torch.randn(T, N), torch.zeros(T, N),
                           vf_clip_range=0.2, ent_coef_card=0.05,
                           ent_coef_placement=0.06, coverage_coef=0.0)
    return stats.ratio_dev_first


def test_the_update_applies_the_hand_mask_the_rollout_used():
    cfg = PPOConfig(num_envs=2, update_timestep=8, bptt_chunk=2,
                    num_minibatches=1, ppo_epochs=1)
    net, batch = _locked_rollout(cfg)
    # Same mask at update time: the policy is unchanged before the first step,
    # so the ratio is 1 up to float noise.
    assert _first_minibatch_ratio_dev(net, batch, cfg) < 1e-4
    # Control that must fire: drop the stored readiness and the update scores
    # the actions under a different mask.
    without = {k: v for k, v in batch.items() if k != "hand_ready"}
    assert _first_minibatch_ratio_dev(net, without, cfg) > 1e-2
