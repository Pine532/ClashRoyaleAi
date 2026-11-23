"""The live encoder and NeuralPolicy with the opponent-tracker tail.

An OLD checkpoint (no tracker keys) must behave exactly as before: the engine
width, no adapter. A TRACKER checkpoint gets the adapter's tail appended. A
tracker net is never fed a silent zero tail: the only zero tail is the explicit
`tracker_tail="zero"` (for checkpoints trained under CLASH_OPP_TRACKER=zero).

Needs the perception venv (scipy, for mvp_loop's detector imports).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
torch = pytest.importorskip("torch")

from contracts import GameState, Phase, TowerObservation  # noqa: E402

FULL = TowerObservation(hp_fraction=1.0, hp_measured=True)


@pytest.fixture(scope="module")
def stack(engine):
    from live import mvp_loop
    from python_ai.models import perception_encoder as enc
    from python_ai.models.net import MicroRoyaleNet
    from python_ai.tracking import opponent_tracker as ot
    return mvp_loop, enc, MicroRoyaleNet, ot


def _state(hand=(15, 6, 25, 40), seconds=30.0, elixir=7.0):
    return GameState(
        units=(), my_elixir=elixir, my_hand=tuple(hand), seconds_elapsed=seconds,
        phase=Phase.SINGLE, own_king=FULL, own_princess_left=FULL,
        own_princess_right=FULL, opp_king=FULL, opp_princess_left=FULL,
        opp_princess_right=FULL)


def _checkpoints(stack, tmp_path):
    """(old, tracker) checkpoints with the SAME function: the tracker net is
    the old weights plus a branch whose last layer is zero."""
    _, enc, Net, ot = stack
    torch.manual_seed(7)
    old = Net(num_ability_slots=0)
    old_path = tmp_path / "old.pth"
    torch.save({"model": old.state_dict(), "episodes_completed": 5}, old_path)
    new = Net(num_ability_slots=0, tracker_size=enc.TRACKER_TAIL_SIZE)
    new.load_state_dict(old.state_dict(), strict=False)
    new.reset_tracker_branch_to_init()
    new_path = tmp_path / "tracker.pth"
    torch.save({"model": new.state_dict(), "episodes_completed": 5}, new_path)
    return old_path, new_path


class _Spy:
    """Records the observation widths extract_features is called with."""

    def __init__(self, policy):
        self.obs = []
        real = policy.net.extract_features

        def wrapped(obs, *a, **k):
            self.obs.append(obs.detach().clone())
            return real(obs, *a, **k)
        policy.net.extract_features = wrapped


def test_encode_appends_the_tail_and_leaves_the_engine_part_alone(stack):
    _, enc, _, ot = stack
    base = enc.encode(_state())
    assert base.shape == (enc.OBSERVATION_SIZE,)
    tail = np.arange(enc.TRACKER_TAIL_SIZE, dtype=np.float32)
    full = enc.encode(_state(), tail=tail)
    assert full.shape == (enc.OBSERVATION_SIZE + enc.TRACKER_TAIL_SIZE,)
    assert np.array_equal(full[:enc.OBSERVATION_SIZE], base)
    assert np.array_equal(full[enc.OBSERVATION_SIZE:], tail)
    assert enc.TRACKER_TAIL_SIZE == ot.feature_width(enc.NUM_CARD_IDS)
    with pytest.raises(ValueError):
        enc.encode(_state(), tail=tail[:-1])


def test_old_checkpoint_is_unchanged(stack, tmp_path):
    mvp_loop, enc, _, _ = stack
    old_path, _ = _checkpoints(stack, tmp_path)
    p = mvp_loop.NeuralPolicy(old_path, tactical=False)
    assert p.net.tracker_size == 0 and p.tracker_adapter is None
    spy = _Spy(p)
    torch.manual_seed(3)
    p.decide(_state(), [True] * 4, 0.0)
    assert spy.obs[0].shape[-1] == enc.OBSERVATION_SIZE
    assert np.array_equal(spy.obs[0][0].numpy(), enc.encode(_state()))


def test_tracker_checkpoint_gets_the_tail_and_the_same_actions_as_old(stack, tmp_path):
    """The migrated branch adds exactly 0, so for any tail the decisions equal
    the old checkpoint's under the same sampling seed."""
    mvp_loop, enc, _, _ = stack
    old_path, new_path = _checkpoints(stack, tmp_path)
    old = mvp_loop.NeuralPolicy(old_path, tactical=False)
    new = mvp_loop.NeuralPolicy(new_path, tactical=False)
    assert new.net.tracker_size == enc.TRACKER_TAIL_SIZE and new.tracker_adapter is not None
    spy = _Spy(new)
    new.tracker_adapter.on_opponent_play(25, 15)      # a non-trivial tail
    new.tracker_adapter.on_own_play(26, 6)
    new.observe_own_next(40)
    seeds = range(8)
    for i, seed in enumerate(seeds):
        gs = _state(seconds=30.0 + i)
        torch.manual_seed(seed)
        a = old.decide(gs, [True] * 4, float(i))
        torch.manual_seed(seed)
        b = new.decide(gs, [True] * 4, float(i))
        assert (a.slot, tuple(a.tile)) == (b.slot, tuple(b.tile)), seed
    o = spy.obs[0][0].numpy()
    assert o.shape == (enc.OBSERVATION_SIZE + enc.TRACKER_TAIL_SIZE,)
    tail = o[enc.OBSERVATION_SIZE:]
    assert tail.any(), "control: the tail must carry the tracker's state"
    from track.tracker_adapter import TrackerAdapter
    ref = TrackerAdapter("real")                      # the same events, fresh
    ref.on_opponent_play(25, 15)
    ref.on_own_play(26, 6)
    ref.set_own_next(40)
    assert np.array_equal(tail, ref.features(300))    # first frame: 30 s = tick 300
    # the own next card is in the tail (own_next one-hot, section 2)
    assert tail[2 * enc.NUM_CARD_IDS + 40] == 1.0


def test_tracker_net_without_a_tail_source_is_an_error_not_zeros(stack, tmp_path):
    mvp_loop, enc, _, _ = stack
    _, new_path = _checkpoints(stack, tmp_path)
    with pytest.raises(ValueError):
        mvp_loop.NeuralPolicy(new_path, tactical=False, tracker_tail="nonsense")
    # the encoder itself refuses to build a tail-less observation for a tail net
    with pytest.raises(ValueError):
        enc.require_tail(enc.encode(_state()), expects_tail=True)


def test_explicit_zero_tail_is_zero(stack, tmp_path):
    mvp_loop, enc, _, _ = stack
    _, new_path = _checkpoints(stack, tmp_path)
    p = mvp_loop.NeuralPolicy(new_path, tactical=False, tracker_tail="zero")
    assert p.tracker_adapter is None
    spy = _Spy(p)
    p.decide(_state(), [True] * 4, 0.0)
    assert spy.obs[0].shape[-1] == enc.OBSERVATION_SIZE + enc.TRACKER_TAIL_SIZE
    assert not spy.obs[0][0, enc.OBSERVATION_SIZE:].any()


def test_tail_size_is_inferred_from_keys_and_checked(stack, tmp_path):
    mvp_loop, enc, Net, _ = stack
    bad = Net(num_ability_slots=0, tracker_size=enc.TRACKER_TAIL_SIZE + 3)
    path = tmp_path / "bad.pth"
    torch.save({"model": bad.state_dict()}, path)
    with pytest.raises(RuntimeError, match="tail"):
        mvp_loop.NeuralPolicy(path, tactical=False)
    # a partial branch is not silently completed
    _, new_path = _checkpoints(stack, tmp_path)
    blob = torch.load(new_path, weights_only=False)
    blob["model"] = {k: v for k, v in blob["model"].items()
                     if k != "tracker_branch.2.bias"}
    torch.save(blob, tmp_path / "partial.pth")
    with pytest.raises(RuntimeError, match="does not match"):
        mvp_loop.NeuralPolicy(tmp_path / "partial.pth", tactical=False)


def test_a_new_match_resets_the_adapter(stack, tmp_path):
    mvp_loop, _, _, _ = stack
    _, new_path = _checkpoints(stack, tmp_path)
    p = mvp_loop.NeuralPolicy(new_path, tactical=False)
    p.on_screen_change(True)
    p.tracker_adapter.on_opponent_play(10, 15)
    p.on_screen_change(False)
    p.on_screen_change(True)
    assert p.tracker_adapter.tracker.seen_order() == []
