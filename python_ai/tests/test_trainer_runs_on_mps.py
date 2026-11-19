"""Training runs on MPS end to end: no tensor built on the CPU meets one on the device.

The network matched CPU on MPS once its trunk input was contiguous, but the trainer still
crashed (2026-11-19) where advisor and rollout code built tensors on the CPU: the legal
table read with `.numpy()`, the coverage slot weights, the coverage slot ids and the
advisor targets stored for the update. Skipped where MPS is unavailable.
"""
import pytest
import torch

from python_ai.advisors import advisor_target
from python_ai.models.net import MicroRoyaleNet
from python_ai.tests.test_rl_base_trainer import (  # noqa: F401  (fixtures)
    _never_touch_the_live_checkpoint, _phase1, workdir)

needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")


@needs_mps
def test_the_legal_table_reads_off_a_net_on_mps():
    net = MicroRoyaleNet(num_ability_slots=0)
    want = advisor_target.build_legal_table(net)
    got = advisor_target.build_legal_table(net.to("mps"))
    assert want.keys() == got.keys()
    assert all((want[c] == got[c]).all() for c in want)


@needs_mps
def test_slot_weights_live_on_the_hand_ids_device():
    cid = sorted(advisor_target.target_cards())[0]
    hand = torch.tensor([[cid, -1, -1, -1]], device="mps")
    w = advisor_target.slot_weights_for(hand)
    assert w is not None, "the weighting is off by default; this test needs it on"
    assert w.device.type == "mps"
    assert w.cpu().tolist()[0][0] == advisor_target.ADVISOR_SLOT_WEIGHT


@needs_mps
def test_a_phase1_update_runs_on_mps(workdir, monkeypatch):
    from python_ai.rl import base_trainer
    seen = []
    real = base_trainer.BaseTrainer.log_update
    monkeypatch.setattr(base_trainer.BaseTrainer, "log_update",
                        lambda self, stats: (seen.append(stats), real(self, stats))[1])
    monkeypatch.setenv("CLASH_DEVICE", "mps")
    trainer = _phase1()
    assert trainer.device.type == "mps"
    assert seen, "no update ran"
    # The rollout sampled on MPS and the update recomputed on MPS: they must agree.
    assert seen[0].ratio_dev_first < 1e-3
    assert seen[0].nonfinite_skips == 0
