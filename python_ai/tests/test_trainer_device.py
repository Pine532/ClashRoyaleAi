"""CLASH_DEVICE picks the device the trainer runs the network on.

Unset keeps the old rule (CUDA when present, else CPU). MPS is opt-in: it was
measured to agree with CPU on a live batch only after the trunk's input was made
contiguous (TODO L11), so it runs only when asked for. An explicit device that
is not available is an error, never a silent fallback to CPU.
"""
import pytest
import torch

from python_ai.rl import base_trainer
from python_ai.rl.checkpointing import OPERATIONAL_SETTINGS


def test_unset_keeps_the_old_rule():
    want = "cuda" if torch.cuda.is_available() else "cpu"
    assert base_trainer.resolve_device({}).type == want
    assert base_trainer.resolve_device({"CLASH_DEVICE": "auto"}).type == want


def test_cpu_by_name():
    assert base_trainer.resolve_device({"CLASH_DEVICE": "cpu"}).type == "cpu"
    assert base_trainer.resolve_device({"CLASH_DEVICE": " CPU "}).type == "cpu"


def test_an_unknown_device_is_refused():
    with pytest.raises(ValueError):
        base_trainer.resolve_device({"CLASH_DEVICE": "gpu"})


def test_an_unavailable_device_is_an_error_not_a_cpu_run(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        base_trainer.resolve_device({"CLASH_DEVICE": "mps"})


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_mps_by_name():
    assert base_trainer.resolve_device({"CLASH_DEVICE": "mps"}).type == "mps"


def test_the_trainer_reads_it(monkeypatch):
    # With CUDA "present", the old rule would pick it; the setting must win.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("CLASH_DEVICE", "cpu")
    assert base_trainer.BaseTrainer().device.type == "cpu"


def test_the_device_is_operational_not_a_settings_drift():
    # It moves the numbers by float32 roundoff, not what the run learns.
    assert "CLASH_DEVICE" in OPERATIONAL_SETTINGS
