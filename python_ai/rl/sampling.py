"""Drawing a rollout action from a categorical distribution, on the CPU.

torch 2.13's MPS sampler (`Categorical.sample()`, i.e. `torch.multinomial` on
an MPS tensor) occasionally returns a category whose probability is exactly 0.
Measured 2026-11-19 (docs/TODO.md L11): 186 in 20,000,000 draws of a live
placement distribution, and about 1 in 10^5 for any masked distribution over
612 cells; about 1 in 10^7 on the 5-arm card head and the 2-arm ability heads.
The CPU drew none. A masked (illegal) action then reaches the buffer with
log-prob -inf, and the update's ratio for that row is NaN, which drops the
whole minibatch.

So every rollout draw is made on the CPU, whatever device the network runs on:
the card, placement and ability actions (`sample_categorical`) and the
coverage slot (`sample_multinomial`; torch.multinomial on MPS drew a
zero-weight entry 17 times in 2,000,000 over 612 weights). On a CPU tensor each
is exactly the torch call it replaces: the same samples and the same generator
state, so CPU training is unchanged.
"""
import torch
from torch.distributions import Categorical


def sample_categorical(logits):
    """One index per row of `logits` (..., n), drawn on the CPU and returned on
    `logits`' device."""
    if logits.device.type == "cpu":
        return Categorical(logits=logits).sample()
    return Categorical(logits=logits.detach().cpu()).sample().to(logits.device)


def sample_multinomial(probs, num_samples=1):
    """`torch.multinomial(probs, num_samples)` (rows of non-negative weights),
    drawn on the CPU and returned on `probs`' device."""
    if probs.device.type == "cpu":
        return torch.multinomial(probs, num_samples)
    return torch.multinomial(probs.detach().cpu(), num_samples).to(probs.device)
