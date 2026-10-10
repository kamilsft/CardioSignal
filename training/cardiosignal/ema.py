"""Exponential moving average (EMA) of model weights.

GAN generators jump around from step to step; the averaged weights are smoother
and usually generalize better. Evaluate and ship the EMA copy, train the raw one.
"""
import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.lerp_(p.detach(), 1.0 - self.decay)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        self.model.load_state_dict(state)
