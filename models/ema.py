# ================================================================
# models/ema.py
# EMA Helper for model parameter smoothing
# ================================================================

import torch

class EMAHelper:
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    new_avg = (1.0 - self.mu) * param.data + self.mu * self.shadow[name]
                    self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.shadow[name].clone()

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
