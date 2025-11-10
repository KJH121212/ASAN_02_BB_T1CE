# ================================================================
# models/ema_helper.py
# Exponential Moving Average (EMA) Helper
# ================================================================
import torch.nn as nn

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu                     # EMA decay rate
        self.shadow = {}                 # EMA 파라미터 저장 딕셔너리

    def register(self, module):
        """모델 파라미터를 shadow dict에 복사"""
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        """EMA 업데이트: θ_ema = μ·θ_ema + (1−μ)·θ"""
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        """현재 모델 파라미터를 EMA 파라미터로 교체"""
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        """EMA 적용된 모델 복사본 생성 (원본 유지)"""
        if isinstance(module, nn.DataParallel):
            inner = module.module
            module_copy = type(inner)(inner.config).to(inner.config.device)
            module_copy.load_state_dict(inner.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        """shadow dict 반환 (저장용)"""
        return self.shadow

    def load_state_dict(self, state_dict):
        """저장된 EMA 파라미터 복원"""
        self.shadow = state_dict
