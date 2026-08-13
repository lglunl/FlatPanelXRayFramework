"""损失函数"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """MSE + L1 混合损失"""
    def __init__(self, w_mse=0.5, w_l1=0.5):
        super().__init__()
        self.w_mse = w_mse
        self.w_l1 = w_l1

    def forward(self, pred, target):
        return self.w_mse * F.mse_loss(pred, target) + self.w_l1 * F.l1_loss(pred, target)


LOSS_FACTORIES = {
    "mse": lambda: nn.MSELoss(),
    "l1": lambda: nn.L1Loss(),
    "combined": lambda: CombinedLoss(0.5, 0.5),
}


def get_loss(name: str) -> nn.Module:
    if name not in LOSS_FACTORIES:
        raise KeyError(f"未知损失函数 '{name}'，可用: {list(LOSS_FACTORIES.keys())}")
    return LOSS_FACTORIES[name]()
