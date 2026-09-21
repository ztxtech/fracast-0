"""损失函数公共组件。"""
from module.losses.quantile import pinball_loss, pinball_loss_mask

__all__ = ["pinball_loss", "pinball_loss_mask"]
