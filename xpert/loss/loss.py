import torch
import torch.nn.functional as F
import torch.nn as nn


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()

# Optional additional masked losses (kept minimal)
def masked_huber_loss(input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 0.5) -> torch.Tensor:
    mask = mask.float()
    loss_fn = nn.HuberLoss(delta=delta, reduction='sum')
    loss = loss_fn(input * mask, target * mask)
    return loss / mask.sum()


__all__ = [
    "masked_mse_loss",
    "masked_relative_error",
    "masked_huber_loss",
]


