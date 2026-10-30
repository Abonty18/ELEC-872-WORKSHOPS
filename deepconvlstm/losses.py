"""Loss used in Workshop Part F. The reference still uses cross entropy."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multiclass focal loss: -(1 - p_true)**gamma * log(p_true).

    Optional class weights multiply the per-example loss. With no class weights
    and gamma=0 this is exactly ordinary mean cross entropy.
    """

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = float(gamma)
        self.register_buffer("weight", weight)

    def forward(self, logits, targets):
        log_prob = F.log_softmax(logits, dim=1)
        log_pt = log_prob.gather(1, targets[:, None]).squeeze(1)
        loss = -((1.0 - log_pt.exp()).clamp_min(0.0) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()
