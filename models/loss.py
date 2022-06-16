from torch import nn
from torchdistill.losses.single import register_single_loss


@register_single_loss
class RelativeSquaredError(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        mean_target = targets.mean()
        return ((preds - targets) ** 2).sum() / ((targets - mean_target) ** 2).sum()
