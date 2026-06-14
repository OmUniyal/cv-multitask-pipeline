import torch
import torch.nn as nn
from typing import Dict, Tuple


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss combining classification and detection.

    total_loss = lambda_cls * CrossEntropyLoss + lambda_det * SmoothL1Loss

    Why SmoothL1 for detection (not MSE)?
    - MSE penalizes large errors quadratically — outlier bboxes
      dominate the loss and destabilize training
    - SmoothL1 is quadratic for small errors, linear for large errors
      (threshold at beta=1.0) — robust to outliers, stable training

    Why CrossEntropy for classification?
    - Standard for multi-class problems
    - Applies log-softmax internally — numerically stable
    - Penalizes confident wrong predictions heavily
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_det: float = 5.0,
        smoothl1_beta: float = 1.0,
    ):
        super().__init__()

        self.lambda_cls = lambda_cls
        self.lambda_det = lambda_det

        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.det_loss_fn = nn.SmoothL1Loss(beta=smoothl1_beta)

    def forward(
        self,
        cls_logits: torch.Tensor,
        bbox_pred: torch.Tensor,
        cls_targets: torch.Tensor,
        bbox_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            cls_logits:   [B, 20] raw class scores from classification head
            bbox_pred:    [B, 4]  predicted bbox from detection head
            cls_targets:  [B]     ground truth class indices (long)
            bbox_targets: [B, 4]  ground truth bbox coordinates

        Returns:
            total_loss: scalar tensor (backprop through this)
            loss_dict:  breakdown for logging
        """
        cls_loss = self.cls_loss_fn(cls_logits, cls_targets)
        det_loss = self.det_loss_fn(bbox_pred, bbox_targets)

        total_loss = (self.lambda_cls * cls_loss) + (self.lambda_det * det_loss)

        loss_dict = {
            "total_loss": total_loss.item(),
            "cls_loss": cls_loss.item(),
            "det_loss": det_loss.item(),
            "cls_loss_weighted": (self.lambda_cls * cls_loss).item(),
            "det_loss_weighted": (self.lambda_det * det_loss).item(),
        }

        return total_loss, loss_dict