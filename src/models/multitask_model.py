import torch
import torch.nn as nn
from typing import Tuple, Dict

from src.models.backbone import Backbone
from src.models.heads import ClassificationHead, DetectionHead
from src.data.utils import NUM_CLASSES


class MultiTaskModel(nn.Module):
    """
    Multi-task vision model for simultaneous classification and detection.

    Architecture:
        Image [B, 3, 224, 224]
            ↓
        Backbone (frozen EfficientNet-B0)
            ↓
        Features [B, 1280]
            ↓          ↓
        ClsHead     DetHead
            ↓          ↓
        [B, 20]     [B, 4]

    Both heads share the same backbone features — one forward
    pass through backbone serves both tasks simultaneously.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        cls_hidden_dim: int = 512,
        det_hidden_dim: int = 256,
        dropout: float = 0.3,
        pretrained_backbone: bool = True,
    ):
        super().__init__()

        self.backbone = Backbone(pretrained=pretrained_backbone)

        self.cls_head = ClassificationHead(
            in_features=self.backbone.output_dim,
            num_classes=num_classes,
            hidden_dim=cls_hidden_dim,
            dropout=dropout,
        )

        self.det_head = DetectionHead(
            in_features=self.backbone.output_dim,
            hidden_dim=det_hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: image batch [B, 3, 224, 224]
        Returns:
            cls_logits: [B, 20] raw class scores
            bbox_pred:  [B, 4]  predicted bbox coordinates in [0, 1]
        """
        features = self.backbone(x)
        cls_logits = self.cls_head(features)
        bbox_pred = self.det_head(features)
        return cls_logits, bbox_pred

    def predict(self, x: torch.Tensor) -> Dict:
        """
        Inference mode — returns human readable predictions.
        Args:
            x: image batch [B, 3, 224, 224]
        Returns:
            dict with class indices, class names, confidence scores, bboxes
        """
        from src.data.utils import IDX_TO_CLASS

        self.eval()
        with torch.no_grad():
            cls_logits, bbox_pred = self.forward(x)

        probs = torch.softmax(cls_logits, dim=1)
        confidence, class_idx = probs.max(dim=1)

        return {
            "class_idx": class_idx.tolist(),
            "class_names": [IDX_TO_CLASS[i] for i in class_idx.tolist()],
            "confidence": confidence.tolist(),
            "bbox": bbox_pred.tolist(),
        }

    def count_parameters(self) -> Dict:
        """Parameter breakdown by component."""
        def count(module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return trainable, total

        bb_t, bb_tot = count(self.backbone)
        cls_t, cls_tot = count(self.cls_head)
        det_t, det_tot = count(self.det_head)

        return {
            "backbone":    {"trainable": bb_t,  "total": bb_tot},
            "cls_head":    {"trainable": cls_t,  "total": cls_tot},
            "det_head":    {"trainable": det_t,  "total": det_tot},
            "model_total": {"trainable": bb_t + cls_t + det_t,
                            "total": bb_tot + cls_tot + det_tot},
        }