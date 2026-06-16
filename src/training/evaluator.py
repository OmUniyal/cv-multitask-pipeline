import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
from src.data.utils import collate_fn, IDX_TO_CLASS
from src.models.multitask_model import MultiTaskModel
from src.training.config import TrainingConfig


class Evaluator:
    """
    Formal evaluation of the trained multi-task model.

    Metrics:
    - Top-1 accuracy (classification)
    - Mean IoU (detection)
    - Per-class accuracy breakdown
    - Grad-CAM visualization
    """

    def __init__(self, model: MultiTaskModel, device: torch.device):
        self.model = model
        self.device = device

    def evaluate(
        self,
        dataset,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> Dict:
        """
        Run full evaluation on a dataset.
        Returns dict of all metrics.
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        self.model.eval()
        all_preds, all_labels = [], []
        iou_scores = []

        with torch.no_grad():
            for batch in loader:
                if batch[0] is None:
                    continue

                images, labels, bboxes, _ = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                bboxes = bboxes.to(self.device)

                cls_logits, bbox_pred = self.model(images)

                preds = cls_logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                iou_scores.extend(
                    self._per_sample_iou(bbox_pred, bboxes)
                )

        metrics = self._compute_metrics(all_preds, all_labels, iou_scores)
        return metrics

    def _per_sample_iou(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> List[float]:
        """IoU for each sample in a batch. Returns list of floats."""
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])

        inter_area = (
            (inter_x2 - inter_x1).clamp(0) *
            (inter_y2 - inter_y1).clamp(0)
        )
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (
            (target[:, 2] - target[:, 0]) *
            (target[:, 3] - target[:, 1])
        )
        union_area = pred_area + target_area - inter_area
        iou = inter_area / union_area.clamp(min=1e-6)
        return iou.cpu().tolist()

    def _compute_metrics(
        self,
        preds: List[int],
        labels: List[int],
        iou_scores: List[float],
    ) -> Dict:
        """Compute top-1 accuracy, per-class accuracy, mean IoU."""
        preds = torch.tensor(preds)
        labels = torch.tensor(labels)

        # overall top-1 accuracy
        top1_acc = (preds == labels).float().mean().item()

        # per-class accuracy
        per_class_acc = {}
        for cls_idx in range(20):
            mask = labels == cls_idx
            if mask.sum() == 0:
                continue
            cls_acc = (preds[mask] == labels[mask]).float().mean().item()
            per_class_acc[IDX_TO_CLASS[cls_idx]] = round(cls_acc, 3)

        # mean IoU
        mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

        # IoU threshold accuracy (IoU > 0.5 = correct detection)
        iou_50_acc = float(np.mean([s > 0.5 for s in iou_scores]))

        return {
            "top1_accuracy": round(top1_acc, 4),
            "mean_iou": round(mean_iou, 4),
            "iou_50_accuracy": round(iou_50_acc, 4),
            "per_class_accuracy": per_class_acc,
            "num_samples": len(preds),
        }

    def grad_cam(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single image.

        Grad-CAM uses gradients flowing into the last conv layer
        to highlight which regions of the image influenced the
        classification decision most.

        Args:
            image: single image tensor [1, 3, 224, 224]
            target_class: class to explain. If None, uses predicted class.
        Returns:
            heatmap: numpy array [224, 224] values in [0, 1]
        """
        self.model.eval()

        # hook to capture feature maps and gradients
        # from last conv block of EfficientNet backbone
        feature_maps = []
        gradients = []

        def forward_hook(module, input, output):
            feature_maps.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        # attach hooks to last conv block
        target_layer = self.model.backbone.features[-1]
        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        image = image.to(self.device)
        image.requires_grad_(True)

        # forward pass
        cls_logits, _ = self.model(image)

        if target_class is None:
            target_class = cls_logits.argmax(dim=1).item()

        # backward pass for target class only
        self.model.zero_grad()
        cls_logits[0, target_class].backward()

        # remove hooks
        fwd_handle.remove()
        bwd_handle.remove()

        # compute Grad-CAM
        grads = gradients[0]           # [1, C, H, W]
        fmaps = feature_maps[0]        # [1, C, H, W]

        # global average pool the gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # weighted sum of feature maps
        cam = (weights * fmaps).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)

        # upsample to input image size
        cam = F.interpolate(
            cam, size=(224, 224), mode="bilinear", align_corners=False
        )

        # normalize to [0, 1]
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, target_class