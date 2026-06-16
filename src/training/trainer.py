import os
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from pathlib import Path
from typing import Optional, Dict, Tuple
import json

from src.models.multitask_model import MultiTaskModel
from src.models.losses import MultiTaskLoss
from src.data.dataset import build_dataloaders
from src.training.config import TrainingConfig


class Trainer:
    """
    Training loop for the multi-task vision model.

    Features:
    - Mixed precision training (torch.amp) for speed
    - Cosine LR scheduler with linear warmup
    - Best model checkpointing
    - Per-step and per-epoch logging
    - Gradient clipping for stability
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {self.device}")

        # model, loss, optimizer
        self.model = MultiTaskModel(
            pretrained_backbone=config.pretrained_backbone,
            cls_hidden_dim=config.cls_hidden_dim,
            det_hidden_dim=config.det_hidden_dim,
            dropout=config.dropout,
        ).to(self.device)

        self.loss_fn = MultiTaskLoss(
            lambda_cls=config.lambda_cls,
            lambda_det=config.lambda_det,
        )

        # only optimize head parameters — backbone is frozen
        head_params = (
            list(self.model.cls_head.parameters()) +
            list(self.model.det_head.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            head_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # cosine scheduler with warmup
        self.scheduler = self._build_scheduler()

        # mixed precision scaler (only active on CUDA)
        self.scaler = GradScaler("cuda", enabled=self.device.type == "cuda")

        # dataloaders
        self.train_loader, self.val_dataset = build_dataloaders(
            voc_root=config.voc_root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            max_train_samples=config.max_train_samples,
            max_val_samples=config.max_val_samples,
        )

        # tracking
        self.best_val_loss = float("inf")
        self.history = []

    def _build_scheduler(self):
        """Cosine annealing with linear warmup."""
        def lr_lambda(epoch):
            if epoch < self.config.warmup_epochs:
                # linear warmup
                return (epoch + 1) / self.config.warmup_epochs
            # cosine decay
            progress = (epoch - self.config.warmup_epochs) / (
                self.config.num_epochs - self.config.warmup_epochs
            )
            return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _train_epoch(self, epoch: int) -> Dict:
        """Run one training epoch."""
        self.model.train()
        total_loss = cls_loss_sum = det_loss_sum = 0.0
        correct = total = 0
        step = 0

        for batch in self.train_loader:
            if batch[0] is None:
                continue

            images, labels, bboxes, _ = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            bboxes = bboxes.to(self.device)

            self.optimizer.zero_grad()

            # mixed precision forward pass
            with autocast("cuda", enabled=self.device.type == "cuda"):
                cls_logits, bbox_pred = self.model(images)
                loss, loss_dict = self.loss_fn(
                    cls_logits, bbox_pred, labels, bboxes
                )

            # backward + gradient clipping + optimizer step
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # metrics
            total_loss += loss_dict["total_loss"]
            cls_loss_sum += loss_dict["cls_loss"]
            det_loss_sum += loss_dict["det_loss"]

            preds = cls_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            step += 1

            if step % self.config.log_every_n_steps == 0:
                print(
                    f"  Epoch {epoch+1} step {step} | "
                    f"loss={loss_dict['total_loss']:.4f} | "
                    f"cls={loss_dict['cls_loss']:.4f} | "
                    f"det={loss_dict['det_loss']:.4f}"
                )

        n = max(step, 1)
        return {
            "train_loss": total_loss / n,
            "train_cls_loss": cls_loss_sum / n,
            "train_det_loss": det_loss_sum / n,
            "train_acc": correct / max(total, 1),
        }

    def _val_epoch(self) -> Dict:
        """Run one validation epoch."""
        from torch.utils.data import DataLoader
        from src.data.utils import collate_fn

        self.model.eval()
        total_loss = cls_loss_sum = det_loss_sum = 0.0
        correct = total = 0
        iou_sum = 0.0
        step = 0

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )

        with torch.no_grad():
            for batch in val_loader:
                if batch[0] is None:
                    continue

                images, labels, bboxes, _ = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                bboxes = bboxes.to(self.device)

                cls_logits, bbox_pred = self.model(images)
                loss, loss_dict = self.loss_fn(
                    cls_logits, bbox_pred, labels, bboxes
                )

                total_loss += loss_dict["total_loss"]
                cls_loss_sum += loss_dict["cls_loss"]
                det_loss_sum += loss_dict["det_loss"]

                preds = cls_logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                iou_sum += self._batch_iou(bbox_pred, bboxes)
                step += 1

        n = max(step, 1)
        return {
            "val_loss": total_loss / n,
            "val_cls_loss": cls_loss_sum / n,
            "val_det_loss": det_loss_sum / n,
            "val_acc": correct / max(total, 1),
            "val_iou": iou_sum / n,
        }

    def _batch_iou(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """
        Mean IoU across a batch.
        IoU = intersection area / union area.
        Both tensors: [B, 4] normalized bbox coordinates.
        """
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])

        inter_area = (
            (inter_x2 - inter_x1).clamp(0) *
            (inter_y2 - inter_y1).clamp(0)
        )

        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union_area = pred_area + target_area - inter_area

        iou = inter_area / union_area.clamp(min=1e-6)
        return iou.mean().item()

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool):
        """Save model checkpoint with metadata."""
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config.__dict__,
        }

        path = Path(self.config.checkpoint_dir)

        if is_best:
            torch.save(checkpoint, path / "best_model.pt")
            print(f"  Saved best model (val_loss={metrics['val_loss']:.4f})")

        if (epoch + 1) % self.config.save_every_n_epochs == 0:
            torch.save(checkpoint, path / f"checkpoint_epoch_{epoch+1}.pt")

    def train(self):
        """Full training loop."""
        self.config.display()
        print(f"Train batches: {len(self.train_loader)}")

        for epoch in range(self.config.num_epochs):
            t0 = time.time()

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._val_epoch()
            self.scheduler.step()

            metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1}
            self.history.append(metrics)

            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]

            self._save_checkpoint(epoch, metrics, is_best)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1}/{self.config.num_epochs} | "
                f"train_loss={train_metrics['train_loss']:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"train_acc={train_metrics['train_acc']:.3f} | "
                f"val_acc={val_metrics['val_acc']:.3f} | "
                f"val_iou={val_metrics['val_iou']:.3f} | "
                f"lr={self.optimizer.param_groups[0]['lr']:.6f} | "
                f"time={elapsed:.1f}s"
            )

        # save training history
        with open(Path(self.config.checkpoint_dir) / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        print("\nTraining complete.")