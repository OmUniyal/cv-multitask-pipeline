import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from typing import Tuple


class Backbone(nn.Module):
    """
    Frozen EfficientNet-B0 feature extractor.

    Takes an image tensor [B, 3, 224, 224] and returns a
    flattened feature vector [B, 1280] ready for task heads.

    We freeze all backbone weights — we're not retraining
    EfficientNet, just using its learned visual features.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # load EfficientNet-B0 with ImageNet weights
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        efficientnet = models.efficientnet_b0(weights=weights)

        # remove the final classifier head — we only want features
        # EfficientNet structure: features -> avgpool -> classifier
        # we keep features + avgpool, discard classifier
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool

        # EfficientNet-B0 outputs 1280 feature channels after avgpool
        self.output_dim = 1280

        # freeze all backbone weights
        self._freeze()

    def _freeze(self):
        """Freeze all backbone parameters — no gradient updates."""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Optionally unfreeze for fine-tuning in later training stages.
        Not used in Phase 2 but useful for Phase 3 experiments.
        """
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image tensor [B, 3, 224, 224]
        Returns:
            features: flattened tensor [B, 1280]
        """
        x = self.features(x)   # [B, 1280, 7, 7]
        x = self.avgpool(x)    # [B, 1280, 1, 1]
        x = torch.flatten(x, 1)  # [B, 1280]
        return x

    def count_parameters(self) -> Tuple[int, int]:
        """Returns (trainable_params, total_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total