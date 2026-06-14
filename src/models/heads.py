import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Classification head — maps backbone features to class scores.

    Input:  [B, 1280] feature vector from backbone
    Output: [B, 20]   raw logits, one per VOC class

    No Softmax here — CrossEntropyLoss applies it internally.
    Architecture: Linear -> BN -> ReLU -> Dropout -> Linear
    """

    def __init__(
        self,
        in_features: int = 1280,
        num_classes: int = 20,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stable training from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1280] backbone features
        Returns:
            logits: [B, 20] raw class scores
        """
        return self.classifier(x)


class DetectionHead(nn.Module):
    """
    Detection head — maps backbone features to bbox coordinates.

    Input:  [B, 1280] feature vector from backbone
    Output: [B, 4]    predicted bbox [x_min, y_min, x_max, y_max]
                      all values in [0, 1] via Sigmoid

    Architecture: Linear -> BN -> ReLU -> Dropout -> Linear -> Sigmoid
    """

    def __init__(
        self,
        in_features: int = 1280,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.regressor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1280] backbone features
        Returns:
            bbox: [B, 4] predicted coordinates in [0, 1]
        """
        return self.regressor(x)