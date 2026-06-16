from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """
    Central config for all training hyperparameters.
    Using dataclass so config is typed, printable, and serializable.
    """

    # --- data ---
    voc_root: str = "data/VOCdevkit/VOC2012"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0            # 0 for Windows, 4 for Linux/Colab
    max_train_samples: Optional[int] = 200   # None = full dataset
    max_val_samples: Optional[int] = 50      # None = full dataset

    # --- model ---
    pretrained_backbone: bool = True
    cls_hidden_dim: int = 512
    det_hidden_dim: int = 256
    dropout: float = 0.3

    # --- loss ---
    lambda_cls: float = 1.0
    lambda_det: float = 5.0

    # --- optimizer ---
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9           # for SGD

    # --- scheduler ---
    scheduler: str = "cosine"       # "cosine" or "step"
    num_epochs: int = 30
    warmup_epochs: int = 3

    # --- checkpointing ---
    checkpoint_dir: str = "models"
    save_every_n_epochs: int = 5
    keep_best_only: bool = True

    # --- logging ---
    log_every_n_steps: int = 10
    experiment_name: str = "multitask_v1"

    def __post_init__(self):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def display(self):
        print("\n--- Training Config ---")
        for k, v in self.__dict__.items():
            print(f"  {k}: {v}")
        print("----------------------\n")