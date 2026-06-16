import argparse
import torch
from src.training.config import TrainingConfig
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the multi-task CV pipeline"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--lambda-cls", type=float, default=1.0,
        help="Classification loss weight"
    )
    parser.add_argument(
        "--lambda-det", type=float, default=5.0,
        help="Detection loss weight"
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=None,
        help="Cap training samples (None = full dataset)"
    )
    parser.add_argument(
        "--max-val-samples", type=int, default=None,
        help="Cap validation samples (None = full dataset)"
    )
    parser.add_argument(
        "--experiment-name", type=str, default="multitask_v1",
        help="Name for this training run"
    )
    parser.add_argument(
        "--voc-root", type=str, default="data/VOCdevkit/VOC2012",
        help="Path to VOC2012 root"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = TrainingConfig(
        voc_root=args.voc_root,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_cls=args.lambda_cls,
        lambda_det=args.lambda_det,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        experiment_name=args.experiment_name,
    )

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()