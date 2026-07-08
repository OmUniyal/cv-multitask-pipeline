# ============================================================
# CV Multitask Pipeline — Full Training on Google Colab
# Run each cell block sequentially in a Colab notebook
# ============================================================

# ---- CELL 1: Check GPU ----
import subprocess
subprocess.run(['nvidia-smi'])

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")


# ---- CELL 2: Clone repo and install dependencies ----
import subprocess

subprocess.run(['git', 'clone', 
    'https://github.com/OmUniyal/cv-multitask-pipeline.git'])

import os
os.chdir('cv-multitask-pipeline')

subprocess.run(['pip', 'install', '-r', 'requirements.txt', '-q'])
subprocess.run(['pip', 'install', '-e', '.', '-q'])


# ---- CELL 3: Download VOC 2012 dataset ----
import subprocess
import os

os.makedirs('data', exist_ok=True)

print("Downloading VOC 2012...")
subprocess.run([
    'wget', '-q',
    'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
    '-O', 'data/VOCtrainval_11-May-2012.tar'
])

print("Extracting...")
subprocess.run(['tar', '-xf', 
    'data/VOCtrainval_11-May-2012.tar', '-C', 'data/'])
print("Done.")


# ---- CELL 4: Verify dataset ----
import os
from pathlib import Path

voc_root = Path('data/VOCdevkit/VOC2012')
train_ids = (voc_root / 'ImageSets/Main/train.txt').read_text().strip().split('\n')
val_ids = (voc_root / 'ImageSets/Main/val.txt').read_text().strip().split('\n')

print(f"Train samples: {len(train_ids)}")
print(f"Val samples:   {len(val_ids)}")

from src.data.dataset import VOCMultiTaskDataset
ds = VOCMultiTaskDataset('data/VOCdevkit/VOC2012', split='train', max_samples=5)
sample = ds[0]
print(f"Sample shape: {list(sample[0].shape)}, class: {ds.get_class_name(sample[1])}")


# ---- CELL 5: Full training run ----
from src.training.config import TrainingConfig
from src.training.trainer import Trainer

config = TrainingConfig(
    voc_root='data/VOCdevkit/VOC2012',
    max_train_samples=None,   # full dataset
    max_val_samples=None,     # full dataset
    batch_size=64,            # larger batch on GPU
    num_epochs=30,
    num_workers=2,            # Colab supports multiple workers
    learning_rate=1e-3,
    lambda_cls=1.0,
    lambda_det=5.0,
    experiment_name='voc_full_v1',
    save_every_n_epochs=5,
)

trainer = Trainer(config)
trainer.train()


# ---- CELL 6: Evaluate best model ----
import torch
from src.models.multitask_model import MultiTaskModel
from src.training.evaluator import Evaluator
from src.data.dataset import VOCMultiTaskDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load('models/best_model.pt', map_location=device)
model = MultiTaskModel()
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)  # add this line
model.eval()      # add this line
print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

evaluator = Evaluator(model, device)
val_dataset = VOCMultiTaskDataset('data/VOCdevkit/VOC2012', split='val')
metrics = evaluator.evaluate(val_dataset, batch_size=64)

print(f"\n=== Final Evaluation ===")
print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
print(f"Mean IoU:       {metrics['mean_iou']:.4f}")
print(f"IoU@0.5:        {metrics['iou_50_accuracy']:.4f}")
print(f"Num samples:    {metrics['num_samples']}")
print(f"\nPer-class accuracy:")
for cls, acc in sorted(metrics['per_class_accuracy'].items()):
    print(f"  {cls:<15} {acc:.3f}")


# ---- CELL 7: Download checkpoint ----
from google.colab import files
files.download('models/best_model.pt')