import os
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.utils import (
    parse_voc_xml,
    get_primary_object,
    load_image_ids,
    CLASS_TO_IDX,
    NUM_CLASSES,
)
from src.data.transforms import get_train_transforms, get_val_transforms, Compose


class VOCMultiTaskDataset(Dataset):
    """
    PASCAL VOC 2012 dataset for multi-task learning.

    Each sample returns:
        image  : Tensor [3, H, W] — normalized
        label  : int — primary object class index
        bbox   : List[float] — [x_min, y_min, x_max, y_max] normalized [0, 1]
        image_id: str — VOC image ID (e.g. '2007_000032')

    Primary object selection: largest non-difficult bounding box per image.
    Images with no valid objects are skipped (returns None, filtered by collate_fn).
    """

    def __init__(
        self,
        voc_root: str,
        split: str = "train",
        transforms: Optional[Compose] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            voc_root   : path to VOCdevkit/VOC2012/
            split      : 'train', 'val', or 'trainval'
            transforms : Compose instance (defaults to split-appropriate transforms)
            max_samples: cap dataset size — useful for local CPU dev runs
        """
        self.voc_root = Path(voc_root)
        self.split = split
        self.image_dir = self.voc_root / "JPEGImages"
        self.annotation_dir = self.voc_root / "Annotations"

        self.image_ids = load_image_ids(str(self.voc_root), split)
        if max_samples is not None:
            self.image_ids = self.image_ids[:max_samples]

        if transforms is not None:
            self.transforms = transforms
        elif split == "train":
            self.transforms = get_train_transforms()
        else:
            self.transforms = get_val_transforms()

        self._skipped = 0

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]

        # --- load annotation ---
        xml_path = self.annotation_dir / f"{image_id}.xml"
        try:
            parsed = parse_voc_xml(str(xml_path))
        except Exception as e:
            self._skipped += 1
            return None

        primary = get_primary_object(parsed)
        if primary is None:
            self._skipped += 1
            return None

        label = primary["label_idx"]
        bbox  = primary["bbox"]

        # --- load image ---
        img_path = self.image_dir / f"{parsed['filename']}"
        if not img_path.exists():
            # some VOC filenames lack extension
            img_path = self.image_dir / f"{image_id}.jpg"
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            self._skipped += 1
            return None

        # --- apply transforms ---
        if self.transforms is not None:
            image, bbox = self.transforms(image, bbox)

        return image, label, bbox, image_id

    def get_class_name(self, idx: int) -> str:
        from src.data.utils import IDX_TO_CLASS
        return IDX_TO_CLASS.get(idx, "unknown")

    def class_distribution(self) -> dict:
        """
        Iterate all annotations and count primary object per image.
        Useful for EDA. Slow — don't call during training.
        """
        from collections import Counter
        from src.data.utils import IDX_TO_CLASS
        counter = Counter()
        for image_id in self.image_ids:
            xml_path = self.annotation_dir / f"{image_id}.xml"
            try:
                parsed = parse_voc_xml(str(xml_path))
                primary = get_primary_object(parsed)
                if primary:
                    counter[primary["name"]] += 1
            except Exception:
                continue
        return dict(counter)


def build_dataloaders(
    voc_root: str,
    batch_size: int = 32,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build train and val DataLoaders.
    num_workers=0 is default for Windows (multiprocessing issues with >0).
    """
    from src.data.utils import collate_fn

    train_ds = VOCMultiTaskDataset(
        voc_root, split="train", max_samples=max_train_samples
    )
    val_ds = VOCMultiTaskDataset(
        voc_root, split="val", max_samples=max_val_samples
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_ds