import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import DataLoader


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

CLASS_TO_IDX: Dict[str, int] = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: cls for cls, idx in CLASS_TO_IDX.items()}
NUM_CLASSES = len(VOC_CLASSES)


def parse_voc_xml(xml_path: str) -> Dict:
    """
    Parse a single PASCAL VOC annotation XML file.

    Returns a dict with:
        - image_path: str
        - width: int
        - height: int
        - objects: List of dicts, each with:
            - name: str (class label)
            - label_idx: int
            - bbox: [x_min, y_min, x_max, y_max] normalized to [0, 1]
            - difficult: bool
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    folder = root.findtext("folder", default="VOC2012")
    filename = root.findtext("filename")

    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASS_TO_IDX:
            continue

        difficult = bool(int(obj.findtext("difficult", default="0")))

        bndbox = obj.find("bndbox")
        x_min = float(bndbox.findtext("xmin"))
        y_min = float(bndbox.findtext("ymin"))
        x_max = float(bndbox.findtext("xmax"))
        y_max = float(bndbox.findtext("ymax"))

        # normalize to [0, 1]
        bbox_norm = [
            x_min / width,
            y_min / height,
            x_max / width,
            y_max / height,
        ]
        # clamp to valid range
        bbox_norm = [max(0.0, min(1.0, v)) for v in bbox_norm]

        objects.append({
            "name": name,
            "label_idx": CLASS_TO_IDX[name],
            "bbox": bbox_norm,
            "difficult": difficult,
        })

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects,
    }


def get_primary_object(parsed: Dict) -> Optional[Dict]:
    """
    For multi-task training we need one label + one bbox per image.
    Strategy: pick the largest non-difficult bounding box by area.
    Falls back to any object if all are marked difficult.
    """
    objects = parsed["objects"]
    if not objects:
        return None

    non_difficult = [o for o in objects if not o["difficult"]]
    candidates = non_difficult if non_difficult else objects

    def bbox_area(obj):
        b = obj["bbox"]
        return (b[2] - b[0]) * (b[3] - b[1])

    return max(candidates, key=bbox_area)


def load_image_ids(voc_root: str, split: str = "train") -> List[str]:
    """
    Load image IDs from VOC ImageSets/Main/<split>.txt.
    split: 'train', 'val', or 'trainval'
    """
    split_file = Path(voc_root) / "ImageSets" / "Main" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file) as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


def collate_fn(batch: List) -> Tuple:
    """
    Custom collate for DataLoader.
    Filters out None samples (images that failed to parse).
    Returns:
        images: Tensor [B, C, H, W]
        labels: Tensor [B] (long)
        bboxes: Tensor [B, 4] (float, normalized)
        image_ids: List[str]
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None, []

    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    bboxes = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    image_ids = [b[3] for b in batch]

    return images, labels, bboxes, image_ids