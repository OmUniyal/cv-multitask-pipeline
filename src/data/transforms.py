import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from typing import Tuple, List
import random


class Compose:
    """Apply a sequence of transforms to both image and bbox."""
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, image, bbox):
        for t in self.transforms:
            image, bbox = t(image, bbox)
        return image, bbox


class ToTensor:
    """Convert PIL image to tensor. Bbox is already a list, pass through."""
    def __call__(self, image, bbox):
        return TF.to_tensor(image), bbox


class Normalize:
    """Normalize image tensor. Bbox unchanged."""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, bbox):
        return TF.normalize(image, self.mean, self.std), bbox


class Resize:
    """
    Resize image to (size, size).
    Bbox is normalized [0,1] so no change needed.
    """
    def __init__(self, size: int = 224):
        self.size = size

    def __call__(self, image, bbox):
        image = TF.resize(image, [self.size, self.size])
        return image, bbox


class RandomHorizontalFlip:
    """
    Flip image horizontally with probability p.
    Bbox x-coords must be mirrored: x_min' = 1 - x_max, x_max' = 1 - x_min.
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, bbox):
        if random.random() < self.p:
            image = TF.hflip(image)
            x_min, y_min, x_max, y_max = bbox
            bbox = [1.0 - x_max, y_min, 1.0 - x_min, y_max]
        return image, bbox


class RandomColorJitter:
    """Color jitter on image only. Bbox unchanged."""
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1):
        self.jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image, bbox):
        return self.jitter(image), bbox


class RandomCrop:
    """
    Crop a random region of the image, keeping at least min_overlap
    of the primary bbox inside the crop. Falls back to center crop
    if a valid crop isn't found within max_attempts.

    Bbox is recalculated relative to the cropped region.
    """
    def __init__(self, size: int = 224, min_overlap: float = 0.7, max_attempts: int = 10):
        self.size = size
        self.min_overlap = min_overlap
        self.max_attempts = max_attempts

    def __call__(self, image, bbox):
        w, h = image.size  # PIL: (width, height)
        x_min, y_min, x_max, y_max = bbox

        # convert normalized bbox to pixel coords
        bx1, by1 = x_min * w, y_min * h
        bx2, by2 = x_max * w, y_max * h

        crop_w = min(self.size, w)
        crop_h = min(self.size, h)

        for _ in range(self.max_attempts):
            left = random.randint(0, max(0, w - crop_w))
            top  = random.randint(0, max(0, h - crop_h))
            right  = left + crop_w
            bottom = top  + crop_h

            # intersection with bbox
            ix1 = max(bx1, left)
            iy1 = max(by1, top)
            ix2 = min(bx2, right)
            iy2 = min(by2, bottom)

            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                bbox_area  = (bx2 - bx1) * (by2 - by1)
                if bbox_area > 0 and (inter_area / bbox_area) >= self.min_overlap:
                    image = TF.crop(image, top, left, crop_h, crop_w)
                    image = TF.resize(image, [self.size, self.size])

                    # recalculate bbox relative to crop, re-normalize
                    new_bbox = [
                        (bx1 - left) / crop_w,
                        (by1 - top)  / crop_h,
                        (bx2 - left) / crop_w,
                        (by2 - top)  / crop_h,
                    ]
                    new_bbox = [max(0.0, min(1.0, v)) for v in new_bbox]
                    return image, new_bbox

        # fallback: center crop
        left = (w - crop_w) // 2
        top  = (h - crop_h) // 2
        image = TF.crop(image, top, left, crop_h, crop_w)
        image = TF.resize(image, [self.size, self.size])
        new_bbox = [
            (bx1 - left) / crop_w,
            (by1 - top)  / crop_h,
            (bx2 - left) / crop_w,
            (by2 - top)  / crop_h,
        ]
        new_bbox = [max(0.0, min(1.0, v)) for v in new_bbox]
        return image, new_bbox


def get_train_transforms(size: int = 224) -> Compose:
    return Compose([
        Resize(size + 32),           # resize slightly larger first
        RandomCrop(size),            # then random crop to target size
        RandomHorizontalFlip(p=0.5),
        RandomColorJitter(),
        ToTensor(),
        Normalize(),
    ])


def get_val_transforms(size: int = 224) -> Compose:
    return Compose([
        Resize(size),
        ToTensor(),
        Normalize(),
    ])