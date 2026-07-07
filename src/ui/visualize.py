import io
import base64
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional
import numpy as np


# colour palette — one per VOC class
VOC_COLORS = [
    (255, 0, 0),     # aeroplane — red
    (0, 255, 0),     # bicycle — green
    (0, 0, 255),     # bird — blue
    (255, 255, 0),   # boat — yellow
    (255, 0, 255),   # bottle — magenta
    (0, 255, 255),   # bus — cyan
    (255, 128, 0),   # car — orange
    (128, 0, 255),   # cat — purple
    (0, 128, 255),   # chair — light blue
    (255, 0, 128),   # cow — pink
    (128, 255, 0),   # diningtable — lime
    (0, 255, 128),   # dog — mint
    (128, 128, 0),   # horse — olive
    (0, 128, 128),   # motorbike — teal
    (128, 0, 0),     # person — dark red
    (0, 0, 128),     # pottedplant — dark blue
    (128, 128, 255), # sheep — lavender
    (255, 128, 128), # sofa — salmon
    (128, 255, 128), # train — light green
    (255, 255, 128), # tvmonitor — light yellow
]


def draw_prediction(
    image: Image.Image,
    class_name: str,
    class_idx: int,
    confidence: float,
    bbox: dict,
    line_width: int = 3,
) -> Image.Image:
    """
    Draw bounding box and label on a PIL image.

    Args:
        image: PIL Image (original, not preprocessed)
        class_name: predicted class name
        class_idx: predicted class index (for colour selection)
        confidence: prediction confidence [0, 1]
        bbox: dict with x_min, y_min, x_max, y_max (normalized [0,1])
        line_width: bbox border thickness in pixels

    Returns:
        annotated PIL Image
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)
    w, h = image.size

    # denormalize bbox to pixel coordinates
    x_min = int(bbox["x_min"] * w)
    y_min = int(bbox["y_min"] * h)
    x_max = int(bbox["x_max"] * w)
    y_max = int(bbox["y_max"] * h)

    # ensure valid bbox — swap if min > max (model not fully trained)
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    # get class colour
    color = VOC_COLORS[class_idx % len(VOC_COLORS)]

    # draw bounding box
    draw.rectangle(
        [x_min, y_min, x_max, y_max],
        outline=color,
        width=line_width,
    )

    # draw label background + text
    label = f"{class_name} {confidence:.0%}"
    font_size = max(12, min(20, h // 30))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # label background box
    text_bbox = draw.textbbox((x_min, y_min), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    label_y = max(0, y_min - text_h - 4)
    draw.rectangle(
        [x_min, label_y, x_min + text_w + 4, label_y + text_h + 4],
        fill=color,
    )
    draw.text(
        (x_min + 2, label_y + 2),
        label,
        fill=(255, 255, 255),
        font=font,
    )

    return image


def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 string for API requests."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(image_base64: str) -> Image.Image:
    """Convert base64 string back to PIL Image."""
    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")