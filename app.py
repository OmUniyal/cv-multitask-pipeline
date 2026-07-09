import os
import torch
import gradio as gr
from pathlib import Path
from huggingface_hub import hf_hub_download
from src.models.multitask_model import MultiTaskModel
from src.data.utils import IDX_TO_CLASS
from src.data.transforms import get_val_transforms
from src.ui.visualize import draw_prediction
from PIL import Image
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable CUDA entirely on Spaces
warnings.filterwarnings("ignore", message=".*NVML.*")
warnings.filterwarnings("ignore", message=".*cuda.*")

# HuggingFace ZeroGPU compatibility
try:
    import spaces
    @spaces.GPU(duration=0)
    def warmup():
        pass
    warmup()
except Exception:
    pass  # not on HuggingFace Spaces, skip

# ---- model loading ----

REPO_ID = "OmUniyal/cv-multitask-pipeline"
MODEL_FILE = "best_model.pt"

# HuggingFace free Spaces are CPU-only
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and not os.environ.get("SPACE_ID")) else "cpu")
print(f"Device: {DEVICE}")


def load_model():
    """
    Download checkpoint from HuggingFace Model Hub and load into memory.
    Falls back to local models/best_model.pt if available.
    """
    local_path = Path("models/best_model.pt")

    if local_path.exists():
        print(f"Loading model from local path: {local_path}")
        checkpoint_path = str(local_path)
    else:
        print(f"Downloading model from HuggingFace Hub: {REPO_ID}")
        checkpoint_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILE,
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )
    model = MultiTaskModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    return model


# load once at startup
model = load_model()


# ---- inference ----

def predict(image: Image.Image) -> tuple:
    """
    Run inference on a PIL image.
    Returns annotated image and prediction details text.
    """
    if image is None:
        return None, "Please upload an image."

    transforms = get_val_transforms(size=224)
    tensor, _ = transforms(image, [0.0, 0.0, 1.0, 1.0])
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        cls_logits, bbox_pred = model(tensor)

    probs = torch.softmax(cls_logits, dim=1)
    confidence, class_idx = probs.max(dim=1)

    class_idx = class_idx.item()
    confidence = confidence.item()
    bbox = bbox_pred[0].tolist()

    # swap if invalid
    if bbox[0] > bbox[2]:
        bbox[0], bbox[2] = bbox[2], bbox[0]
    if bbox[1] > bbox[3]:
        bbox[1], bbox[3] = bbox[3], bbox[1]

    bbox_dict = {
        "x_min": bbox[0],
        "y_min": bbox[1],
        "x_max": bbox[2],
        "y_max": bbox[3],
    }

    annotated = draw_prediction(
        image=image,
        class_name=IDX_TO_CLASS[class_idx],
        class_idx=class_idx,
        confidence=confidence,
        bbox=bbox_dict,
    )

    result_text = (
        f"Class: {IDX_TO_CLASS[class_idx]}\n"
        f"Confidence: {confidence:.1%}\n"
        f"Bounding Box:\n"
        f"  x_min: {bbox[0]:.3f}\n"
        f"  y_min: {bbox[1]:.3f}\n"
        f"  x_max: {bbox[2]:.3f}\n"
        f"  y_max: {bbox[3]:.3f}"
    )

    return annotated, result_text


# ---- Gradio interface ----

with gr.Blocks(title="CV Multitask Pipeline") as demo:
    gr.Markdown("""
    # CV Multitask Pipeline
    **Multi-task object classification and detection**

    Upload an image containing any PASCAL VOC object to get:
    - Object class prediction (20 classes)
    - Bounding box localisation

    *Supported classes: aeroplane, bicycle, bird, boat, bottle, bus, car,
    cat, chair, cow, diningtable, dog, horse, motorbike, person,
    pottedplant, sheep, sofa, train, tvmonitor*
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            predict_btn = gr.Button("Run Prediction", variant="primary")

        with gr.Column():
            output_image = gr.Image(type="pil", label="Prediction (with bbox overlay)")
            output_text = gr.Textbox(label="Prediction Details", lines=8)

    predict_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_image, output_text],
    )

    gr.Markdown("""
    ---
    **Model:** EfficientNet-B0 backbone (frozen) + custom classification and detection heads

    **Training:** PASCAL VOC 2012 — 5,717 images, 30 epochs, Google Colab T4 GPU

    **Results:** Top-1 Accuracy: 79.9% | Mean IoU: 0.468 | IoU@0.5: 49.2%

    [GitHub](https://github.com/OmUniyal/cv-multitask-pipeline) | 
    [Model weights](https://huggingface.co/OmUniyal/cv-multitask-pipeline)
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )