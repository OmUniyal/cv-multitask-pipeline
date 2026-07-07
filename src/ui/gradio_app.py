import gradio as gr
import requests
from PIL import Image
from src.ui.visualize import draw_prediction, image_to_base64

import os
API_URL = os.environ.get("API_URL", "http://localhost:8000")


def check_api_health() -> str:
    """Check if the FastAPI server is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return f"API healthy — models loaded: {data['available_versions']}"
        return f"API returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "API not reachable — make sure FastAPI server is running on port 8000"


def predict_image(
    image: Image.Image,
    model_version: str,
) -> tuple:
    """
    Send image to FastAPI, get prediction, draw bbox overlay.

    Returns:
        annotated_image: PIL Image with bbox drawn
        result_text: formatted prediction summary
    """
    if image is None:
        return None, "Please upload an image."

    # check API is reachable
    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        if health.status_code != 200:
            return None, "API server not healthy."
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach API server. Make sure FastAPI is running on port 8000."

    # encode image to base64
    image_base64 = image_to_base64(image)

    # call API
    try:
        response = requests.post(
            f"{API_URL}/{model_version}/predict",
            json={
                "image_base64": image_base64,
                "model_version": model_version,
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        return None, f"API error: {e.response.json().get('detail', str(e))}"
    except Exception as e:
        return None, f"Request failed: {str(e)}"

    # parse response
    data = response.json()
    class_name = data["class_name"]
    class_idx = data["class_idx"]
    confidence = data["confidence"]
    bbox = data["bbox"]
    inference_time = data["inference_time_ms"]

    # draw bbox overlay on original image
    annotated = draw_prediction(
        image=image,
        class_name=class_name,
        class_idx=class_idx,
        confidence=confidence,
        bbox=bbox,
    )

    # format result text
    result_text = (
        f"Class: {class_name}\n"
        f"Confidence: {confidence:.1%}\n"
        f"Bounding Box:\n"
        f"  x_min: {bbox['x_min']:.3f}\n"
        f"  y_min: {bbox['y_min']:.3f}\n"
        f"  x_max: {bbox['x_max']:.3f}\n"
        f"  y_max: {bbox['y_max']:.3f}\n"
        f"Model: {model_version}\n"
        f"Inference time: {inference_time:.1f}ms"
    )

    return annotated, result_text


def build_interface() -> gr.Blocks:
    """Build and return the Gradio interface."""

    with gr.Blocks(
        title="CV Multitask Pipeline",
    ) as demo:

        gr.Markdown("""
        # CV Multitask Pipeline
        **Multi-task object classification and detection**
        Upload any image containing a PASCAL VOC object to get:
        - Object class prediction (20 classes)
        - Bounding box localization
        """)

        # API status
        with gr.Row():
            api_status = gr.Textbox(
                label="API Status",
                value=check_api_health(),
                interactive=False,
            )
            refresh_btn = gr.Button("Refresh Status", scale=0)

        with gr.Row():
            # left column — inputs
            with gr.Column():
                input_image = gr.Image(
                    type="pil",
                    label="Input Image",
                )
                model_version = gr.Radio(
                    choices=["v1", "v2"],
                    value="v1",
                    label="Model Version",
                )
                predict_btn = gr.Button(
                    "Run Prediction",
                    variant="primary",
                )

            # right column — outputs
            with gr.Column():
                output_image = gr.Image(
                    type="pil",
                    label="Prediction (with bbox overlay)",
                )
                output_text = gr.Textbox(
                    label="Prediction Details",
                    lines=10,
                )

        # example images from VOC dataset
        gr.Examples(
            examples=[
                ["data/VOCdevkit/VOC2012/JPEGImages/2008_000008.jpg", "v1"],
                ["data/VOCdevkit/VOC2012/JPEGImages/2008_000015.jpg", "v1"],
                ["data/VOCdevkit/VOC2012/JPEGImages/2008_000019.jpg", "v1"],
            ],
            inputs=[input_image, model_version],
        )

        # wire up events
        predict_btn.click(
            fn=predict_image,
            inputs=[input_image, model_version],
            outputs=[output_image, output_text],
        )

        refresh_btn.click(
            fn=check_api_health,
            outputs=api_status,
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )