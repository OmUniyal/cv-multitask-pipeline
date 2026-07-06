import base64
import io
import time
import uuid
from typing import Optional
import torch
from PIL import Image
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
    BoundingBox,
)
from src.data.transforms import get_val_transforms


router = APIRouter()


def decode_base64_image(image_base64: str) -> Image.Image:
    """
    Decode a base64 string to a PIL Image.
    Raises HTTPException if decoding fails.
    """
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image data: {str(e)}"
        )


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Apply val transforms and add batch dimension.
    Returns tensor [1, 3, 224, 224].
    """
    transforms = get_val_transforms(size=224)
    tensor, _ = transforms(image, [0.0, 0.0, 1.0, 1.0])
    return tensor.unsqueeze(0)


def run_inference(model, image_tensor, device, version):
    """
    Run a single forward pass and return a PredictionResponse.
    """
    from src.data.utils import IDX_TO_CLASS

    t0 = time.time()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        cls_logits, bbox_pred = model(image_tensor)

    probs = torch.softmax(cls_logits, dim=1)
    confidence, class_idx = probs.max(dim=1)

    inference_time_ms = (time.time() - t0) * 1000

    bbox = bbox_pred[0].tolist()

    return PredictionResponse(
        class_name=IDX_TO_CLASS[class_idx.item()],
        class_idx=class_idx.item(),
        confidence=round(confidence.item(), 4),
        bbox=BoundingBox(
            x_min=bbox[0],
            y_min=bbox[1],
            x_max=bbox[2],
            y_max=bbox[3],
        ),
        model_version=version,
        inference_time_ms=round(inference_time_ms, 2),
    )


@router.get("/health", response_model=HealthResponse)
async def health_check(request=None):
    """
    Liveness check — is the server process alive?
    Always returns 200 if the process is running.
    """
    from src.api.main import model_manager
    return HealthResponse(
        status="healthy",
        model_loaded=len(model_manager.get_loaded_versions()) > 0,
        available_versions=model_manager.get_available_versions(),
    )


@router.get("/ready")
async def readiness_check():
    """
    Readiness check — is the model loaded and ready to serve?
    Returns 503 if model not yet loaded.
    """
    from src.api.main import model_manager
    loaded = model_manager.get_loaded_versions()
    if not loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "loaded_versions": []}
        )
    return {"status": "ready", "loaded_versions": loaded}


@router.post("/v1/predict", response_model=PredictionResponse)
async def predict_v1(request: PredictionRequest):
    """
    Single image prediction using model v1.
    Accepts base64-encoded image, returns class + bbox.
    """
    from src.api.main import model_manager, device

    model = model_manager.get_model("v1")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model v1 not loaded"
        )

    image = decode_base64_image(request.image_base64)
    image_tensor = preprocess_image(image)
    return run_inference(model, image_tensor, device, "v1")


@router.post("/v2/predict", response_model=PredictionResponse)
async def predict_v2(request: PredictionRequest):
    """
    Single image prediction using model v2.
    Returns 503 if v2 not available.
    """
    from src.api.main import model_manager, device

    model = model_manager.get_model("v2")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model v2 not loaded or not available"
        )

    image = decode_base64_image(request.image_base64)
    image_tensor = preprocess_image(image)
    return run_inference(model, image_tensor, device, "v2")


@router.post("/v1/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_v1(request: BatchPredictionRequest):
    """
    Batch prediction — multiple images in one call.
    More efficient than calling /v1/predict repeatedly.
    """
    from src.api.main import model_manager, device

    model = model_manager.get_model("v1")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model v1 not loaded"
        )

    t0 = time.time()
    predictions = []

    for image_base64 in request.images_base64:
        image = decode_base64_image(image_base64)
        image_tensor = preprocess_image(image)
        pred = run_inference(model, image_tensor, device, "v1")
        predictions.append(pred)

    total_time = (time.time() - t0) * 1000

    return BatchPredictionResponse(
        predictions=predictions,
        total_inference_time_ms=round(total_time, 2),
        batch_size=len(predictions),
    )