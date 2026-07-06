from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class PredictionRequest(BaseModel):
    """
    Incoming request for a single image prediction.
    Image is sent as a base64-encoded string so it can
    travel over HTTP as JSON.
    """
    image_base64: str = Field(
        ...,
        description="Base64-encoded JPEG or PNG image"
    )
    model_version: str = Field(
        default="v1",
        description="Model version to use: 'v1' or 'v2'"
    )

    @field_validator("model_version")
    @classmethod
    def validate_version(cls, v):
        if v not in ("v1", "v2"):
            raise ValueError("model_version must be 'v1' or 'v2'")
        return v


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates [0, 1]."""
    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    x_max: float = Field(..., ge=0.0, le=1.0)
    y_max: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """
    Response from a single image prediction.
    Always returns the same structure regardless of model version.
    """
    class_name: str
    class_idx: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    model_version: str
    inference_time_ms: float


class BatchPredictionRequest(BaseModel):
    """
    Request for multiple images in one call.
    Capped at 16 images per batch to prevent overload.
    """
    images_base64: List[str] = Field(
        ...,
        min_length=1,
        max_length=16,
        description="List of base64-encoded images"
    )
    model_version: str = Field(default="v1")

    @field_validator("model_version")
    @classmethod
    def validate_version(cls, v):
        if v not in ("v1", "v2"):
            raise ValueError("model_version must be 'v1' or 'v2'")
        return v


class BatchPredictionResponse(BaseModel):
    """Response for a batch prediction request."""
    predictions: List[PredictionResponse]
    total_inference_time_ms: float
    batch_size: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    available_versions: List[str]


class ErrorResponse(BaseModel):
    """Standard error response structure."""
    error: str
    detail: Optional[str] = None
    status_code: int