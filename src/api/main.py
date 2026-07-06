import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.model_manager import ModelManager
from src.api.router import router


# global instances — shared across all requests
model_manager: ModelManager = None
device: torch.device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager — runs startup and shutdown logic.
    Replaces the deprecated @app.on_event("startup") pattern.

    Startup: load models into memory before serving any requests.
    Shutdown: clean up resources.
    """
    global model_manager, device

    # --- startup ---
    print("Starting CV Multitask Pipeline API...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_manager = ModelManager()
    available = model_manager.get_available_versions()
    print(f"Available model versions: {available}")

    # load all available versions on startup
    for version in available:
        success = model_manager.load_version(version)
        if success:
            print(f"Loaded model {version} successfully")
        else:
            print(f"Failed to load model {version}")

    print("API ready to serve requests.")
    yield

    # --- shutdown ---
    print("Shutting down API...")
    for version in model_manager.get_loaded_versions():
        model_manager.unload_version(version)
    print("Shutdown complete.")


app = FastAPI(
    title="CV Multitask Pipeline",
    description=(
        "Multi-task computer vision API — simultaneous object "
        "classification and detection using EfficientNet-B0 backbone "
        "with custom task heads."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allows the Gradio UI and other frontends to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# register all routes
app.include_router(router)


@app.get("/")
async def root():
    return {
        "name": "CV Multitask Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }