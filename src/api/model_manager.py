import json
import time
import torch
from pathlib import Path
from typing import Dict, Optional
from src.models.multitask_model import MultiTaskModel


class ModelManager:
    """
    Manages loading, caching, and serving of versioned models.

    Registry pattern:
        models/registry.json maps version -> checkpoint filename
        {
            "v1": "best_model.pt",
            "v2": null
        }

    Models are loaded once and cached in memory — no disk reads
    on every request.
    """

    def __init__(
        self,
        registry_path: str = "models/registry.json",
        checkpoint_dir: str = "models",
    ):
        self.registry_path = Path(registry_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # cache: version -> loaded model
        self._models: Dict[str, MultiTaskModel] = {}
        self._load_times: Dict[str, float] = {}

        self._registry = self._read_registry()

    def _read_registry(self) -> Dict:
        """Read registry.json — source of truth for version -> checkpoint."""
        if not self.registry_path.exists():
            return {}
        with open(self.registry_path) as f:
            return json.load(f)

    def load_version(self, version: str) -> bool:
        """
        Load a model version into memory cache.
        Returns True if successful, False if checkpoint not found.
        """
        if version in self._models:
            print(f"Model {version} already loaded.")
            return True

        checkpoint_file = self._registry.get(version)
        if not checkpoint_file:
            print(f"No checkpoint registered for version {version}.")
            return False

        checkpoint_path = self.checkpoint_dir / checkpoint_file
        if not checkpoint_path.exists():
            print(f"Checkpoint file not found: {checkpoint_path}")
            return False

        print(f"Loading model {version} from {checkpoint_path}...")
        t0 = time.time()

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        model = MultiTaskModel()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self._models[version] = model
        self._load_times[version] = time.time() - t0
        print(f"Model {version} loaded in {self._load_times[version]:.2f}s")
        return True

    def get_model(self, version: str) -> Optional[MultiTaskModel]:
        """
        Get a loaded model by version.
        Returns None if version not loaded.
        """
        return self._models.get(version)

    def get_available_versions(self):
        """Return list of versions that have checkpoints registered."""
        return [v for v, f in self._registry.items() if f is not None]

    def get_loaded_versions(self):
        """Return list of versions currently in memory cache."""
        return list(self._models.keys())

    def is_loaded(self, version: str) -> bool:
        return version in self._models

    def reload_registry(self):
        """
        Re-read registry.json without restarting the server.
        Useful when a new checkpoint is added during a running session.
        """
        self._registry = self._read_registry()
        print("Registry reloaded.")

    def unload_version(self, version: str):
        """Remove a model from memory cache to free up RAM/VRAM."""
        if version in self._models:
            del self._models[version]
            torch.cuda.empty_cache()
            print(f"Model {version} unloaded.")