import json
from pathlib import Path

# Constants moved from learner.py
# should be environment variable or something ...
R49DIR = Path("/Users/boser/Documents/personal/iot/rail49")

MODELS_DIR = R49DIR / "classifier/models"
DATA_DIR = R49DIR / "datasets/train-track/r49"

VALID_PCT = 0.25

class LearnerConfig:
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model_dir = MODELS_DIR / model_name
        if not self._model_dir.exists():
            raise ValueError(f"Model directory '{self._model_dir}' does not exist.")
            
        config_path = self._model_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                self._config = json.load(f)
        else:
            raise ValueError(f"No or invalid configuration {self._model_dir / 'config.json'}.")

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def config(self):
        return self._config

    @property
    def size(self):
        return self._config.get("size", 96)

    @property
    def dpt(self):
        return self._config.get("dpt", 30)

    @property
    def labels(self):
        return self._config.get("labels", ["track", "train", "other"])

    @property
    def batch_size(self):
        return self._config.get("batch_size", 64)

    @property
    def arch_name(self):
        return self._config.get("model", self._model_name)
    
    @property
    def data_dir(self):
        return DATA_DIR

    @property
    def valid_pct(self):
        return VALID_PCT

    def get_architecture(self, model_name: str | None = None):
        """
        Resolves the model architecture. 
        If model_name is not provided, uses the one from config or init.
        """
        model = model_name or self.arch_name
        
        # Imports needed for resolution
        import torchvision.models as tvm
        
        # Handle prefixes
        if ":" in model:
            prefix, name = model.split(":", 1)
            if prefix == "fastai":
                # fastai models are usually functions in globals or torchvision
                model = name # Fallthrough to lookup
            elif prefix == "timm":
                # vision_learner handles timm strings directly
                return name
            else:
                raise ValueError(f"Unknown prefix '{prefix}' in model name '{model}'.")
        
        # Check globals of the caller or common locations? 
        # Since we moved this, we can't easily access 'globals()' of learner.py unless passed.
        # But commonly used fastai arches are in torchvision.models or fastai.vision.all
        
        # fastai.vision.all exports (like resnet18, resnet34) are actually from torchvision usually
        if hasattr(tvm, model):
            return getattr(tvm, model)
            
        # If it was in fastai globals, we might miss it here if not in tvm.
        # But standard fastai workflow usually uses tvm models or timm.
        
        # Fallback: check if it's available in current global scope (unlikely to work well across files)
        # or just return string for fastai to handle if it can. 
        # fastai `vision_learner` accepts a string for timm, or a function/class for pytorch models.
        
        # If we can't find it in tvm, maybe it's a fastai specific one (like xresnet).
        import fastai.vision.all as fastai_vision
        if hasattr(fastai_vision, model):
             return getattr(fastai_vision, model)

        raise ValueError(f"Architecture '{model}' not found in torchvision or fastai.")
