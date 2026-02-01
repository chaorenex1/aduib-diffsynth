"""
Model configuration management for Gradio UI.

Loads and parses YAML configuration files for model selection.
"""
import logging
from enum import StrEnum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from torch import dtype

logger = logging.getLogger(__name__)

class ModelTorchDtype(StrEnum):
    """Enumeration of supported torch data types."""
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT8 = "float8"

    @classmethod
    def list(cls) -> list[str]:
        """List all torch dtype values."""
        return [item.value for item in cls]

    def get_torch_dtype(self) -> dtype:
        """Get the corresponding torch dtype."""
        import torch
        match self:
            case ModelTorchDtype.FLOAT16:
                return torch.float16
            case ModelTorchDtype.BFLOAT16:
                return torch.bfloat16
            case ModelTorchDtype.FLOAT32:
                return torch.float32
            case ModelTorchDtype.FLOAT8:
                return torch.float8
            case _:
                raise ValueError(f"Unsupported torch dtype: {self.value}")


class PipelineClass(StrEnum):
    """Enumeration of supported pipeline classes."""
    QWEN_IMAGE = "Qwen-Image"
    Z_IMAGE = "Z-Image"
    FLUX2_IMAGE = "FLUX2"

    @classmethod
    def list(cls) -> list[str]:
        """List all pipeline class values."""
        return [item.value for item in cls]

class PipelineType(StrEnum):
    """Enumeration of supported pipeline types."""
    NONE = "none"
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_EDIT = "image_edit"
    IMAGE_TO_IMAGE = "image_to_image"

    @classmethod
    def list(cls) -> list[str]:
        """List all pipeline type values."""
        return [item.value for item in cls]

    def is_pipeline_type(self, pipeline_type: str) -> bool:
        """Check if the current type matches the given pipeline type."""
        return self.value == pipeline_type

class ModelConfig(BaseModel):
    """Single model configuration."""
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Display name")
    description: str = Field(default="", description="Model description")
    pipeline: str = Field(default="", description="Pipeline type")
    d_type: ModelTorchDtype = Field(default=ModelTorchDtype.FLOAT16, description="Torch data type")
    text_encoder: str = Field(default="", description="Text encoder type")
    vae_encoder: str = Field(default="", description="VAE encoder type")
    tokenizer: str = Field(default="", description="Tokenizer type")
    provider: str = Field(default="", description="Model provider")
    default_resolution: list[int] = Field(default_factory=lambda: [1024, 1024], description="Default resolution")
    default_steps: int = Field(default=50, description="Default inference steps")
    default_guidance: float = Field(default=7.5, description="Default guidance scale")
    supports_lora: bool = Field(default=False, description="Supports LoRA adapters")
    supports_offload: bool = Field(default=False, description="Supports VRAM offload")
    tags: list[str] = Field(default_factory=list, description="Model tags")
    requires_input: bool = Field(default=False, description="Requires input image for editing")

    @property
    def choice_id(self) -> str:
        """Get the ID used for dropdown choices."""
        return self.id

    @property
    def choice_name(self) -> str:
        """Get the name used for dropdown display."""
        return self.name


class ModelCategory(BaseModel):
    """Category containing multiple models."""
    name: str = Field(..., description="Category name")
    description: str = Field(default="", description="Category description")
    models: list[ModelConfig] = Field(default_factory=list, description="Models in this category")

    def get_model_by_id(self, model_id: str) -> Optional[ModelConfig]:
        """Find a model by its ID."""
        for model in self.models:
            if model.id == model_id:
                return model
        return None

    def get_model_choices(self) -> list[tuple[str, str]]:
        """Get (name, id) pairs for Gradio dropdown."""
        return [(model.name, model.id) for model in self.models]


class ModelDefaults(BaseModel):
    """Default model selections."""
    text_to_image: str = "Z-Image-Turbo"
    image_edit: str = "Qwen-Image-Edit-2511"
    lora: str = "none"
    ocr: str = "paddleocr"
    asr: str = "whisper"
    tts: str = "edge-tts"


class ModelRegistry(BaseModel):
    """
    Central registry for all model configurations.

    Loaded from YAML configuration files.
    """
    categories: dict[str, ModelCategory] = Field(default_factory=dict, description="Model categories")
    defaults: ModelDefaults = Field(default_factory=ModelDefaults, description="Default selections")

    def get_category(self, category_name: str) -> Optional[ModelCategory]:
        """Get a category by name."""
        return self.categories.get(category_name)

    def get_model(self, category_name: str, model_id: str) -> Optional[ModelConfig]:
        """Get a specific model from a category."""
        category = self.get_category(category_name)
        if category:
            return category.get_model_by_id(model_id)
        return None

    def get_text_to_image_choices(self) -> list[tuple[str, str]]:
        """Get choices for text-to-image dropdown."""
        category = self.get_category("text_to_image")
        if category:
            return category.get_model_choices()
        return []

    def get_image_edit_choices(self) -> list[tuple[str, str]]:
        """Get choices for image-edit dropdown."""
        category = self.get_category("image_edit")
        if category:
            return category.get_model_choices()
        return []

    def get_lora_choices(self) -> list[tuple[str, str]]:
        """Get choices for LoRA dropdown."""
        category = self.get_category("lora")
        if category:
            return category.get_model_choices()
        return []

    def get_ocr_choices(self) -> list[tuple[str, str]]:
        """Get choices for OCR dropdown."""
        category = self.get_category("ocr")
        if category:
            return category.get_model_choices()
        return []

    def get_asr_choices(self) -> list[tuple[str, str]]:
        """Get choices for ASR dropdown."""
        category = self.get_category("asr")
        if category:
            return category.get_model_choices()
        return []

    def get_tts_choices(self) -> list[tuple[str, str]]:
        """Get choices for TTS dropdown."""
        category = self.get_category("tts")
        if category:
            return category.get_model_choices()
        return []

    def get_default_model(self, category: str) -> str:
        """Get default model ID for a category."""
        if category == "text_to_image":
            return self.defaults.text_to_image
        elif category == "image_edit":
            return self.defaults.image_edit
        elif category == "lora":
            return self.defaults.lora
        elif category == "ocr":
            return self.defaults.ocr
        elif category == "asr":
            return self.defaults.asr
        elif category == "tts":
            return self.defaults.tts
        return ""


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_model_config_path() -> Path:
    """Get the path to the model configuration file."""
    # Try to find from current directory or package directory
    possible_paths = [
        Path("configs/models/models.yaml"),
        Path(__file__).parent / "models.yaml",
    ]

    # Also check if running from different directory
    module_dir = Path(__file__).parent
    possible_paths.append(module_dir / "models.yaml")

    for path in possible_paths:
        if path.exists():
            logger.debug(f"Found model config at: {path}")
            return path

    # Default path
    default_path = Path(__file__).parent / "models.yaml"
    logger.warning(f"Model config file not found, using default: {default_path}")
    return default_path


def load_model_registry(config_path: Optional[Path] = None) -> ModelRegistry:
    """
    Load model registry from YAML configuration file.

    Args:
        config_path: Optional path to config file. If None, uses default location.

    Returns:
        ModelRegistry instance with all loaded configurations.
    """
    global _registry

    if _registry is not None:
        return _registry

    if config_path is None:
        config_path = get_model_config_path()

    if not config_path.exists():
        logger.error(f"Model config file not found: {config_path}")
        # Return empty registry
        return ModelRegistry(categories={}, defaults=ModelDefaults())

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Parse categories
        categories = {}
        for cat_name, cat_data in data.get("categories", {}).items():
            models = []
            for model_data in cat_data.get("models", []):
                models.append(ModelConfig(**model_data))
            categories[cat_name] = ModelCategory(
                name=cat_name,
                description=cat_data.get("description", ""),
                models=models,
            )

        # Parse defaults
        defaults_data = data.get("defaults", {})
        defaults = ModelDefaults(**defaults_data)

        _registry = ModelRegistry(
            categories=categories,
            defaults=defaults,
        )

        logger.info(f"Loaded model registry from {config_path}: "
                   f"{len(categories)} categories, "
                   f"{sum(len(c.models) for c in categories.values())} models")

        return _registry

    except Exception as e:
        logger.error(f"Failed to load model config from {config_path}: {e}", exc_info=True)
        return ModelRegistry(categories={}, defaults=ModelDefaults())


def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry instance.

    Loads on first call and caches the result.
    """
    global _registry
    if _registry is None:
        _registry = load_model_registry()
    return _registry


def reload_model_registry() -> ModelRegistry:
    """
    Force reload the model registry from disk.

    Useful when config files change at runtime.
    """
    global _registry
    _registry = None
    return load_model_registry()


# Convenience functions for Gradio
def get_text_to_image_choices() -> list[tuple[str, str]]:
    """Get text-to-image model choices for Gradio dropdown."""
    return get_model_registry().get_text_to_image_choices()


def get_image_edit_choices() -> list[tuple[str, str]]:
    """Get image-edit model choices for Gradio dropdown."""
    return get_model_registry().get_image_edit_choices()


def get_lora_choices() -> list[tuple[str, str]]:
    """Get LoRA model choices for Gradio dropdown."""
    return get_model_registry().get_lora_choices()


def get_ocr_choices() -> list[tuple[str, str]]:
    """Get OCR model choices for Gradio dropdown."""
    return get_model_registry().get_ocr_choices()


def get_asr_choices() -> list[tuple[str, str]]:
    """Get ASR model choices for Gradio dropdown."""
    return get_model_registry().get_asr_choices()


def get_tts_choices() -> list[tuple[str, str]]:
    """Get TTS model choices for Gradio dropdown."""
    return get_model_registry().get_tts_choices()


def get_default_text_to_image_model() -> str:
    """Get default text-to-image model ID."""
    return get_model_registry().get_default_model("text_to_image")


def get_default_image_edit_model() -> str:
    """Get default image-edit model ID."""
    return get_model_registry().get_default_model("image_edit")


def get_default_lora_model() -> str:
    """Get default LoRA model ID."""
    return get_model_registry().get_default_model("lora")


def get_default_ocr_model() -> str:
    """Get default OCR model ID."""
    return get_model_registry().get_default_model("ocr")


def get_default_asr_model() -> str:
    """Get default ASR model ID."""
    return get_model_registry().get_default_model("asr")


def get_default_tts_model() -> str:
    """Get default TTS model ID."""
    return get_model_registry().get_default_model("tts")


def get_model_config(category: str, model_id: str) -> Optional[ModelConfig]:
    """Get configuration for a specific model."""
    return get_model_registry().get_model(category, model_id)


__all__ = [
    "ModelConfig",
    "ModelCategory",
    "ModelRegistry",
    "ModelDefaults",
    "get_model_config_path",
    "load_model_registry",
    "get_model_registry",
    "reload_model_registry",
    "get_text_to_image_choices",
    "get_image_edit_choices",
    "get_lora_choices",
    "get_ocr_choices",
    "get_asr_choices",
    "get_tts_choices",
    "get_default_text_to_image_model",
    "get_default_image_edit_model",
    "get_default_lora_model",
    "get_default_ocr_model",
    "get_default_asr_model",
    "get_default_tts_model",
    "get_model_config",
]
