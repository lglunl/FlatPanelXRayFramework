"""内置模型"""

from .registry import (
    MODEL_REGISTRY,
    register_model,
    discover,
    list_models,
    get_model,
    model_info,
    add_external_model,
    list_external_models,
)

__all__ = [
    "MODEL_REGISTRY",
    "register_model",
    "discover",
    "list_models",
    "get_model",
    "model_info",
    "add_external_model",
    "list_external_models",
]
