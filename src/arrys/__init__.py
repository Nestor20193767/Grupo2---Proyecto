"""Utilidades de inferencia y reproducibilidad para ARRYS."""

from .constants import CLASSIFIER_CLASSES, FS, GENERATOR_CLASSES, LATENT_DIM, SEQ_LEN

__all__ = [
    "ArrysInference",
    "FS",
    "SEQ_LEN",
    "LATENT_DIM",
    "GENERATOR_CLASSES",
    "CLASSIFIER_CLASSES",
]
__version__ = "1.0.0"


def __getattr__(name: str):
    """Carga ONNX Runtime solo cuando se solicita la clase de inferencia."""
    if name == "ArrysInference":
        from .inference import ArrysInference

        return ArrysInference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
