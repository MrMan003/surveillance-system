"""recognition package."""
"""Face embedding extraction."""

from recognition.backbones import AdaFaceBackbone, build_backbone
from recognition.encoder import (
    AdaFaceEncoder,
    ArcFaceOnnxEncoder,
    Embedding,
    EncoderError,
    FaceEncoder,
    build_encoder,
)

__all__ = [
    "AdaFaceBackbone",
    "AdaFaceEncoder",
    "ArcFaceOnnxEncoder",
    "Embedding",
    "EncoderError",
    "FaceEncoder",
    "build_backbone",
    "build_encoder",
]