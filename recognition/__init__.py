"""Face embedding extraction, quality gating and temporal fusion."""

from recognition.backbones import AdaFaceBackbone, build_backbone
from recognition.encoder import (
    AdaFaceEncoder,
    ArcFaceOnnxEncoder,
    Embedding,
    EncoderError,
    FaceEncoder,
    build_encoder,
)
from recognition.fusion import FusedEmbedding, TemporalFusion
from recognition.quality import QualityAssessment, QualityGate

__all__ = [
    "AdaFaceBackbone",
    "AdaFaceEncoder",
    "ArcFaceOnnxEncoder",
    "Embedding",
    "EncoderError",
    "FaceEncoder",
    "FusedEmbedding",
    "QualityAssessment",
    "QualityGate",
    "TemporalFusion",
    "build_backbone",
    "build_encoder",
]