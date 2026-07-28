"""Configuration package for the surveillance system.

Re-exports the public configuration API so callers can write
``from configs import SurveillanceConfig`` instead of reaching into
``configs.config``.
"""

from configs.config import (
    ARCFACE_TEMPLATE_112,
    AlignmentConfig,
    AssociationConfig,
    ConfigError,
    DetectionConfig,
    FusionConfig,
    FusionStrategy,
    GovernanceConfig,
    IndexType,
    InterpolationMode,
    PathConfig,
    QualityConfig,
    RecognitionConfig,
    RenderingConfig,
    RuntimeConfig,
    SearchConfig,
    SurveillanceConfig,
    ThreadType,
    TrackingConfig,
    VideoConfig,
    get_config,
    set_config,
)

__all__ = [
    "ARCFACE_TEMPLATE_112",
    "AlignmentConfig",
    "AssociationConfig",
    "ConfigError",
    "DetectionConfig",
    "FusionConfig",
    "FusionStrategy",
    "GovernanceConfig",
    "IndexType",
    "InterpolationMode",
    "PathConfig",
    "QualityConfig",
    "RecognitionConfig",
    "RenderingConfig",
    "RuntimeConfig",
    "SearchConfig",
    "SurveillanceConfig",
    "ThreadType",
    "TrackingConfig",
    "VideoConfig",
    "get_config",
    "set_config",
]

__version__ = "0.1.0"