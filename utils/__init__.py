"""Shared types, geometry helpers and logging for the surveillance system."""

from utils.log import AuditLogger, ColourFormatter, get_logger, setup_logging
from utils.profiling import PipelineProfiler, StageStats
from utils.types import (
    BoundingBox,
    Detection,
    FaceDetection,
    TrackState,
    box_areas,
    box_centres,
    boxes_to_array,
    clip_boxes,
    containment_matrix,
    pairwise_iou,
    scale_boxes,
)

__all__ = [
    "AuditLogger",
    "BoundingBox",
    "ColourFormatter",
    "Detection",
    "PipelineProfiler",
    "StageStats",
    "FaceDetection",
    "TrackState",
    "box_areas",
    "box_centres",
    "boxes_to_array",
    "clip_boxes",
    "containment_matrix",
    "get_logger",
    "pairwise_iou",
    "scale_boxes",
    "setup_logging",
]