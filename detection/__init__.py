"""Body and face detection."""

from detection.base import Detector, DetectorError, DetectorStats
from detection.body_detector import BodyDetector
from detection.combined import CombinedDetector, FrameDetections
from detection.face_detector import FaceDetector

__all__ = [
    "BodyDetector",
    "CombinedDetector",
    "Detector",
    "DetectorError",
    "DetectorStats",
    "FaceDetector",
    "FrameDetections",
]