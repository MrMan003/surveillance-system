"""detection package."""
"""Body and face detection."""

from detection.base import Detector, DetectorError, DetectorStats
from detection.body_detector import BodyDetector

__all__ = ["BodyDetector", "Detector", "DetectorError", "DetectorStats"]