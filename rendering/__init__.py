"""rendering package."""
"""Annotated video rendering."""

from rendering.annotator import Annotator, FrameAnnotation
from rendering.writer import AnnotatedVideoWriter, VideoWriterError

__all__ = ["AnnotatedVideoWriter", "Annotator", "FrameAnnotation", "VideoWriterError"]