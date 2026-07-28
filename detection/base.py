"""Detector interface shared by the body and face detectors.

Every detector in this pipeline is a batch-in, batch-out function from frames
to detections.  Fixing that contract here is what lets the tracker consume
detections without knowing whether they came from YOLO, SCRFD, or something
swapped in later (Dependency Inversion).

Two responsibilities live in the base class rather than in each detector:

Warmup
    The first forward pass through a freshly loaded model is far slower than
    steady state -- CUDA context creation, kernel autotuning, lazy weight
    materialisation.  Benchmarking without warmup measures initialisation and
    calls it inference latency.

Statistics
    Frame counts and latency percentiles are collected uniformly, so the
    Phase 11 profiler does not need per-detector special cases.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import List, Optional, Sequence, Type

import numpy as np

from configs.config import DetectionConfig, PathConfig, RuntimeConfig
from utils.log import get_logger
from utils.types import Detection

__all__ = ["DetectorError", "DetectorStats", "Detector"]

LOGGER = get_logger(__name__)


class DetectorError(RuntimeError):
    """Raised when a detector cannot be loaded or a forward pass fails."""


@dataclass
class DetectorStats:
    """Rolling latency and throughput statistics for one detector.

    Attributes:
        name: Detector identifier used in reports.
        frames: Total frames processed.
        batches: Total forward passes executed.
        detections: Total detections emitted after filtering.
        latencies_ms: Per-batch wall-clock latency in milliseconds.
    """

    name: str
    frames: int = 0
    batches: int = 0
    detections: int = 0
    latencies_ms: List[float] = field(default_factory=list)

    def record(self, frames: int, detections: int, elapsed_ms: float) -> None:
        """Add one batch to the running totals.

        Args:
            frames: Frames in the batch.
            detections: Detections emitted for the batch.
            elapsed_ms: Wall-clock duration of the forward pass.
        """
        self.frames += frames
        self.batches += 1
        self.detections += detections
        self.latencies_ms.append(elapsed_ms)

    def summary(self) -> dict:
        """Summarise throughput and latency.

        Percentiles rather than a bare mean: a detector averaging 30 ms with a
        200 ms p99 will stall a real-time pipeline, and the mean hides that.

        Returns:
            A mapping of statistics; ``frames`` is ``0`` when nothing ran.
        """
        if not self.latencies_ms:
            return {"name": self.name, "frames": 0}

        latencies = np.asarray(self.latencies_ms, dtype=np.float64)
        total_ms = float(latencies.sum())
        return {
            "name": self.name,
            "frames": self.frames,
            "batches": self.batches,
            "detections": self.detections,
            "detections_per_frame": self.detections / max(self.frames, 1),
            "total_seconds": total_ms / 1000.0,
            "fps": self.frames / (total_ms / 1000.0) if total_ms > 0 else 0.0,
            "latency_ms_mean": float(latencies.mean()),
            "latency_ms_p50": float(np.percentile(latencies, 50)),
            "latency_ms_p95": float(np.percentile(latencies, 95)),
            "latency_ms_p99": float(np.percentile(latencies, 99)),
            "latency_ms_max": float(latencies.max()),
        }

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self.frames = 0
        self.batches = 0
        self.detections = 0
        self.latencies_ms.clear()


class Detector(ABC):
    """Base class for batch detectors.

    Subclasses implement :meth:`_infer`; batching, timing, statistics and
    lifecycle are handled here.

    Args:
        config: Detector thresholds and geometry.
        runtime: Device and precision policy.
        paths: Filesystem layout used to resolve weight files.
        name: Identifier used in logs and statistics.
    """

    def __init__(
        self,
        config: DetectionConfig,
        runtime: RuntimeConfig,
        paths: Optional[PathConfig] = None,
        name: str = "detector",
    ) -> None:
        self._config = config
        self._runtime = runtime
        self._paths = paths or PathConfig()
        self._name = name
        self._device = runtime.resolve_device()
        self._half = runtime.use_half(config.half)
        self._loaded = False
        self.stats = DetectorStats(name=name)

        LOGGER.debug(
            "%s configured: device=%s half=%s batch=%d",
            name,
            self._device,
            self._half,
            config.batch_size,
        )

    # -- properties -------------------------------------------------------- #
    @property
    def name(self) -> str:
        """Identifier used in logs and statistics."""
        return self._name

    @property
    def device(self) -> str:
        """Resolved compute device string."""
        return self._device

    @property
    def half(self) -> bool:
        """Whether FP16 inference is active.

        Always ``False`` on CPU: half precision there is both slower and
        numerically worse than FP32.
        """
        return self._half

    @property
    def is_loaded(self) -> bool:
        """Whether model weights are resident and ready."""
        return self._loaded

    # -- lifecycle --------------------------------------------------------- #
    @abstractmethod
    def load(self) -> "Detector":
        """Load model weights onto the target device.

        Returns:
            ``self``, to allow fluent chaining.

        Raises:
            DetectorError: If weights are missing or cannot be loaded.
        """

    @abstractmethod
    def _infer(self, frames: Sequence[np.ndarray]) -> List[List[Detection]]:
        """Run one forward pass over a batch.

        Args:
            frames: BGR ``uint8`` arrays, each shaped ``(H, W, 3)``.

        Returns:
            One detection list per input frame, in the same order.
        """

    @abstractmethod
    def close(self) -> None:
        """Release model resources and free device memory."""

    def __enter__(self) -> "Detector":
        """Load the model on entering a ``with`` block."""
        return self.load()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Release the model on leaving a ``with`` block."""
        self.close()

    # -- inference --------------------------------------------------------- #
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect in a single frame.

        Args:
            frame: BGR ``uint8`` array shaped ``(H, W, 3)``.

        Returns:
            Detections for that frame.
        """
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: Sequence[np.ndarray]) -> List[List[Detection]]:
        """Detect across a sequence of frames, chunked to the configured batch size.

        Args:
            frames: BGR ``uint8`` arrays.

        Returns:
            One detection list per input frame, in input order.

        Raises:
            DetectorError: If the model is not loaded or a forward pass fails.
        """
        if not self._loaded:
            raise DetectorError(f"{self._name} not loaded; call load() or use a with block")
        if not frames:
            return []

        results: List[List[Detection]] = []
        size = self._config.batch_size

        for start in range(0, len(frames), size):
            chunk = list(frames[start : start + size])
            begin = time.perf_counter()
            try:
                chunk_results = self._infer(chunk)
            except DetectorError:
                raise
            except Exception as exc:  # noqa: BLE001 - backends raise widely
                raise DetectorError(f"{self._name} forward pass failed: {exc}") from exc
            elapsed_ms = (time.perf_counter() - begin) * 1000.0

            if len(chunk_results) != len(chunk):
                raise DetectorError(
                    f"{self._name} returned {len(chunk_results)} result(s) "
                    f"for {len(chunk)} frame(s)"
                )

            self.stats.record(
                frames=len(chunk),
                detections=sum(len(r) for r in chunk_results),
                elapsed_ms=elapsed_ms,
            )
            results.extend(chunk_results)

        return results

    def warmup(self, iterations: int = 2, size: Optional[tuple] = None) -> None:
        """Run throwaway passes so later timings measure steady-state inference.

        Args:
            iterations: Number of warmup batches.
            size: ``(height, width)`` for the synthetic frames; defaults to the
                configured inference resolution.
        """
        if not self._loaded:
            raise DetectorError(f"{self._name} not loaded; call load() first")

        height, width = size or (self._config.body_imgsz, self._config.body_imgsz)
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        LOGGER.debug("Warming up %s for %d iteration(s)", self._name, iterations)
        for _ in range(iterations):
            self._infer([blank])
        self.stats.reset()

    # -- weight resolution ------------------------------------------------- #
    def _resolve_weights(self, weights: str) -> str:
        """Resolve a weight reference to a concrete path when one exists locally.

        A bare filename that exists under ``weights_dir`` is returned as an
        absolute path.  Anything else is passed through untouched, so backends
        that download by alias keep working.

        Args:
            weights: Filename, path, or backend-specific alias.

        Returns:
            An absolute path, or the original string.
        """
        candidate = Path(weights)
        if candidate.is_absolute() and candidate.is_file():
            return str(candidate)

        local = self._paths.weights_dir / candidate.name
        if local.is_file():
            LOGGER.debug("Using local weights %s", local)
            return str(local)

        LOGGER.debug("No local copy of %s; delegating to the backend", weights)
        return weights