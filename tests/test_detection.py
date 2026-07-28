"""Tests for the detector interface and the YOLOv8s body detector.

Tests split into two groups.  The base-class tests use a fake backend and run
anywhere in milliseconds.  The tests that exercise real YOLOv8s inference are
marked ``weights`` and skip when the checkpoint is absent, so a fresh clone can
run the suite without a 22 MB download.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pytest

from configs import SurveillanceConfig
from detection.base import Detector, DetectorError, DetectorStats
from detection.body_detector import BodyDetector
from utils.types import BoundingBox, Detection

REPO_ROOT = Path(__file__).resolve().parent.parent
YOLO_WEIGHTS = REPO_ROOT / "weights" / "yolov8s.pt"
requires_weights = pytest.mark.skipif(
    not YOLO_WEIGHTS.is_file(),
    reason="yolov8s.pt not in weights/; run scripts/fetch_weights.py",
)


# --------------------------------------------------------------------------- #
# Fake backend
# --------------------------------------------------------------------------- #
class FakeDetector(Detector):
    """A detector that emits a fixed number of boxes without any model.

    Exists so the batching, timing and lifecycle logic in :class:`Detector` can
    be tested independently of any inference backend.
    """

    def __init__(self, config, runtime, per_frame: int = 2, fail: bool = False) -> None:
        super().__init__(config, runtime, name="fake")
        self._per_frame = per_frame
        self._fail = fail
        self.infer_calls: List[int] = []

    def load(self) -> "FakeDetector":
        """Mark the detector as ready."""
        self._loaded = True
        return self

    def close(self) -> None:
        """Mark the detector as released."""
        self._loaded = False

    def _infer(self, frames: Sequence[np.ndarray]) -> List[List[Detection]]:
        """Emit ``per_frame`` synthetic detections for each frame."""
        if self._fail:
            raise RuntimeError("synthetic backend failure")
        self.infer_calls.append(len(frames))
        return [
            [
                Detection(box=BoundingBox(i, i, i + 10, i + 20), score=0.9, class_id=0)
                for i in range(self._per_frame)
            ]
            for _ in frames
        ]


@pytest.fixture()
def config() -> SurveillanceConfig:
    """A default configuration pinned to CPU for deterministic tests."""
    return SurveillanceConfig.from_dict({"runtime": {"device": "cpu"}})


@pytest.fixture()
def blank_frames() -> List[np.ndarray]:
    """Ten blank BGR frames."""
    return [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]


# --------------------------------------------------------------------------- #
# Base class behaviour
# --------------------------------------------------------------------------- #
def test_detect_requires_load(config, blank_frames) -> None:
    """Inference before load() must raise rather than return nothing."""
    detector = FakeDetector(config.detection, config.runtime)
    with pytest.raises(DetectorError, match="not loaded"):
        detector.detect(blank_frames[0])


def test_empty_input_returns_empty(config) -> None:
    """An empty batch must be a no-op, not an error."""
    with FakeDetector(config.detection, config.runtime) as detector:
        assert detector.detect_batch([]) == []


def test_batching_respects_configured_size(config, blank_frames) -> None:
    """Ten frames at batch_size=4 must produce chunks of 4, 4 and 2."""
    config.detection.batch_size = 4
    with FakeDetector(config.detection, config.runtime) as detector:
        results = detector.detect_batch(blank_frames)
    assert len(results) == 10
    assert detector.infer_calls == [4, 4, 2]


def test_output_order_matches_input(config, blank_frames) -> None:
    """Results must align one-to-one with input frames across chunk boundaries."""
    config.detection.batch_size = 3
    with FakeDetector(config.detection, config.runtime, per_frame=5) as detector:
        results = detector.detect_batch(blank_frames)
    assert len(results) == len(blank_frames)
    assert all(len(r) == 5 for r in results)


def test_backend_failure_is_wrapped(config, blank_frames) -> None:
    """Arbitrary backend exceptions must surface as DetectorError."""
    with FakeDetector(config.detection, config.runtime, fail=True) as detector:
        with pytest.raises(DetectorError, match="forward pass failed"):
            detector.detect(blank_frames[0])


def test_length_mismatch_is_detected(config, blank_frames) -> None:
    """A backend returning the wrong number of results must not pass silently."""

    class Truncating(FakeDetector):
        def _infer(self, frames):  # noqa: ANN001, ANN202
            return super()._infer(frames)[:-1]

    with Truncating(config.detection, config.runtime) as detector:
        with pytest.raises(DetectorError, match="returned"):
            detector.detect_batch(blank_frames[:4])


def test_half_is_never_enabled_on_cpu(config) -> None:
    """FP16 must stay off on CPU regardless of the config flag."""
    config.detection.half = True
    detector = FakeDetector(config.detection, config.runtime)
    assert detector.device == "cpu"
    assert detector.half is False


def test_warmup_clears_statistics(config) -> None:
    """Warmup passes must not pollute the measured latencies."""
    with FakeDetector(config.detection, config.runtime) as detector:
        detector.warmup(iterations=3, size=(64, 64))
        assert detector.stats.frames == 0
        assert detector.stats.latencies_ms == []


def test_warmup_requires_load(config) -> None:
    """Warming an unloaded detector must raise."""
    with pytest.raises(DetectorError):
        FakeDetector(config.detection, config.runtime).warmup()


def test_context_manager_manages_lifecycle(config) -> None:
    """Entering must load and leaving must release."""
    detector = FakeDetector(config.detection, config.runtime)
    assert detector.is_loaded is False
    with detector:
        assert detector.is_loaded is True
    assert detector.is_loaded is False


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_stats_empty_before_use() -> None:
    """A report with no observations must say so rather than fabricate numbers."""
    assert DetectorStats(name="x").summary() == {"name": "x", "frames": 0}


def test_stats_accumulate(config, blank_frames) -> None:
    """Counts must track frames, batches and detections."""
    config.detection.batch_size = 5
    with FakeDetector(config.detection, config.runtime, per_frame=3) as detector:
        detector.detect_batch(blank_frames)
        summary = detector.stats.summary()
    assert summary["frames"] == 10
    assert summary["batches"] == 2
    assert summary["detections"] == 30
    assert summary["detections_per_frame"] == pytest.approx(3.0)


def test_stats_report_percentiles_not_just_mean() -> None:
    """Tail latency must be visible; a mean hides a stalling detector."""
    stats = DetectorStats(name="x")
    for value in [10.0] * 99 + [500.0]:
        stats.record(frames=1, detections=1, elapsed_ms=value)
    summary = stats.summary()
    assert summary["latency_ms_p50"] == pytest.approx(10.0)
    assert summary["latency_ms_max"] == pytest.approx(500.0)
    assert summary["latency_ms_p99"] > summary["latency_ms_p50"]


def test_stats_reset(config, blank_frames) -> None:
    """Reset must clear every accumulator."""
    with FakeDetector(config.detection, config.runtime) as detector:
        detector.detect_batch(blank_frames)
        detector.stats.reset()
        assert detector.stats.summary() == {"name": "fake", "frames": 0}


# --------------------------------------------------------------------------- #
# Real YOLOv8s
# --------------------------------------------------------------------------- #
@requires_weights
def test_yolo_loads_and_reports_device(config) -> None:
    """The checkpoint must load and report a resolved device."""
    with BodyDetector(config.detection, config.runtime, config.paths) as detector:
        assert detector.is_loaded is True
        assert detector.device == "cpu"


@requires_weights
def test_yolo_returns_no_detections_on_blank_frame(config) -> None:
    """A blank frame must yield nothing, not spurious boxes."""
    with BodyDetector(config.detection, config.runtime, config.paths) as detector:
        assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []


@requires_weights
def test_yolo_emits_only_person_class(config) -> None:
    """Class filtering must exclude the other 79 COCO classes."""
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    with BodyDetector(config.detection, config.runtime, config.paths) as detector:
        for detection in detector.detect(frame):
            assert detection.class_id == 0


@requires_weights
def test_yolo_scores_are_in_range(config) -> None:
    """Every emitted score must satisfy Detection's own validation."""
    frame = np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    with BodyDetector(config.detection, config.runtime, config.paths) as detector:
        for detection in detector.detect(frame):
            assert 0.0 <= detection.score <= 1.0


@requires_weights
def test_yolo_boxes_lie_inside_the_frame(config) -> None:
    """Coordinates must be in original frame space, not inference space."""
    height, width = 480, 640
    frame = np.random.default_rng(2).integers(0, 255, (height, width, 3), dtype=np.uint8)
    with BodyDetector(config.detection, config.runtime, config.paths) as detector:
        for detection in detector.detect(frame):
            box = detection.box
            assert -1 <= box.x1 < box.x2 <= width + 1
            assert -1 <= box.y1 < box.y2 <= height + 1


@requires_weights
def test_yolo_min_area_filter(config) -> None:
    """Raising min_body_area must not increase the detection count."""
    frame = np.random.default_rng(3).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    with BodyDetector(config.detection, config.runtime, config.paths) as detector:
        baseline = len(detector.detect(frame))

    strict = SurveillanceConfig.from_dict(
        {"runtime": {"device": "cpu"}, "detection": {"min_body_area": 200_000}}
    )
    with BodyDetector(strict.detection, strict.runtime, strict.paths) as detector:
        assert len(detector.detect(frame)) <= baseline


@requires_weights
def test_yolo_batch_matches_single(config) -> None:
    """Batched inference must agree with per-frame inference."""
    frame = np.random.default_rng(4).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    with BodyDetector(config.detection, config.runtime, config.paths) as detector:
        single = detector.detect(frame)
        batched = detector.detect_batch([frame, frame])
    assert len(batched) == 2
    assert len(batched[0]) == len(single)


@requires_weights
def test_yolo_close_is_idempotent(config) -> None:
    """Closing twice must not raise."""
    detector = BodyDetector(config.detection, config.runtime, config.paths).load()
    detector.close()
    detector.close()
    assert detector.is_loaded is False