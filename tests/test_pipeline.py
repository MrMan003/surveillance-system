"""Tests for profiling and end-to-end pipeline integration."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from configs import SurveillanceConfig
from utils.profiling import PipelineProfiler, StageStats

REPO_ROOT = Path(__file__).resolve().parent.parent
YOLO_WEIGHTS = REPO_ROOT / "weights" / "yolov8s.pt"
SCRFD_WEIGHTS = REPO_ROOT / "weights" / "models" / "buffalo_l" / "det_10g.onnx"

requires_models = pytest.mark.skipif(
    not (YOLO_WEIGHTS.is_file() and SCRFD_WEIGHTS.is_file()),
    reason="detector weights not present in weights/",
)


# --------------------------------------------------------------------------- #
# StageStats
# --------------------------------------------------------------------------- #
def test_empty_stage_reports_nothing() -> None:
    """A stage that never ran must say so rather than fabricate numbers."""
    assert StageStats(name="x").summary() == {"name": "x", "calls": 0}


def test_stage_reports_percentiles() -> None:
    """Tail latency must be visible; a mean hides an occasional stall."""
    stats = StageStats(name="x")
    for value in [10.0] * 99 + [500.0]:
        stats.record(value)

    summary = stats.summary()
    assert summary["p50_ms"] == pytest.approx(10.0)
    assert summary["max_ms"] == pytest.approx(500.0)
    assert summary["p99_ms"] > summary["p50_ms"]


def test_per_item_cost() -> None:
    """Batched stages must report cost per item, not per call."""
    stats = StageStats(name="x")
    stats.record(100.0, count=10)
    assert stats.summary()["ms_per_item"] == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Profiler
# --------------------------------------------------------------------------- #
def test_profiler_times_a_block() -> None:
    """The context manager must record real elapsed time."""
    profiler = PipelineProfiler(enabled=True)
    profiler.start()
    with profiler.stage("work"):
        time.sleep(0.01)
    profiler.finish()

    stage = profiler.summary()["stages"][0]
    assert stage["name"] == "work"
    assert stage["p50_ms"] >= 9.0


def test_disabled_profiler_records_nothing() -> None:
    """Instrumentation must be free to leave in when profiling is off."""
    profiler = PipelineProfiler(enabled=False)
    profiler.start()
    with profiler.stage("work"):
        pass
    profiler.record("other", 100.0)
    assert profiler.summary()["stages"] == []


def test_stage_records_even_when_the_block_raises() -> None:
    """A failing stage must still be attributed, not vanish from the profile."""
    profiler = PipelineProfiler(enabled=True)
    profiler.start()
    with pytest.raises(ValueError):
        with profiler.stage("failing"):
            raise ValueError("boom")
    assert profiler.summary()["stages"][0]["name"] == "failing"


def test_stages_sorted_by_total_cost() -> None:
    """The most expensive stage must come first; that is what gets optimised."""
    profiler = PipelineProfiler(enabled=True)
    profiler.start()
    profiler.record("cheap", 1.0)
    profiler.record("expensive", 100.0)
    profiler.record("medium", 10.0)
    profiler.finish()

    names = [stage["name"] for stage in profiler.summary()["stages"]]
    assert names == ["expensive", "medium", "cheap"]


def test_unaccounted_time_is_exposed() -> None:
    """The gap between measured stages and wall clock must be visible.

    A large gap means the instrumentation is missing something real, which is
    how the missing decode timer was found.
    """
    profiler = PipelineProfiler(enabled=True)
    profiler.start()
    with profiler.stage("measured"):
        time.sleep(0.01)
    time.sleep(0.02)
    profiler.finish()

    assert profiler.summary()["unaccounted_ms"] > 5.0


def test_throughput_is_computed() -> None:
    """Frames per second must derive from wall clock, not stage sums."""
    profiler = PipelineProfiler(enabled=True)
    profiler.start()
    profiler.count_frame(10)
    time.sleep(0.05)
    profiler.finish()

    summary = profiler.summary()
    assert summary["frames"] == 10
    assert 0 < summary["fps"] < 1000


def test_report_is_renderable() -> None:
    """The text report must render without a terminal or a GPU."""
    profiler = PipelineProfiler(enabled=True)
    profiler.start()
    profiler.count_frame(5)
    profiler.record("detect", 50.0)
    profiler.finish()

    report = profiler.report()
    assert "detect" in report
    assert "unaccounted" in report


def test_empty_report_says_so() -> None:
    """A profiler with no frames must not print an empty table."""
    assert "No frames" in PipelineProfiler(enabled=True).report()


def test_memory_report_on_cpu() -> None:
    """Host memory must be reported without requiring CUDA."""
    memory = PipelineProfiler(enabled=True, device="cpu").memory()
    assert "cuda_peak_gib" not in memory


def test_reset_clears_measurements() -> None:
    """Reset must return the profiler to its initial state."""
    profiler = PipelineProfiler(enabled=True)
    profiler.start()
    profiler.count_frame(3)
    profiler.record("x", 1.0)
    profiler.reset()

    assert profiler.frames == 0
    assert profiler.summary()["stages"] == []


# --------------------------------------------------------------------------- #
# Pipeline integration
# --------------------------------------------------------------------------- #
@pytest.fixture()
def sample_video(tmp_path_factory) -> Path:
    """A short variable-frame-rate clip for integration tests."""
    pytest.importorskip("av")
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.make_test_video import write_clip

    directory = tmp_path_factory.mktemp("pipeline")
    return write_clip(directory / "vfr.mp4", "vfr", 1.0, 320, 240)


@requires_models
def test_pipeline_runs_without_a_gallery(sample_video, tmp_path) -> None:
    """Detection, tracking and rendering must work with no enrolled identities.

    This is the enrolment-capture and stage-tuning mode.
    """
    from pipeline import SurveillancePipeline

    config = SurveillanceConfig.from_dict(
        {
            "runtime": {"device": "cpu", "profile": True},
            "video": {"max_frames": 4},
            "paths": {"root": str(tmp_path)},
        }
    )
    output = tmp_path / "out.mp4"
    result = SurveillancePipeline(config, gallery=None).run(sample_video, output)

    assert result.frames_processed == 4
    assert result.identifications == {}
    assert output.is_file()


@requires_models
def test_pipeline_preserves_source_timestamps(sample_video, tmp_path) -> None:
    """The rendered output's timeline must match the source exactly.

    Every earlier phase is undermined if the artefact people actually watch is
    re-timed at a fixed frame rate.
    """
    from configs import VideoConfig
    from datasets.frame_stream import FrameStream
    from pipeline import SurveillancePipeline

    config = SurveillanceConfig.from_dict(
        {
            "runtime": {"device": "cpu"},
            "video": {"max_frames": 6},
            "paths": {"root": str(tmp_path)},
        }
    )
    output = tmp_path / "out.mp4"
    SurveillancePipeline(config, gallery=None).run(sample_video, output)

    with FrameStream(VideoConfig(input_path=sample_video, max_frames=6)) as stream:
        source_pts = [frame.pts for frame in stream]
    with FrameStream(VideoConfig(input_path=output)) as stream:
        output_pts = [frame.pts for frame in stream]

    assert output_pts == source_pts


@requires_models
def test_pipeline_manifest_is_serialisable(sample_video, tmp_path) -> None:
    """The run manifest must be valid JSON with the expected fields."""
    from pipeline import SurveillancePipeline

    config = SurveillanceConfig.from_dict(
        {
            "runtime": {"device": "cpu"},
            "video": {"max_frames": 3},
            "paths": {"root": str(tmp_path)},
        }
    )
    result = SurveillancePipeline(config, gallery=None).run(sample_video, None)
    manifest = result.save_manifest(tmp_path / "run.json")

    payload = json.loads(manifest.read_text())
    assert payload["frames_processed"] == 3
    assert "profile" in payload
    assert payload["stream"]["codec"] == "h264"


@requires_models
def test_pipeline_profiles_every_stage(sample_video, tmp_path) -> None:
    """Decode and startup must be attributed, not left as unaccounted time."""
    from pipeline import SurveillancePipeline

    config = SurveillanceConfig.from_dict(
        {
            "runtime": {"device": "cpu", "profile": True},
            "video": {"max_frames": 3},
            "paths": {"root": str(tmp_path)},
        }
    )
    pipeline = SurveillancePipeline(config, gallery=None)
    pipeline.run(sample_video, None)

    names = {stage["name"] for stage in pipeline.profiler.summary()["stages"]}
    assert {"startup", "decode", "detect", "track", "associate"} <= names


@requires_models
def test_rendering_can_be_skipped(sample_video, tmp_path) -> None:
    """Skipping the render must not produce a file or a write stage."""
    from pipeline import SurveillancePipeline

    config = SurveillanceConfig.from_dict(
        {
            "runtime": {"device": "cpu"},
            "video": {"max_frames": 3},
            "paths": {"root": str(tmp_path)},
        }
    )
    pipeline = SurveillancePipeline(config, gallery=None)
    result = pipeline.run(sample_video, None)

    assert result.output_video is None
    names = {stage["name"] for stage in pipeline.profiler.summary()["stages"]}
    assert "write" not in names


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_rejects_missing_input(tmp_path) -> None:
    """A nonexistent input must exit non-zero rather than raise."""
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from main import main

    assert main(["--input", str(tmp_path / "nope.mp4"), "--no-render"]) == 2


def test_cli_parser_accepts_documented_flags() -> None:
    """Every flag in the module docstring must actually parse."""
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from main import build_parser

    args = build_parser().parse_args(
        [
            "--input", "a.mkv",
            "--output", "b.mp4",
            "--gallery", "g",
            "--manifest", "m.json",
            "--max-frames", "100",
            "--stride", "2",
            "--start", "1.5",
            "--end", "9.0",
            "--device", "cpu",
            "--no-render",
            "--no-audit",
            "--allow-vfr-estimate",
        ]
    )
    assert args.max_frames == 100
    assert args.allow_vfr_estimate is True