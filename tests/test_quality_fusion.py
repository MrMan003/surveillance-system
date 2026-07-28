"""Tests for embedding quality gating and temporal fusion."""

from __future__ import annotations

import numpy as np
import pytest

from alignment.umeyama import AlignedFace
from configs import SurveillanceConfig
from recognition.encoder import Embedding
from recognition.fusion import TemporalFusion
from recognition.quality import QualityGate

cv2 = pytest.importorskip("cv2", reason="OpenCV is required for sharpness measurement")


def unit_vector(seed: int) -> np.ndarray:
    """Produce a deterministic unit vector."""
    vector = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return vector / np.linalg.norm(vector)


def sharp_crop(seed: int = 0) -> np.ndarray:
    """Produce a crop with strong high-frequency content."""
    rng = np.random.default_rng(seed)
    crop = rng.integers(40, 215, (112, 112, 3), dtype=np.uint8)
    cv2.rectangle(crop, (30, 30), (80, 80), (255, 255, 255), 2)
    return crop


def make_aligned(
    image: np.ndarray, scale: float = 0.6, eye_distance: float = 55.0
) -> AlignedFace:
    """Wrap a crop in an AlignedFace with plausible warp geometry."""
    return AlignedFace(
        image=image,
        matrix=np.eye(2, 3, dtype=np.float64),
        scale=scale,
        rotation_degrees=0.0,
        residual=0.1,
        source_eye_distance=eye_distance,
    )


def make_embedding(
    seed: int = 0, norm: float = 32.0, score: float = 0.9, calibrated: bool = True
) -> Embedding:
    """Build an embedding with controllable quality metadata."""
    return Embedding(
        vector=unit_vector(seed),
        norm=norm,
        detection_score=score,
        norm_is_quality_calibrated=calibrated,
    )


@pytest.fixture()
def gate() -> QualityGate:
    """A quality gate on stock thresholds."""
    return QualityGate(SurveillanceConfig.default().quality)


# --------------------------------------------------------------------------- #
# Quality gate
# --------------------------------------------------------------------------- #
def test_good_face_passes(gate: QualityGate) -> None:
    """A sharp, well-lit, fully visible face must pass."""
    assessment = gate.assess(make_embedding(), make_aligned(sharp_crop()))
    assert assessment.passed is True
    assert assessment.reasons == []
    assert 0.0 <= assessment.score <= 1.0


def test_blurred_face_is_rejected(gate: QualityGate) -> None:
    """Motion blur must be caught.

    A blurred embedding is not random; it clusters with other blurred faces,
    so it can match the wrong identity confidently.
    """
    blurred = cv2.GaussianBlur(sharp_crop(), (15, 15), 0)
    assessment = gate.assess(make_embedding(), make_aligned(blurred))
    assert assessment.passed is False
    assert any("blurred" in reason for reason in assessment.reasons)


def test_dark_face_is_rejected(gate: QualityGate) -> None:
    """A crushed crop has lost the local contrast the encoder relies on."""
    dark = (sharp_crop() * 0.05).astype(np.uint8)
    assert gate.assess(make_embedding(), make_aligned(dark)).passed is False


def test_truncated_face_is_rejected(gate: QualityGate) -> None:
    """A partial face embeds half a face: confident, repeatable and wrong."""
    assessment = gate.assess(
        make_embedding(), make_aligned(sharp_crop()), truncation=0.45
    )
    assert assessment.passed is False
    assert any("partial" in reason for reason in assessment.reasons)


def test_tiny_face_is_rejected(gate: QualityGate) -> None:
    """Below roughly 20px inter-eye there is not enough detail to recognise."""
    assessment = gate.assess(
        make_embedding(), make_aligned(sharp_crop(), eye_distance=12.0)
    )
    assert assessment.passed is False
    assert any("too small" in reason for reason in assessment.reasons)


def test_low_detector_confidence_is_rejected(gate: QualityGate) -> None:
    """Weak detections must not reach the gallery."""
    assessment = gate.assess(
        make_embedding(score=0.2), make_aligned(sharp_crop())
    )
    assert assessment.passed is False


def test_low_norm_is_rejected(gate: QualityGate) -> None:
    """A calibrated feature norm below threshold indicates a poor crop."""
    assessment = gate.assess(make_embedding(norm=8.0), make_aligned(sharp_crop()))
    assert assessment.passed is False
    assert any("norm" in reason for reason in assessment.reasons)


def test_all_failures_are_reported(gate: QualityGate) -> None:
    """The gate must list every reason, not just the first."""
    dark_blur = cv2.GaussianBlur((sharp_crop() * 0.05).astype(np.uint8), (15, 15), 0)
    assessment = gate.assess(
        make_embedding(norm=5.0, score=0.1),
        make_aligned(dark_blur, eye_distance=10.0),
        truncation=0.5,
    )
    assert len(assessment.reasons) >= 4


# --------------------------------------------------------------------------- #
# Norm calibration
# --------------------------------------------------------------------------- #
def test_uncalibrated_norm_skips_the_norm_gate(gate: QualityGate) -> None:
    """ArcFace norms must not be read as quality.

    Gating on an uncalibrated norm would threshold noise while appearing to
    work, so the check is skipped entirely and pixel measures decide.
    """
    assessment = gate.assess(
        make_embedding(norm=3.0, calibrated=False), make_aligned(sharp_crop())
    )
    assert assessment.passed is True
    assert not any("norm" in reason for reason in assessment.reasons)


def test_uncalibrated_norms_do_not_calibrate_the_gate(gate: QualityGate) -> None:
    """Uncalibrated norms must not enter the adaptive history."""
    for index in range(100):
        gate.assess(
            make_embedding(seed=index, calibrated=False), make_aligned(sharp_crop())
        )
    assert gate.samples_seen == 0
    assert gate.is_calibrated is False


# --------------------------------------------------------------------------- #
# Adaptive thresholding
# --------------------------------------------------------------------------- #
def test_threshold_starts_at_the_configured_floor(gate: QualityGate) -> None:
    """Before warmup the configured minimum applies."""
    assert gate.is_calibrated is False
    assert gate.norm_threshold == pytest.approx(
        SurveillanceConfig.default().quality.min_norm
    )


def test_threshold_adapts_after_warmup(gate: QualityGate) -> None:
    """The gate must calibrate to the norms it actually observes.

    A fixed threshold is wrong for any specific camera: norm distributions
    shift with lens, lighting and typical subject distance.
    """
    rng = np.random.default_rng(0)
    for index in range(100):
        gate.assess(
            make_embedding(seed=index, norm=float(28 + rng.normal(0, 4))),
            make_aligned(sharp_crop()),
        )
    assert gate.is_calibrated is True
    assert gate.norm_threshold > SurveillanceConfig.default().quality.min_norm


def test_threshold_never_falls_below_the_floor() -> None:
    """A stream of uniformly poor faces must not calibrate the gate down."""
    config = SurveillanceConfig.default().quality
    gate = QualityGate(config)
    for index in range(100):
        gate.assess(make_embedding(seed=index, norm=2.0), make_aligned(sharp_crop()))
    assert gate.norm_threshold >= config.min_norm


def test_adaptive_can_be_disabled() -> None:
    """With adaptation off the threshold must stay fixed."""
    config = SurveillanceConfig.from_dict({"quality": {"adaptive": False}}).quality
    gate = QualityGate(config)
    for index in range(100):
        gate.assess(make_embedding(seed=index, norm=40.0), make_aligned(sharp_crop()))
    assert gate.norm_threshold == pytest.approx(config.min_norm)


def test_upsampled_crops_get_a_relaxed_sharpness_threshold(gate: QualityGate) -> None:
    """Interpolation is a low-pass filter, so upsampled crops measure blurrier.

    Without this adjustment the gate rejects small faces for being blurry when
    they are merely small, which the scale term already penalises.
    """
    crop = sharp_crop()
    native = gate.assess(make_embedding(), make_aligned(crop, scale=1.0))
    upsampled = gate.assess(make_embedding(), make_aligned(crop, scale=3.0))
    assert upsampled.metrics["sharpness_threshold"] < native.metrics["sharpness_threshold"]


def test_statistics_track_pass_rate(gate: QualityGate) -> None:
    """The gate must report how much it is discarding."""
    for _ in range(5):
        gate.assess(make_embedding(), make_aligned(sharp_crop()))
    blurred = cv2.GaussianBlur(sharp_crop(), (15, 15), 0)
    for _ in range(5):
        gate.assess(make_embedding(), make_aligned(blurred))

    statistics = gate.statistics()
    assert statistics["assessed"] == 10
    assert 0.0 < statistics["pass_rate"] < 1.0


def test_reset_clears_calibration(gate: QualityGate) -> None:
    """Reset must return the gate to its uncalibrated state."""
    for index in range(100):
        gate.assess(make_embedding(seed=index), make_aligned(sharp_crop()))
    gate.reset()
    assert gate.samples_seen == 0
    assert gate.is_calibrated is False


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
@pytest.fixture()
def fusion() -> TemporalFusion:
    """A fusion buffer on stock settings."""
    return TemporalFusion(SurveillanceConfig.default().fusion)


def test_track_needs_min_samples(fusion: TemporalFusion) -> None:
    """A track must not be searchable before min_samples observations."""
    fusion.add("t", make_embedding(), quality=0.9)
    assert fusion.is_ready("t") is False
    assert fusion.fuse("t") is None


def test_unknown_track_fuses_to_none(fusion: TemporalFusion) -> None:
    """Fusing a track that was never seen must return None, not raise."""
    assert fusion.fuse("nope") is None


def test_fusion_reduces_noise(fusion: TemporalFusion) -> None:
    """The central claim: fusing beats any single observation.

    Per-frame noise is roughly zero-mean while identity is not, so averaging
    on the hypersphere moves the estimate toward the true embedding.
    """
    truth = unit_vector(100)
    rng = np.random.default_rng(0)

    single_similarities = []
    for _ in range(20):
        noisy = truth + rng.normal(0, 0.25, 512).astype(np.float32)
        noisy /= np.linalg.norm(noisy)
        single_similarities.append(float(np.dot(noisy, truth)))
        fusion.add("t", Embedding(vector=noisy, norm=30.0), quality=0.8)

    fused = fusion.fuse("t")
    assert fused is not None
    assert float(np.dot(fused.vector, truth)) > max(single_similarities)


def test_quality_weighting_suppresses_a_bad_observation(
    fusion: TemporalFusion,
) -> None:
    """A low-quality outlier must barely move the fused vector."""
    truth = unit_vector(100)
    impostor = unit_vector(999)

    for _ in range(6):
        fusion.add("t", Embedding(vector=truth, norm=30.0), quality=0.95)
    fusion.add("t", Embedding(vector=impostor, norm=30.0), quality=0.02)

    fused = fusion.fuse("t")
    assert fused is not None
    assert float(np.dot(fused.vector, truth)) > 0.99


def test_fused_vector_is_unit_length(fusion: TemporalFusion) -> None:
    """Downstream search treats inner product as cosine similarity."""
    for index in range(6):
        fusion.add("t", make_embedding(seed=index), quality=0.8)
    fused = fusion.fuse("t")
    assert fused is not None
    assert np.linalg.norm(fused.vector) == pytest.approx(1.0, abs=1e-5)
    assert fused.vector.dtype == np.float32


@pytest.mark.parametrize("strategy", ["weighted_mean", "ema", "median"])
def test_every_strategy_produces_a_usable_vector(strategy: str) -> None:
    """All three aggregation rules must emit a unit vector near the truth."""
    config = SurveillanceConfig.from_dict({"fusion": {"strategy": strategy}}).fusion
    fusion = TemporalFusion(config)
    truth = unit_vector(100)
    rng = np.random.default_rng(1)

    for _ in range(12):
        noisy = truth + rng.normal(0, 0.15, 512).astype(np.float32)
        noisy /= np.linalg.norm(noisy)
        fusion.add("t", Embedding(vector=noisy, norm=30.0), quality=0.8)

    fused = fusion.fuse("t")
    assert fused is not None
    assert fused.strategy == strategy
    assert np.linalg.norm(fused.vector) == pytest.approx(1.0, abs=1e-5)
    assert float(np.dot(fused.vector, truth)) > 0.5


def test_median_resists_gross_outliers() -> None:
    """Median fusion must survive a minority of badly wrong observations."""
    config = SurveillanceConfig.from_dict({"fusion": {"strategy": "median"}}).fusion
    fusion = TemporalFusion(config)
    truth = unit_vector(100)

    for _ in range(9):
        fusion.add("t", Embedding(vector=truth, norm=30.0), quality=0.9)
    for seed in range(3):
        fusion.add("t", Embedding(vector=unit_vector(500 + seed), norm=30.0), quality=0.9)

    fused = fusion.fuse("t")
    assert fused is not None
    assert float(np.dot(fused.vector, truth)) > 0.9


def test_coherence_detects_a_mixed_track(fusion: TemporalFusion) -> None:
    """Low coherence means the track holds more than one person.

    The fused vector then represents nobody, which is worth knowing before
    searching a gallery with it.
    """
    first = unit_vector(100)
    second = unit_vector(999)
    for _ in range(5):
        fusion.add("pure", Embedding(vector=first, norm=30.0), quality=0.9)
    for _ in range(5):
        fusion.add("mixed", Embedding(vector=first, norm=30.0), quality=0.9)
        fusion.add("mixed", Embedding(vector=second, norm=30.0), quality=0.9)

    assert fusion.fuse("pure").coherence > 0.99
    assert fusion.fuse("mixed").coherence < 0.7


def test_buffer_respects_max_samples() -> None:
    """Retained observations must stay bounded on a long track."""
    config = SurveillanceConfig.from_dict(
        {"fusion": {"min_samples": 2, "max_samples": 8}}
    ).fusion
    fusion = TemporalFusion(config)
    for index in range(200):
        fusion.add("t", make_embedding(seed=index), quality=0.8)
    assert fusion.sample_count("t") == 8


def test_dropping_a_track_frees_its_buffer(fusion: TemporalFusion) -> None:
    """Memory must scale with live tracks, not with tracks ever seen."""
    for index in range(5):
        fusion.add("t", make_embedding(seed=index), quality=0.8)
    fusion.drop("t")
    assert fusion.track_count == 0
    assert fusion.fuse("t") is None


def test_cancelling_observations_return_none(fusion: TemporalFusion) -> None:
    """Observations that sum to nothing must not be normalised into noise."""
    vector = unit_vector(7)
    for _ in range(3):
        fusion.add("t", Embedding(vector=vector, norm=30.0), quality=0.9)
        fusion.add("t", Embedding(vector=-vector, norm=30.0), quality=0.9)
    assert fusion.fuse("t") is None


def test_invalid_quality_rejected(fusion: TemporalFusion) -> None:
    """A quality score outside [0, 1] indicates an upstream bug."""
    with pytest.raises(ValueError, match="quality"):
        fusion.add("t", make_embedding(), quality=1.5)


def test_fuse_all_returns_only_ready_tracks(fusion: TemporalFusion) -> None:
    """Tracks below min_samples must be omitted rather than half-fused."""
    for index in range(5):
        fusion.add("ready", make_embedding(seed=index), quality=0.8)
    fusion.add("young", make_embedding(seed=99), quality=0.8)

    results = fusion.fuse_all()
    assert set(results) == {"ready"}


def test_weighting_power_sharpens_preference(fusion: TemporalFusion) -> None:
    """quality_power above one must widen the gap between good and bad frames."""
    assert fusion.apply_weighting(0.5) < 0.5
    assert fusion.apply_weighting(1.0) == pytest.approx(1.0)
    assert fusion.apply_weighting(0.0) == pytest.approx(0.0)