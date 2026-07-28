"""Tests for the AdaFace backbone and the face encoders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from configs import SurveillanceConfig
from recognition.encoder import (
    ArcFaceOnnxEncoder,
    Embedding,
    EncoderError,
    build_encoder,
)

torch = pytest.importorskip("torch", reason="torch is required for the backbone")

from recognition.backbones import build_backbone  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCFACE = REPO_ROOT / "weights" / "models" / "buffalo_l" / "w600k_r50.onnx"
requires_arcface = pytest.mark.skipif(
    not ARCFACE.is_file(), reason="buffalo_l/w600k_r50.onnx not in weights/"
)


@pytest.fixture()
def config() -> SurveillanceConfig:
    """A configuration pinned to CPU for deterministic tests."""
    return SurveillanceConfig.from_dict({"runtime": {"device": "cpu"}})


def face_crop(seed: int = 0) -> np.ndarray:
    """Produce a deterministic synthetic 112x112 BGR crop."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Backbone
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("architecture", "blocks"), [("ir18", 8), ("ir50", 24)])
def test_backbone_layout(architecture: str, blocks: int) -> None:
    """Stage layouts must match the published IResNet configurations."""
    model = build_backbone(architecture)
    assert len(model.body) == blocks


def test_backbone_parameter_counts() -> None:
    """Parameter counts must match the reference implementations."""
    assert sum(p.numel() for p in build_backbone("ir18").parameters()) / 1e6 == pytest.approx(
        24.0, abs=0.5
    )
    assert sum(p.numel() for p in build_backbone("ir50").parameters()) / 1e6 == pytest.approx(
        43.6, abs=0.5
    )


def test_backbone_output_shapes() -> None:
    """The backbone must emit a 512-d embedding and a scalar norm per face."""
    model = build_backbone("ir18")
    with torch.inference_mode():
        embedding, norm = model(torch.randn(4, 3, 112, 112))
    assert embedding.shape == (4, 512)
    assert norm.shape == (4, 1)


def test_backbone_emits_unit_vectors() -> None:
    """Embeddings must be L2 normalised so inner product is cosine similarity."""
    model = build_backbone("ir18")
    with torch.inference_mode():
        embedding, _ = model(torch.randn(8, 3, 112, 112))
    assert embedding.norm(dim=1).numpy() == pytest.approx(np.ones(8), abs=1e-5)


def test_feature_map_is_seven_by_seven() -> None:
    """Four stride-2 stages must reduce 112x112 to 7x7 for the projection."""
    model = build_backbone("ir18")
    with torch.inference_mode():
        features = model.input_layer(torch.randn(1, 3, 112, 112))
        features = model.body(features)
    assert features.shape == (1, 512, 7, 7)


def test_zero_input_does_not_produce_nan() -> None:
    """A zero feature vector must not divide by zero into the gallery."""
    model = build_backbone("ir18")
    with torch.inference_mode():
        embedding, _ = model(torch.zeros(2, 3, 112, 112))
    assert torch.isfinite(embedding).all()


def test_backbone_is_deterministic() -> None:
    """Two passes over the same input must agree exactly in eval mode."""
    model = build_backbone("ir18")
    data = torch.randn(2, 3, 112, 112)
    with torch.inference_mode():
        first, _ = model(data)
        second, _ = model(data)
    assert torch.equal(first, second)


def test_state_dict_uses_reference_key_names() -> None:
    """Key names must match published checkpoints so they load unmapped."""
    keys = set(build_backbone("ir18").state_dict())
    assert "input_layer.0.weight" in keys
    assert any(key.startswith("body.0.res_layer") for key in keys)
    # body.0 has an identity shortcut (MaxPool2d, no parameters); parameterised
    # shortcuts appear only where a stage changes channel count.
    assert any("shortcut_layer" in key for key in keys)


def test_unsupported_architecture_rejected() -> None:
    """Only the implemented backbones may be requested."""
    with pytest.raises(ValueError, match="Unsupported architecture"):
        build_backbone("ir101")


def test_non_square_input_rejected() -> None:
    """The projection is sized for 112x112; other sizes must raise."""
    from recognition.backbones import AdaFaceBackbone, BlockSpec

    with pytest.raises(ValueError, match="112x112"):
        AdaFaceBackbone([BlockSpec(64, 64, 1)], input_size=(224, 224))


# --------------------------------------------------------------------------- #
# Embedding value object
# --------------------------------------------------------------------------- #
def test_embedding_rejects_wrong_dimensionality() -> None:
    """A non-512 vector indicates a model mismatch and must raise."""
    with pytest.raises(ValueError, match="shape"):
        Embedding(vector=np.zeros(256, dtype=np.float32), norm=1.0)


def test_embedding_rejects_unnormalised_vector() -> None:
    """Downstream code treats inner product as cosine; unit length is required."""
    with pytest.raises(ValueError, match="unit length"):
        Embedding(vector=np.ones(512, dtype=np.float32), norm=1.0)


def test_similarity_is_inner_product() -> None:
    """Cosine similarity of unit vectors reduces to a dot product."""
    first = np.zeros(512, dtype=np.float32)
    first[0] = 1.0
    second = np.zeros(512, dtype=np.float32)
    second[1] = 1.0

    assert Embedding(first, 1.0).similarity(Embedding(first, 1.0)) == pytest.approx(1.0)
    assert Embedding(first, 1.0).similarity(Embedding(second, 1.0)) == pytest.approx(0.0)
    assert Embedding(first, 1.0).similarity(Embedding(-first, 1.0)) == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# Encoder contract
# --------------------------------------------------------------------------- #
@requires_arcface
def test_encode_requires_load(config) -> None:
    """Inference before load() must raise."""
    encoder = ArcFaceOnnxEncoder(config.recognition, config.runtime, config.paths)
    with pytest.raises(EncoderError, match="not loaded"):
        encoder.encode([face_crop()])


@requires_arcface
def test_empty_input(config) -> None:
    """An empty batch must be a no-op."""
    with ArcFaceOnnxEncoder(config.recognition, config.runtime, config.paths) as encoder:
        assert encoder.encode([]) == []


@requires_arcface
def test_malformed_crop_rejected(config) -> None:
    """Wrong dtype or shape must fail loudly rather than embed noise."""
    with ArcFaceOnnxEncoder(config.recognition, config.runtime, config.paths) as encoder:
        with pytest.raises(EncoderError, match="uint8"):
            encoder.encode([np.zeros((112, 112, 3), dtype=np.float32)])
        with pytest.raises(EncoderError, match=r"H, W, 3"):
            encoder.encode([np.zeros((112, 112), dtype=np.uint8)])


@requires_arcface
def test_mismatched_scores_rejected(config) -> None:
    """A scores list of the wrong length must raise, not misalign."""
    with ArcFaceOnnxEncoder(config.recognition, config.runtime, config.paths) as encoder:
        with pytest.raises(ValueError, match="detection_scores"):
            encoder.encode([face_crop(0), face_crop(1)], [0.9])


@requires_arcface
def test_output_order_matches_input(config) -> None:
    """Results must align one-to-one with input crops across batch boundaries."""
    config.recognition.batch_size = 2
    crops = [face_crop(index) for index in range(5)]
    with ArcFaceOnnxEncoder(config.recognition, config.runtime, config.paths) as encoder:
        embeddings = encoder.encode(crops, [0.1 * (i + 1) for i in range(5)])

    assert len(embeddings) == 5
    assert [round(e.detection_score, 2) for e in embeddings] == [0.1, 0.2, 0.3, 0.4, 0.5]


@requires_arcface
def test_embeddings_are_unit_length(config) -> None:
    """Every emitted vector must be normalised."""
    with ArcFaceOnnxEncoder(config.recognition, config.runtime, config.paths) as encoder:
        for embedding in encoder.encode([face_crop(i) for i in range(3)]):
            assert np.linalg.norm(embedding.vector) == pytest.approx(1.0, abs=1e-5)
            assert embedding.vector.dtype == np.float32


@requires_arcface
def test_identical_crops_give_identical_embeddings(config) -> None:
    """The encoder must be deterministic, including across a batch."""
    crop = face_crop(7)
    with ArcFaceOnnxEncoder(config.recognition, config.runtime, config.paths) as encoder:
        first, second = encoder.encode([crop, crop])
    assert first.similarity(second) == pytest.approx(1.0, abs=1e-5)


@requires_arcface
def test_batching_does_not_change_results(config) -> None:
    """Batched and single-crop inference must agree.

    A model with a fixed output batch dimension in its metadata can compute
    correctly while reporting the wrong shape; this checks the values.
    """
    crops = [face_crop(index) for index in range(4)]

    config.recognition.batch_size = 1
    with ArcFaceOnnxEncoder(config.recognition, config.runtime, config.paths) as encoder:
        singly = encoder.encode(crops)

    config.recognition.batch_size = 4
    with ArcFaceOnnxEncoder(config.recognition, config.runtime, config.paths) as encoder:
        batched = encoder.encode(crops)

    for one, many in zip(singly, batched):
        assert one.similarity(many) == pytest.approx(1.0, abs=1e-4)


@requires_arcface
def test_arcface_norm_is_flagged_uncalibrated(config) -> None:
    """ArcFace norms must not be presented as a quality signal.

    Only AdaFace trains with a norm-scaled margin; treating an ArcFace norm as
    quality would gate on noise.
    """
    with ArcFaceOnnxEncoder(config.recognition, config.runtime, config.paths) as encoder:
        embedding = encoder.encode([face_crop()])[0]
    assert embedding.norm_is_quality_calibrated is False
    assert embedding.norm > 0


@requires_arcface
def test_close_is_idempotent(config) -> None:
    """Closing twice must not raise."""
    encoder = ArcFaceOnnxEncoder(config.recognition, config.runtime, config.paths).load()
    encoder.close()
    encoder.close()
    assert encoder.is_loaded is False


# --------------------------------------------------------------------------- #
# Encoder selection
# --------------------------------------------------------------------------- #
def test_build_encoder_falls_back_without_adaface(config, tmp_path) -> None:
    """A missing AdaFace checkpoint must fall back rather than crash."""
    config.paths.weights_dir = tmp_path
    encoder = build_encoder(config.recognition, config.runtime, config.paths)
    assert isinstance(encoder, ArcFaceOnnxEncoder)


def test_build_encoder_can_force_arcface(config) -> None:
    """The interim encoder must be selectable explicitly."""
    encoder = build_encoder(
        config.recognition, config.runtime, config.paths, prefer_adaface=False
    )
    assert isinstance(encoder, ArcFaceOnnxEncoder)


def test_missing_checkpoint_reports_the_path(config, tmp_path) -> None:
    """A missing AdaFace checkpoint must name where it was expected."""
    from recognition.encoder import AdaFaceEncoder

    config.paths.weights_dir = tmp_path
    with pytest.raises(EncoderError, match="not found"):
        AdaFaceEncoder(config.recognition, config.runtime, config.paths).load()


def test_incompatible_checkpoint_is_rejected(config, tmp_path) -> None:
    """A checkpoint for another architecture must fail loudly, not partially load.

    Loading with strict=False and ignoring missing keys would leave most of the
    network at random initialisation while appearing to succeed.
    """
    from recognition.encoder import AdaFaceEncoder

    config.paths.weights_dir = tmp_path
    torch.save({"state_dict": {"input_layer.0.weight": torch.randn(64, 3, 3, 3)}},
               tmp_path / config.recognition.weights)

    with pytest.raises(EncoderError, match="missing"):
        AdaFaceEncoder(config.recognition, config.runtime, config.paths).load()