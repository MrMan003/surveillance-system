"""Tests for gallery enrolment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

cv2 = pytest.importorskip("cv2")
pytest.importorskip("faiss")

from scripts.enrol import (  # noqa: E402
    IdentityReport,
    build_parser,
    discover_identities,
    write_consent_template,
)

SCRFD = REPO_ROOT / "weights" / "models" / "buffalo_l" / "det_10g.onnx"
requires_scrfd = pytest.mark.skipif(not SCRFD.is_file(), reason="SCRFD weights absent")


@pytest.fixture()
def source_tree(tmp_path: Path) -> Path:
    """A source tree with two identities, one empty folder and a stray file."""
    root = tmp_path / "src"
    for name in ("alice", "bob"):
        (root / name).mkdir(parents=True)
        for index in range(2):
            image = np.random.default_rng(index).integers(
                0, 255, (200, 200, 3), dtype=np.uint8
            )
            cv2.imwrite(str(root / name / f"{index}.jpg"), image)
    (root / "no_images").mkdir()
    (root / "stray.txt").write_text("not an identity")
    return root


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def test_discovers_identity_folders(source_tree: Path) -> None:
    """One folder per person, images sorted, non-directories ignored."""
    identities = discover_identities(source_tree)
    assert set(identities) == {"alice", "bob"}
    assert len(identities["alice"]) == 2


def test_empty_folders_are_skipped(source_tree: Path) -> None:
    """A folder with no images cannot enrol anyone."""
    assert "no_images" not in discover_identities(source_tree)


def test_missing_directory_exits(tmp_path: Path) -> None:
    """A nonexistent input must exit rather than raise."""
    with pytest.raises(SystemExit):
        discover_identities(tmp_path / "nope")


def test_directory_without_identities_exits(tmp_path: Path) -> None:
    """A directory holding no identity folders must exit."""
    (tmp_path / "loose.jpg").write_bytes(b"")
    with pytest.raises(SystemExit):
        discover_identities(tmp_path)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_report_mean_quality() -> None:
    """Mean quality must be defined even when nothing was accepted."""
    empty = IdentityReport(identity="x")
    assert empty.mean_quality == 0.0

    populated = IdentityReport(identity="y", accepted=["a", "b"], quality_scores=[0.6, 0.8])
    assert populated.mean_quality == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# Consent
# --------------------------------------------------------------------------- #
def test_consent_template_lists_every_identity(tmp_path: Path) -> None:
    """The template must cover everyone enrolled."""
    path = write_consent_template(tmp_path, ["bob", "alice"])
    payload = json.loads(path.read_text())
    assert [r["identity"] for r in payload["identities"]] == ["alice", "bob"]


def test_consent_template_is_deliberately_incomplete(tmp_path: Path) -> None:
    """An empty basis field must not read as consent.

    Filling it in is a decision a person makes; a script defaulting it would
    manufacture a lawful basis that nobody actually established.
    """
    path = write_consent_template(tmp_path, ["alice"])
    payload = json.loads(path.read_text())
    assert payload["identities"][0]["basis"] == ""

    from configs import SurveillanceConfig
    from search.gallery import Gallery, GalleryEntry

    config = SurveillanceConfig.default()
    gallery = Gallery(config.search, config.governance)
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 1.0
    gallery.add(GalleryEntry(identity="alice", vector=vector))
    # The manifest satisfies the structural check; a human must still fill it in.
    gallery.verify_consent(tmp_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_parser_accepts_documented_flags() -> None:
    """Every documented flag must parse."""
    args = build_parser().parse_args(
        ["-i", "src", "-o", "gal", "--device", "cpu",
         "--min-quality", "0.5", "--write-consent-template"]
    )
    assert args.min_quality == 0.5
    assert args.write_consent_template is True


# --------------------------------------------------------------------------- #
# Enrolment rules
# --------------------------------------------------------------------------- #
@requires_scrfd
def test_faceless_image_is_rejected(tmp_path: Path) -> None:
    """An image with no face cannot become a reference."""
    from alignment import FaceAligner
    from configs import SurveillanceConfig
    from detection.face_detector import FaceDetector
    from recognition import QualityGate, build_encoder
    from scripts.enrol import enrol_image

    config = SurveillanceConfig.from_dict({"runtime": {"device": "cpu"}})
    path = tmp_path / "noise.jpg"
    cv2.imwrite(
        str(path),
        np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8),
    )

    encoder = build_encoder(config.recognition, config.runtime, config.paths)
    with FaceDetector(config.detection, config.runtime, config.paths) as detector, encoder:
        vector, _, reason = enrol_image(
            path, detector, FaceAligner(config.alignment),
            encoder, QualityGate(config.quality), 0.0,
        )
    assert vector is None
    assert "no face" in reason


@requires_scrfd
def test_unreadable_file_is_rejected(tmp_path: Path) -> None:
    """A corrupt file must be reported, not crash enrolment."""
    from alignment import FaceAligner
    from configs import SurveillanceConfig
    from detection.face_detector import FaceDetector
    from recognition import QualityGate, build_encoder
    from scripts.enrol import enrol_image

    config = SurveillanceConfig.from_dict({"runtime": {"device": "cpu"}})
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not an image")

    encoder = build_encoder(config.recognition, config.runtime, config.paths)
    with FaceDetector(config.detection, config.runtime, config.paths) as detector, encoder:
        vector, _, reason = enrol_image(
            path, detector, FaceAligner(config.alignment),
            encoder, QualityGate(config.quality), 0.0,
        )
    assert vector is None
    assert "unreadable" in reason