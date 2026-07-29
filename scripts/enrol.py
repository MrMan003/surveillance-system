#!/usr/bin/env python3
"""Build a searchable gallery from folders of reference face images.

Expected input layout, one folder per person::

    gallery_source/
        alice/
            front.jpg
            side.jpg
        bob/
            id_photo.png

Why enrolment is stricter than inference
----------------------------------------
A bad frame at inference time costs one observation out of dozens, and temporal
fusion absorbs it.  A bad *enrolment* is permanent: it becomes the reference
every future query is compared against, so a blurred or mis-detected reference
quietly degrades recall for that person forever, and can attract false matches
from other people.

Two rules follow.  Images containing zero or several faces are skipped rather
than guessed at -- enrolling the wrong face under a name is worse than not
enrolling the person at all.  And the quality gate runs with the same
thresholds as inference, so a reference that would be discarded during a run is
not accepted into the gallery either.

Multiple enrolments per person are strongly preferred.  A single reference
covers one point on that person's appearance manifold; three covering different
lighting and pose raise recall substantially at negligible search cost.

Usage:
    python scripts/enrol.py --input datasets/gallery_source --output datasets/gallery
    python scripts/enrol.py --input src --output gal --write-consent-template
    python scripts/enrol.py --input src --output gal --min-quality 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alignment.umeyama import FaceAligner  # noqa: E402
from configs.config import SurveillanceConfig  # noqa: E402
from detection.face_detector import FaceDetector  # noqa: E402
from recognition.encoder import build_encoder  # noqa: E402
from recognition.quality import QualityGate  # noqa: E402
from search.gallery import CONSENT_MANIFEST, Gallery, GalleryEntry  # noqa: E402
from utils.log import get_logger, setup_logging  # noqa: E402

LOGGER = get_logger("enrol")

#: Image extensions accepted as enrolment references.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class IdentityReport:
    """Per-identity enrolment outcome.

    Attributes:
        identity: The person's label.
        accepted: Files that produced a gallery entry.
        rejected: ``(filename, reason)`` for each file that did not.
        quality_scores: Composite quality of each accepted entry.
    """

    identity: str
    accepted: List[str] = field(default_factory=list)
    rejected: List[Tuple[str, str]] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)

    @property
    def mean_quality(self) -> float:
        """Mean quality of accepted entries, or ``0.0`` when none were."""
        return float(np.mean(self.quality_scores)) if self.quality_scores else 0.0


def discover_identities(root: Path) -> Dict[str, List[Path]]:
    """Find identity folders and their images.

    Args:
        root: Directory holding one subdirectory per person.

    Returns:
        A mapping from identity label to sorted image paths.

    Raises:
        SystemExit: If the directory is missing or holds no identity folders.
    """
    if not root.is_dir():
        LOGGER.error("Input directory not found: %s", root)
        raise SystemExit(2)

    identities: Dict[str, List[Path]] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        images = sorted(
            path
            for path in entry.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith(".")
        )
        if images:
            identities[entry.name] = images
        else:
            LOGGER.warning("No images in %s; skipping", entry)

    if not identities:
        LOGGER.error(
            "No identity folders in %s. Expected one subdirectory per person, "
            "each containing reference images.",
            root,
        )
        raise SystemExit(2)
    return identities


def enrol_image(
    path: Path,
    detector: FaceDetector,
    aligner: FaceAligner,
    encoder: object,
    gate: QualityGate,
    min_quality: float,
) -> Tuple[Optional[np.ndarray], float, str]:
    """Produce one reference embedding from a single image.

    Args:
        path: Image file.
        detector: Loaded face detector.
        aligner: Face aligner.
        encoder: Loaded face encoder.
        gate: Quality gate.
        min_quality: Composite quality floor for acceptance.

    Returns:
        A tuple ``(vector, quality, reason)``.  ``vector`` is ``None`` when the
        image was rejected, and ``reason`` then explains why.
    """
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return None, 0.0, "unreadable image"

    faces = detector.detect(image)
    if not faces:
        return None, 0.0, "no face detected"
    if len(faces) > 1:
        # Guessing which face belongs to the name would enrol the wrong person
        # under it, which is worse than not enrolling them at all.
        return None, 0.0, f"{len(faces)} faces detected; enrolment must be unambiguous"

    face = faces[0]
    aligned = aligner.align(image, face)
    if aligned is None:
        return None, 0.0, f"alignment rejected (roll {face.roll_degrees:+.0f} degrees)"

    embedding = encoder.encode([aligned.image], [face.score])[0]
    height, width = image.shape[:2]
    verdict = gate.assess(embedding, aligned, face.box.truncation(width, height))

    if not verdict.passed:
        return None, verdict.score, "; ".join(verdict.reasons)
    if verdict.score < min_quality:
        return None, verdict.score, f"quality {verdict.score:.3f} below {min_quality:.2f}"

    return embedding.vector, verdict.score, ""


def write_consent_template(directory: Path, identities: List[str]) -> Path:
    """Write a consent manifest skeleton for manual completion.

    The template is deliberately incomplete: every ``basis`` field is left
    empty so it cannot be used as-is.  Filling it in is a decision a person
    makes, not something a script should default.

    Args:
        directory: Destination directory.
        identities: Identity labels to include.

    Returns:
        The written path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CONSENT_MANIFEST
    payload = {
        "_note": (
            "Face embeddings are biometric identifiers. Record the lawful basis "
            "for each person before using this gallery. Leaving 'basis' empty is "
            "not consent."
        ),
        "identities": [
            {"identity": name, "basis": "", "obtained_on": "", "expires_on": ""}
            for name in sorted(identities)
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote consent template to %s -- complete it before use", path)
    return path


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="source folders")
    parser.add_argument("--output", "-o", type=Path, required=True, help="gallery destination")
    parser.add_argument("--config", "-c", type=Path, help="YAML or JSON configuration")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.0,
        help="composite quality floor beyond the standard gate",
    )
    parser.add_argument(
        "--write-consent-template",
        action="store_true",
        help="emit a consent manifest skeleton alongside the gallery",
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO"
    )
    return parser


def _print_report(reports: Dict[str, IdentityReport], gallery: Gallery) -> None:
    """Print a per-identity enrolment summary.

    Args:
        reports: Per-identity outcomes.
        gallery: The built gallery.
    """
    print()
    print("=" * 70)
    print(f"  {'identity':<24}{'enrolled':>10}{'rejected':>10}{'mean quality':>16}")
    print("  " + "-" * 66)
    for identity in sorted(reports):
        report = reports[identity]
        quality = f"{report.mean_quality:.3f}" if report.accepted else "-"
        print(
            f"  {identity:<24}{len(report.accepted):>10}"
            f"{len(report.rejected):>10}{quality:>16}"
        )
    print("  " + "-" * 66)
    print(f"  {gallery.size} entries covering {len(gallery.identities)} identities")

    impostors = gallery.impostor_similarities()
    if impostors.size:
        print(
            f"  impostor similarity: mean {impostors.mean():+.3f}, "
            f"p99 {np.percentile(impostors, 99):.3f}, max {impostors.max():.3f}"
        )
        if float(impostors.max()) > 0.5:
            print(
                "  WARNING: some different-identity pairs are highly similar. "
                "Check for one person enrolled under two names."
            )
    print("=" * 70)


def main(argv: Optional[list] = None) -> int:
    """Run enrolment.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level)

    config = (
        SurveillanceConfig.from_file(args.config)
        if args.config
        else SurveillanceConfig.default()
    )
    config.runtime.device = args.device
    config = SurveillanceConfig.from_dict(config.to_dict()).validate()
    config.runtime.apply()

    identities = discover_identities(args.input)
    LOGGER.info(
        "Found %d identity(ies), %d image(s)",
        len(identities),
        sum(len(paths) for paths in identities.values()),
    )

    aligner = FaceAligner(config.alignment)
    gate = QualityGate(config.quality)
    encoder = build_encoder(config.recognition, config.runtime, config.paths)
    gallery = Gallery(config.search, config.governance)
    reports: Dict[str, IdentityReport] = {}

    with FaceDetector(config.detection, config.runtime, config.paths) as detector, encoder:
        for identity, images in identities.items():
            report = IdentityReport(identity=identity)
            for path in images:
                vector, quality, reason = enrol_image(
                    path, detector, aligner, encoder, gate, args.min_quality
                )
                if vector is None:
                    report.rejected.append((path.name, reason))
                    LOGGER.warning("  %s/%s rejected: %s", identity, path.name, reason)
                    continue

                gallery.add(
                    GalleryEntry(
                        identity=identity,
                        vector=vector,
                        source=str(path.relative_to(args.input)),
                        metadata={"quality": f"{quality:.4f}"},
                    )
                )
                report.accepted.append(path.name)
                report.quality_scores.append(quality)
            reports[identity] = report

    enrolled = [name for name, report in reports.items() if report.accepted]
    if not enrolled:
        LOGGER.error("No images passed enrolment; the gallery would be empty")
        return 1

    gallery.build()
    gallery.save(args.output)

    if args.write_consent_template:
        write_consent_template(args.output, enrolled)

    _print_report(reports, gallery)

    missing = [name for name, report in reports.items() if not report.accepted]
    if missing:
        LOGGER.warning(
            "%d identity(ies) enrolled nothing and cannot be recognised: %s",
            len(missing),
            missing,
        )
    single = [name for name, report in reports.items() if len(report.accepted) == 1]
    if single:
        LOGGER.warning(
            "%d identity(ies) have a single reference image. Recall improves "
            "substantially with three or more covering different lighting and "
            "pose: %s",
            len(single),
            single[:5],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())