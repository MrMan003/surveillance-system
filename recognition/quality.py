"""Quality gating for face embeddings.

Why gate at all
---------------
Every stage upstream of this one succeeds on inputs it should not.  SCRFD will
happily return a confident box around a 16-pixel face turned three-quarters
away.  The aligner will warp it to 112x112 -- upsampling by four, inventing
every added pixel.  The encoder will return a perfectly valid unit vector.
Nothing errors, and the embedding is noise wearing the shape of a face.

Fused into a track, one such vector drags the track's mean embedding away from
the identity it belongs to.  Fused into a gallery entry, it corrupts an
enrolled identity permanently.  The cheapest place to stop it is here.

The signals
-----------
Five, deliberately independent, because each catches failures the others miss:

Detector confidence
    Nearly free, and a weak-but-real prior.

Sharpness
    Variance of the Laplacian over the aligned crop.  Catches motion blur and
    defocus, which produce embeddings that are stable but wrong -- a blurred
    face embeds near other blurred faces rather than near itself.

Illumination
    Mean luminance.  A crushed or blown-out crop has lost the local contrast
    the encoder relies on, and no amount of confidence recovers it.

Truncation
    Fraction of the face box outside the frame.  A half face produces an
    embedding of half a face, which is a confident, repeatable, wrong answer.

Feature norm
    Only when the encoder declares it quality calibrated.  AdaFace trains with
    a norm-scaled margin, so its norm tracks quality closely; ArcFace's does
    not, and reading it as quality would gate on noise.

Adaptive thresholding
---------------------
A fixed norm threshold is wrong for any specific camera.  Norm distributions
shift with lens, lighting, compression and typical subject distance, so a
threshold tuned on one deployment rejects everything or nothing on the next.
After a warmup period the gate switches to a percentile of the norms it has
actually observed, which makes it self-calibrating per stream.

The interaction with upsampling is the subtle part.  Bicubic interpolation is a
low-pass filter, so an upsampled crop looks *smoother* -- its Laplacian variance
falls, and a naive sharpness gate rejects small faces for being blurry when they
are merely small.  Sharpness is therefore compared against a threshold relaxed
in proportion to the upsampling factor, and the smallness is penalised
separately through the scale term.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from alignment.umeyama import AlignedFace
from configs.config import QualityConfig
from recognition.encoder import Embedding
from utils.log import get_logger

__all__ = ["QualityAssessment", "QualityGate"]

LOGGER = get_logger(__name__)

#: Number of recent norms retained for the adaptive threshold. Bounded so a
#: long run cannot grow the buffer without limit, and short enough that the
#: threshold tracks changing conditions rather than averaging over an hour.
_NORM_WINDOW = 512

#: Inter-eye distance, in source pixels, below which a face carries too little
#: real detail to recognise reliably regardless of detector confidence.
_MIN_EYE_DISTANCE = 20.0

#: Eye distance at which the scale term saturates at full marks.
_GOOD_EYE_DISTANCE = 60.0


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    """Verdict and evidence for one candidate embedding.

    Attributes:
        passed: Whether the embedding may be stored and fused.
        score: Composite quality in ``[0, 1]``, used as the fusion weight.
            Meaningful even when ``passed`` is ``False``, for diagnostics.
        reasons: Human-readable failure descriptions; empty when passed.
        metrics: Raw measurements, for tuning and the audit trail.
    """

    passed: bool
    score: float
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a compact debug representation."""
        verdict = "pass" if self.passed else f"fail({', '.join(self.reasons)})"
        return f"QualityAssessment({verdict}, score={self.score:.3f})"


class QualityGate:
    """Decides which embeddings are good enough to keep.

    Stateful only in the norm history backing the adaptive threshold.  That
    history is per-instance, so one gate per video stream keeps calibration
    independent between cameras.

    Args:
        config: Thresholds and adaptive-gating settings.
    """

    def __init__(self, config: QualityConfig) -> None:
        self._config = config
        self._norms: Deque[float] = deque(maxlen=_NORM_WINDOW)
        self._assessed = 0
        self._passed = 0

    # -- properties -------------------------------------------------------- #
    @property
    def samples_seen(self) -> int:
        """Number of norms observed, calibrated or not."""
        return len(self._norms)

    @property
    def is_calibrated(self) -> bool:
        """Whether enough samples have been seen for the adaptive threshold."""
        return self._config.adaptive and len(self._norms) >= self._config.warmup_samples

    @property
    def norm_threshold(self) -> float:
        """Current acceptance floor for the feature norm.

        Returns:
            The configured floor during warmup, and afterwards the higher of
            that floor and the configured percentile of observed norms.  Taking
            the maximum stops a stream of uniformly poor faces from calibrating
            the gate down to accept them all.
        """
        if not self.is_calibrated:
            return self._config.min_norm
        percentile = float(
            np.percentile(np.asarray(self._norms), self._config.adaptive_percentile)
        )
        return max(self._config.min_norm, percentile)

    def statistics(self) -> Dict[str, float]:
        """Summarise gate behaviour so far.

        Returns:
            Counts, pass rate and the current norm threshold.
        """
        return {
            "assessed": self._assessed,
            "passed": self._passed,
            "pass_rate": self._passed / self._assessed if self._assessed else 0.0,
            "samples": len(self._norms),
            "calibrated": float(self.is_calibrated),
            "norm_threshold": self.norm_threshold,
        }

    def reset(self) -> None:
        """Clear the norm history and counters."""
        self._norms.clear()
        self._assessed = 0
        self._passed = 0

    # -- assessment -------------------------------------------------------- #
    def assess(
        self,
        embedding: Embedding,
        aligned: AlignedFace,
        truncation: float = 0.0,
    ) -> QualityAssessment:
        """Evaluate one candidate embedding against every signal.

        Args:
            embedding: The extracted embedding, carrying its norm and the flag
                stating whether that norm is quality calibrated.
            aligned: The crop the embedding came from, carrying warp geometry.
            truncation: Fraction of the source face box outside the frame, from
                :meth:`utils.types.BoundingBox.truncation`.

        Returns:
            The verdict, its composite score and the raw measurements.
        """
        self._assessed += 1
        reasons: List[str] = []
        metrics: Dict[str, float] = {}

        sharpness = self._laplacian_variance(aligned.image)
        brightness = float(aligned.image.mean())
        eye_distance = aligned.source_eye_distance

        # Interpolation is a low-pass filter, so an upsampled crop measures as
        # blurrier than its source. Relaxing the threshold in proportion keeps
        # this from rejecting small faces for the wrong reason.
        upsample_factor = max(aligned.scale, 1.0)
        sharpness_threshold = self._config.blur_threshold / (upsample_factor**2)

        metrics.update(
            {
                "sharpness": sharpness,
                "sharpness_threshold": sharpness_threshold,
                "brightness": brightness,
                "truncation": truncation,
                "eye_distance": eye_distance,
                "warp_scale": aligned.scale,
                "residual": aligned.residual,
                "norm": embedding.norm,
                "detection_score": embedding.detection_score,
            }
        )

        if embedding.detection_score < self._config.min_detection_score:
            reasons.append(
                f"detector confidence {embedding.detection_score:.2f} "
                f"< {self._config.min_detection_score:.2f}"
            )

        if sharpness < sharpness_threshold:
            reasons.append(f"blurred (laplacian var {sharpness:.1f} < {sharpness_threshold:.1f})")

        if not self._config.min_brightness <= brightness <= self._config.max_brightness:
            reasons.append(f"illumination {brightness:.0f} outside acceptable range")

        if truncation > self._config.max_edge_truncation:
            reasons.append(f"partial face ({truncation:.0%} outside frame)")

        if eye_distance < _MIN_EYE_DISTANCE:
            reasons.append(f"face too small (eye distance {eye_distance:.0f}px)")

        # The norm gate applies only when the encoder says its norm means
        # something. Under ArcFace this branch is skipped entirely and the
        # pixel-domain signals carry the decision.
        norm_score = 0.5
        if embedding.norm_is_quality_calibrated:
            threshold = self.norm_threshold
            metrics["norm_threshold"] = threshold
            self._norms.append(embedding.norm)

            if embedding.norm < threshold:
                reasons.append(f"feature norm {embedding.norm:.1f} < {threshold:.1f}")
            elif embedding.norm > self._config.max_norm:
                reasons.append(f"feature norm {embedding.norm:.1f} anomalously high")

            span = max(self._config.max_norm - self._config.min_norm, 1e-6)
            norm_score = float(
                np.clip((embedding.norm - self._config.min_norm) / span, 0.0, 1.0)
            )

        score = self._composite_score(
            norm_score=norm_score,
            sharpness=sharpness,
            sharpness_threshold=sharpness_threshold,
            brightness=brightness,
            truncation=truncation,
            eye_distance=eye_distance,
            detection_score=embedding.detection_score,
        )
        metrics["score"] = score

        passed = not reasons
        if passed:
            self._passed += 1
        else:
            LOGGER.debug("Rejected embedding: %s", "; ".join(reasons))

        return QualityAssessment(passed=passed, score=score, reasons=reasons, metrics=metrics)

    # -- scoring ----------------------------------------------------------- #
    def _composite_score(
        self,
        norm_score: float,
        sharpness: float,
        sharpness_threshold: float,
        brightness: float,
        truncation: float,
        eye_distance: float,
        detection_score: float,
    ) -> float:
        """Blend the signals into a single fusion weight.

        Each term is mapped to ``[0, 1]`` independently so no signal can
        dominate through raw magnitude -- Laplacian variance is unbounded above
        while truncation is a fraction, and averaging them unscaled would let
        one sharp frame outvote every other consideration.

        Args:
            norm_score: Normalised feature norm, or ``0.5`` when uncalibrated.
            sharpness: Laplacian variance of the crop.
            sharpness_threshold: Upsample-adjusted acceptance floor.
            brightness: Mean luminance of the crop.
            truncation: Fraction of the face outside the frame.
            eye_distance: Inter-eye distance in source pixels.
            detection_score: Detector confidence.

        Returns:
            A composite quality score in ``[0, 1]``.
        """
        # Saturating at four times the threshold: past that, extra sharpness
        # stops indicating a better face and starts indicating compression
        # artefacts or sensor noise.
        sharpness_score = float(np.clip(sharpness / (sharpness_threshold * 4.0 + 1e-6), 0.0, 1.0))

        midpoint = (self._config.min_brightness + self._config.max_brightness) / 2.0
        half_span = (self._config.max_brightness - self._config.min_brightness) / 2.0
        brightness_score = float(np.clip(1.0 - abs(brightness - midpoint) / half_span, 0.0, 1.0))

        truncation_score = float(np.clip(1.0 - truncation, 0.0, 1.0))

        scale_score = float(
            np.clip(
                (eye_distance - _MIN_EYE_DISTANCE) / (_GOOD_EYE_DISTANCE - _MIN_EYE_DISTANCE),
                0.0,
                1.0,
            )
        )

        weights = {
            "norm": 0.30,
            "sharpness": 0.20,
            "scale": 0.20,
            "brightness": 0.10,
            "truncation": 0.10,
            "detector": 0.10,
        }
        total = (
            weights["norm"] * norm_score
            + weights["sharpness"] * sharpness_score
            + weights["scale"] * scale_score
            + weights["brightness"] * brightness_score
            + weights["truncation"] * truncation_score
            + weights["detector"] * float(np.clip(detection_score, 0.0, 1.0))
        )
        return float(np.clip(total, 0.0, 1.0))

    # -- measurement ------------------------------------------------------- #
    @staticmethod
    def _laplacian_variance(image: np.ndarray) -> float:
        """Measure sharpness as the variance of the Laplacian.

        A sharp image has strong second derivatives at edges, giving high
        variance; a blurred one has weak derivatives everywhere.

        Args:
            image: Aligned crop, ``uint8``, shaped ``(H, W, 3)``.

        Returns:
            The variance, or ``0.0`` for a degenerate image.
        """
        import cv2

        if image.size == 0:
            return 0.0
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return float(cv2.Laplacian(grey, cv2.CV_64F).var())