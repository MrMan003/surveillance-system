"""Open-set identification: naming a person, or declining to.

Closed set versus open set
--------------------------
A closed-set classifier assumes every query belongs to some known class and
answers "which one".  Applied to surveillance footage that assumption is false
for almost every person who walks past a camera, and the consequence is not a
subtle accuracy loss.  Nearest-neighbour search always returns a neighbour, so
a closed-set system labels every stranger as whoever in the gallery they most
resemble, with a confidence score that looks perfectly reasonable.

An open-set system answers "which one, or none".  ``UNKNOWN`` is the correct
answer for most queries and this module is built to give it freely.

Three independent rejections
----------------------------
Similarity floor
    The best match must clear an absolute threshold.  Necessary, and on its
    own badly insufficient.

Margin
    The best match must beat the runner-up by a gap.  This is the check most
    implementations omit, and the one that matters most.  A best match at 0.36
    with a second at 0.355 clears any sane floor while being a coin flip
    between two people; reporting the winner is a confident guess dressed as an
    identification.  Requiring separation rejects the ambiguity instead.

Coherence
    A fused track whose contributing embeddings disagree contains more than one
    person, so its fused vector describes nobody.  Searching with it produces a
    match to a face that was never in the footage.  Phase 8 measures this, and
    it is cheaper and more reliable to reject here than to detect afterwards.

Calibration
------------
Thresholds are estimated from the gallery's own impostor distribution rather
than guessed.  The similarity at which a chosen fraction of impostor pairs fall
below is, by construction, the threshold achieving that false-accept rate on
this gallery, with this encoder, for this population.  A number transplanted
from a paper's benchmark is a claim about a different distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from configs.config import SearchConfig
from recognition.fusion import FusedEmbedding
from search.gallery import Gallery, SearchHit
from utils.log import get_logger

__all__ = ["Identification", "OpenSetIdentifier"]

LOGGER = get_logger(__name__)

#: Coherence below which a fused track is treated as containing more than one
#: person. Empirically a single-identity track sits near 1.0 and a two-person
#: track near 0.5, so the midpoint separates them with room to spare.
_MIN_COHERENCE = 0.70


@dataclass(frozen=True, slots=True)
class Identification:
    """The outcome of identifying one fused track embedding.

    Attributes:
        track_id: The track that was identified.
        identity: The resolved label, or the configured unknown sentinel.
        similarity: Similarity of the best candidate, whether accepted or not.
        margin: Gap between the best and second-best candidate.  ``inf`` when
            the gallery holds a single identity, since there is nothing to be
            confused with.
        accepted: Whether a named identity was returned.
        threshold: The similarity floor in force for this decision.
        rejection_reason: Why the match was declined; empty when accepted.
        candidates: The full ranked neighbour list, retained so a decision can
            be reviewed after the fact.
    """

    track_id: str
    identity: str
    similarity: float
    margin: float
    accepted: bool
    threshold: float
    rejection_reason: str = ""
    candidates: List[SearchHit] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return a compact debug representation."""
        if self.accepted:
            return (
                f"Identification({self.track_id[:8]} -> {self.identity}, "
                f"sim={self.similarity:.3f}, margin={self.margin:.3f})"
            )
        return (
            f"Identification({self.track_id[:8]} -> {self.identity}, "
            f"{self.rejection_reason})"
        )

    def to_dict(self) -> Dict[str, object]:
        """Serialise for the audit trail.

        Returns:
            A JSON-friendly mapping.
        """
        return {
            "track_id": self.track_id,
            "identity": self.identity,
            "similarity": round(self.similarity, 6),
            "margin": None if np.isinf(self.margin) else round(self.margin, 6),
            "accepted": self.accepted,
            "threshold": round(self.threshold, 6),
            "rejection_reason": self.rejection_reason,
            "candidates": [
                {"identity": hit.identity, "similarity": round(hit.similarity, 6)}
                for hit in self.candidates[:5]
            ],
        }


class OpenSetIdentifier:
    """Resolves fused track embeddings against a gallery, or declines to.

    Args:
        gallery: The enrolled identities to search.
        config: Thresholds and search parameters.
    """

    def __init__(self, gallery: Gallery, config: SearchConfig) -> None:
        self._gallery = gallery
        self._config = config
        self._threshold = config.similarity_threshold
        self._calibrated = False
        self._counts = {"accepted": 0, "rejected": 0}

    # -- properties -------------------------------------------------------- #
    @property
    def threshold(self) -> float:
        """The similarity floor currently in force."""
        return self._threshold

    @property
    def is_calibrated(self) -> bool:
        """Whether the threshold was estimated from impostor statistics."""
        return self._calibrated

    def statistics(self) -> Dict[str, float]:
        """Summarise decisions made so far.

        Returns:
            Counts, acceptance rate and the active threshold.
        """
        total = self._counts["accepted"] + self._counts["rejected"]
        return {
            "accepted": self._counts["accepted"],
            "rejected": self._counts["rejected"],
            "acceptance_rate": self._counts["accepted"] / total if total else 0.0,
            "threshold": self._threshold,
            "calibrated": float(self._calibrated),
        }

    # -- calibration ------------------------------------------------------- #
    def calibrate(self, target_far: Optional[float] = None) -> float:
        """Estimate the similarity threshold from the gallery's impostor pairs.

        The threshold is placed at the ``(1 - target_far)`` quantile of
        similarities between entries belonging to *different* identities: by
        construction, only that fraction of impostor pairs exceed it.

        Args:
            target_far: Desired false-accept rate; defaults to the configured
                value.

        Returns:
            The threshold now in force.  The configured value is retained
            unchanged when the gallery is too small to estimate from -- with
            two identities there are too few impostor pairs for a quantile to
            mean anything.
        """
        far = target_far if target_far is not None else self._config.target_far
        impostors = self._gallery.impostor_similarities()

        if impostors.size < 100:
            LOGGER.warning(
                "Only %d impostor pair(s); keeping the configured threshold "
                "%.3f rather than estimating from too little data",
                impostors.size,
                self._threshold,
            )
            return self._threshold

        quantile = float(np.quantile(impostors, 1.0 - far))
        self._threshold = max(quantile, self._config.similarity_threshold)
        self._calibrated = True

        LOGGER.info(
            "Calibrated threshold to %.4f for FAR=%.1e "
            "(%d impostor pairs, mean %.4f, max %.4f)",
            self._threshold,
            far,
            impostors.size,
            float(impostors.mean()),
            float(impostors.max()),
        )
        return self._threshold

    # -- identification ---------------------------------------------------- #
    def identify(
        self, fused: FusedEmbedding, min_coherence: float = _MIN_COHERENCE
    ) -> Identification:
        """Resolve one fused track embedding.

        Args:
            fused: The track's aggregated embedding.
            min_coherence: Floor on the track's internal agreement.

        Returns:
            The decision, accepted or otherwise.  Never raises on a query that
            matches nothing; that is a normal outcome.
        """
        unknown = self._config.unknown_label

        # A track holding two people fuses to a vector describing neither.
        # Rejecting here is cheaper and more reliable than noticing later that
        # a named identity never appeared in the footage.
        if fused.coherence < min_coherence:
            return self._reject(
                fused.track_id,
                similarity=0.0,
                margin=0.0,
                reason=(
                    f"track incoherent (coherence {fused.coherence:.3f} "
                    f"< {min_coherence:.2f}); it likely contains more than one person"
                ),
                candidates=[],
            )

        hits = self._gallery.search(fused.vector, top_k=self._config.top_k)
        if not hits:
            return self._reject(
                fused.track_id, 0.0, 0.0, "gallery returned no candidates", []
            )

        best = hits[0]

        # The runner-up must be a *different* identity. Two enrolments of the
        # same person are not competing hypotheses, and treating them as such
        # would penalise exactly the multi-enrolment strategy that improves
        # recall.
        runner_up = next(
            (hit for hit in hits[1:] if hit.identity != best.identity), None
        )
        margin = (
            float("inf") if runner_up is None else best.similarity - runner_up.similarity
        )

        if best.similarity < self._threshold:
            return self._reject(
                fused.track_id,
                best.similarity,
                margin,
                f"similarity {best.similarity:.3f} < threshold {self._threshold:.3f}",
                hits,
            )

        if margin < self._config.margin_threshold:
            return self._reject(
                fused.track_id,
                best.similarity,
                margin,
                (
                    f"ambiguous: {best.identity} at {best.similarity:.3f} versus "
                    f"{runner_up.identity} at {runner_up.similarity:.3f} "
                    f"(margin {margin:.3f} < {self._config.margin_threshold:.3f})"
                ),
                hits,
            )

        self._counts["accepted"] += 1
        return Identification(
            track_id=fused.track_id,
            identity=best.identity,
            similarity=best.similarity,
            margin=margin,
            accepted=True,
            threshold=self._threshold,
            candidates=hits,
        )

    def identify_all(
        self, fused_embeddings: Sequence[FusedEmbedding]
    ) -> List[Identification]:
        """Resolve several tracks.

        Args:
            fused_embeddings: The tracks to identify.

        Returns:
            One decision per track, in input order.
        """
        return [self.identify(fused) for fused in fused_embeddings]

    def _reject(
        self,
        track_id: str,
        similarity: float,
        margin: float,
        reason: str,
        candidates: List[SearchHit],
    ) -> Identification:
        """Build a rejection, recording it in the counters.

        Args:
            track_id: The track being decided.
            similarity: Best candidate similarity.
            margin: Gap to the runner-up.
            reason: Why the match was declined.
            candidates: The ranked neighbour list.

        Returns:
            The rejection.
        """
        self._counts["rejected"] += 1
        LOGGER.debug("Track %s -> UNKNOWN: %s", track_id[:8], reason)
        return Identification(
            track_id=track_id,
            identity=self._config.unknown_label,
            similarity=similarity,
            margin=margin,
            accepted=False,
            threshold=self._threshold,
            rejection_reason=reason,
            candidates=candidates,
        )