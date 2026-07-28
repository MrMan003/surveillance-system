"""Temporal fusion of per-frame embeddings into one vector per track.

The problem
-----------
A track observed over eighty frames yields up to eighty embeddings of the same
person.  Searching the gallery with each one separately is wasteful and, worse,
inconsistent: some frames match the right identity, others match nobody, and
there is no principled way to reconcile the disagreement after the fact.

Fusing first is strictly better.  Averaging unit vectors on the hypersphere
suppresses per-frame noise -- pose jitter, compression artefacts, lighting
flicker -- because that noise is roughly zero-mean while identity is not.  The
fused vector is closer to the person's true embedding than almost any single
frame's, and one search per track replaces eighty.

Why weight by quality
---------------------
An unweighted mean lets a badly blurred frame contribute as much as a sharp
frontal one.  Since blurred embeddings do not point randomly but cluster with
*other blurred faces*, several of them can pull a fused vector toward a generic
"blurry face" direction that matches the wrong identity confidently.  Weighting
by the Phase 7 quality score, raised to a configurable power, makes good frames
dominate.  ``quality_power`` above one sharpens that preference.

The three strategies
--------------------
Weighted mean
    Default.  Every retained observation contributes in proportion to its
    quality.  Best when observations are independent and roughly comparable.

Exponential moving average
    Recency-biased and updated in place.  Appropriate for long-running tracks
    where appearance drifts -- a person walking from shade into sunlight -- and
    where the oldest observations describe a materially different image.

Median
    Component-wise, then renormalised.  Robust to a small number of gross
    outliers, which is what a face briefly associated with the wrong track
    produces.  Note this is the component-wise median rather than the geometric
    median: it is not rotation invariant, but it is O(n log n) and near enough
    for unit vectors clustered in a narrow cone.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from configs.config import FusionConfig, FusionStrategy
from recognition.encoder import Embedding
from utils.log import get_logger

__all__ = ["FusedEmbedding", "TemporalFusion"]

LOGGER = get_logger(__name__)

#: Floor for a fused vector's magnitude before normalisation. Below this the
#: observations have cancelled out, which means they were not the same person.
_MIN_MAGNITUDE = 1e-8


@dataclass(frozen=True, slots=True)
class FusedEmbedding:
    """One track's aggregated identity vector.

    Attributes:
        vector: L2-normalised ``float32`` array of shape ``(512,)``.
        track_id: The track this describes.
        sample_count: Observations behind the vector.
        mean_quality: Mean quality score of those observations.
        coherence: Mean pairwise cosine similarity among the contributing
            embeddings, in ``[-1, 1]``.  High coherence means they agree, and
            the fused vector represents them well.  Low coherence means the
            track contains more than one person, so the fused vector represents
            nobody -- which is worth knowing before searching a gallery with it.
        strategy: The aggregation rule used.
    """

    vector: np.ndarray
    track_id: str
    sample_count: int
    mean_quality: float
    coherence: float
    strategy: str

    def similarity(self, other: "FusedEmbedding") -> float:
        """Cosine similarity against another fused embedding.

        Args:
            other: The embedding to compare against.

        Returns:
            A value in ``[-1, 1]``.
        """
        return float(np.dot(self.vector, other.vector))

    def __repr__(self) -> str:
        """Return a compact debug representation."""
        return (
            f"FusedEmbedding({self.track_id[:8]}, n={self.sample_count}, "
            f"quality={self.mean_quality:.3f}, coherence={self.coherence:.3f}, "
            f"{self.strategy})"
        )


@dataclass
class _TrackBuffer:
    """Bounded observation buffer for one track.

    Attributes:
        vectors: Retained embeddings, oldest first.
        weights: Quality score per retained embedding.
        ema: Running exponential moving average, when that strategy is active.
        total_seen: Observations added over the track's lifetime, including
            those evicted by the size cap.
    """

    vectors: Deque[np.ndarray]
    weights: Deque[float]
    ema: Optional[np.ndarray] = None
    total_seen: int = 0


class TemporalFusion:
    """Accumulates per-frame embeddings and fuses them per track.

    Args:
        config: Fusion strategy and buffer sizing.
    """

    def __init__(self, config: FusionConfig) -> None:
        self._config = config
        self._buffers: Dict[str, _TrackBuffer] = {}

    # -- properties -------------------------------------------------------- #
    @property
    def track_count(self) -> int:
        """Number of tracks holding at least one observation."""
        return len(self._buffers)

    def sample_count(self, track_id: str) -> int:
        """Observations currently retained for a track.

        Args:
            track_id: The track to query.

        Returns:
            The retained count, or ``0`` for an unknown track.
        """
        buffer = self._buffers.get(track_id)
        return len(buffer.vectors) if buffer else 0

    def is_ready(self, track_id: str) -> bool:
        """Whether a track has enough observations to be searchable.

        Args:
            track_id: The track to query.

        Returns:
            ``True`` once ``min_samples`` observations have been retained.
        """
        return self.sample_count(track_id) >= self._config.min_samples

    def ready_tracks(self) -> List[str]:
        """List every track with enough observations to fuse.

        Returns:
            Track identifiers, in insertion order.
        """
        return [track_id for track_id in self._buffers if self.is_ready(track_id)]

    # -- accumulation ------------------------------------------------------ #
    def add(self, track_id: str, embedding: Embedding, quality: float) -> None:
        """Record one observation against a track.

        Args:
            track_id: The track the embedding belongs to.
            embedding: The extracted embedding.
            quality: Composite quality score in ``[0, 1]``, from Phase 7.

        Raises:
            ValueError: If the quality score is outside ``[0, 1]``.
        """
        if not 0.0 <= quality <= 1.0:
            raise ValueError(f"quality must be in [0, 1], got {quality}")

        buffer = self._buffers.get(track_id)
        if buffer is None:
            buffer = _TrackBuffer(
                vectors=deque(maxlen=self._config.max_samples),
                weights=deque(maxlen=self._config.max_samples),
            )
            self._buffers[track_id] = buffer

        vector = np.asarray(embedding.vector, dtype=np.float32)
        buffer.vectors.append(vector)
        buffer.weights.append(float(quality))
        buffer.total_seen += 1

        if self._config.strategy is FusionStrategy.EMA:
            self._update_ema(buffer, vector, quality)

    def _update_ema(self, buffer: _TrackBuffer, vector: np.ndarray, quality: float) -> None:
        """Fold one observation into the running exponential moving average.

        The smoothing factor is scaled by quality, so a poor observation moves
        the average less than a good one. A fixed alpha would let a single
        blurred frame shift the estimate as much as a sharp one.

        Args:
            buffer: The track's buffer.
            vector: The new embedding.
            quality: Its quality score.
        """
        if buffer.ema is None:
            buffer.ema = vector.copy()
            return
        alpha = self._config.ema_alpha * quality
        buffer.ema = (1.0 - alpha) * buffer.ema + alpha * vector

    # -- fusion ------------------------------------------------------------ #
    def fuse(self, track_id: str) -> Optional[FusedEmbedding]:
        """Aggregate a track's observations into one vector.

        Args:
            track_id: The track to fuse.

        Returns:
            The fused embedding, or ``None`` when the track is unknown or has
            fewer than ``min_samples`` observations.

        Raises:
            ValueError: If the configured strategy is unrecognised.
        """
        buffer = self._buffers.get(track_id)
        if buffer is None or len(buffer.vectors) < self._config.min_samples:
            return None

        vectors = np.stack(buffer.vectors).astype(np.float32)
        weights = np.asarray(buffer.weights, dtype=np.float32)
        strategy = self._config.strategy

        if strategy is FusionStrategy.WEIGHTED_MEAN:
            fused = self._weighted_mean(vectors, weights)
        elif strategy is FusionStrategy.EMA:
            fused = buffer.ema if buffer.ema is not None else self._weighted_mean(vectors, weights)
        elif strategy is FusionStrategy.MEDIAN:
            fused = np.median(vectors, axis=0)
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"Unknown fusion strategy {strategy!r}")

        fused = np.asarray(fused, dtype=np.float32)
        if self._config.renormalize:
            magnitude = float(np.linalg.norm(fused))
            if magnitude < _MIN_MAGNITUDE:
                LOGGER.warning(
                    "Track %s fused to a near-zero vector; its observations "
                    "cancelled out, which usually means the track contains "
                    "more than one person",
                    track_id[:8],
                )
                return None
            fused = fused / magnitude

        return FusedEmbedding(
            vector=np.ascontiguousarray(fused, dtype=np.float32),
            track_id=track_id,
            sample_count=len(buffer.vectors),
            mean_quality=float(weights.mean()),
            coherence=self._coherence(vectors),
            strategy=strategy.value,
        )

    def fuse_all(self) -> Dict[str, FusedEmbedding]:
        """Fuse every track that has enough observations.

        Returns:
            A mapping from track id to fused embedding.
        """
        results: Dict[str, FusedEmbedding] = {}
        for track_id in self._buffers:
            fused = self.fuse(track_id)
            if fused is not None:
                results[track_id] = fused
        return results

    @staticmethod
    def _weighted_mean(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute the quality-weighted mean of a set of vectors.

        Args:
            vectors: Array of shape ``(N, 512)``.
            weights: Array of shape ``(N,)``.

        Returns:
            The weighted mean, shape ``(512,)``.  Falls back to an unweighted
            mean when every weight is zero, which is preferable to returning a
            zero vector that would then fail normalisation.
        """
        total = float(weights.sum())
        if total < _MIN_MAGNITUDE:
            return vectors.mean(axis=0)
        return (vectors * weights[:, None]).sum(axis=0) / total

    @staticmethod
    def _coherence(vectors: np.ndarray) -> float:
        """Measure agreement among a track's embeddings.

        Computed as the mean off-diagonal entry of the Gram matrix.  The
        vectors are unit length, so that product *is* the pairwise cosine
        similarity matrix and no explicit pair loop is needed.

        Args:
            vectors: Array of shape ``(N, 512)`` with unit rows.

        Returns:
            Mean pairwise cosine similarity, or ``1.0`` for a single vector.
        """
        count = vectors.shape[0]
        if count < 2:
            return 1.0
        gram = vectors @ vectors.T
        off_diagonal = gram.sum() - np.trace(gram)
        return float(off_diagonal / (count * (count - 1)))

    # -- lifecycle --------------------------------------------------------- #
    def apply_weighting(self, quality: float) -> float:
        """Shape a quality score into a fusion weight.

        Args:
            quality: Composite quality in ``[0, 1]``.

        Returns:
            ``quality`` raised to ``quality_power``.
        """
        return float(np.clip(quality, 0.0, 1.0) ** self._config.quality_power)

    def drop(self, track_id: str) -> None:
        """Discard a track's buffer.

        Called when a track is removed, so memory does not grow with the number
        of tracks ever seen rather than the number currently alive.

        Args:
            track_id: The track to forget.
        """
        self._buffers.pop(track_id, None)

    def reset(self) -> None:
        """Discard every buffer."""
        self._buffers.clear()

    def statistics(self) -> Dict[str, float]:
        """Summarise buffer occupancy.

        Returns:
            Track counts, retained samples and readiness.
        """
        retained = sum(len(b.vectors) for b in self._buffers.values())
        return {
            "tracks": len(self._buffers),
            "ready": len(self.ready_tracks()),
            "retained_samples": retained,
            "total_seen": sum(b.total_seen for b in self._buffers.values()),
            "strategy": self._config.strategy.value,
        }