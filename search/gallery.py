"""FAISS-backed gallery of enrolled identities.

Index choice
------------
``IndexFlatIP`` performs exhaustive inner-product search.  Every embedding in
this pipeline is L2 normalised, so an inner product *is* cosine similarity and
no separate metric is needed.

Exhaustive search is the right default and not a placeholder.  An approximate
index trades recall for speed, and recall loss in an open-set system does not
degrade gracefully: a missed nearest neighbour silently becomes an ``UNKNOWN``,
or worse, promotes the second-best match into a confident wrong identity.  A
flat index over a few thousand 512-dimensional vectors is microseconds on CPU,
which is far below the cost of a single detector forward pass.  IVF only starts
paying for itself past roughly a million identities, and this module refuses to
build one below the point where its clustering is statistically meaningful.

Consent
-------
Enrolment is the moment a person's biometric identifier enters the system.
Under GDPR Art. 9 and India's DPDP Act that requires a lawful basis, and the
practical consequence is that a gallery should not be constructible from a
folder of images someone happened to have.  :class:`Gallery` therefore refuses
to build without a consent manifest when
``GovernanceConfig.require_consent_manifest`` is set.  It is a deliberate
speed bump, not a compliance guarantee.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from configs.config import GovernanceConfig, IndexType, SearchConfig
from utils.log import get_logger

__all__ = ["GalleryError", "GalleryEntry", "SearchHit", "Gallery"]

LOGGER = get_logger(__name__)

#: Dimensionality every enrolled embedding must have.
EMBEDDING_DIM = 512

#: Below this many identities an IVF index has too few points per cell for its
#: clustering to mean anything, so a flat index is used instead.
_MIN_IVF_IDENTITIES = 10_000

#: Filename of the consent manifest expected alongside enrolment data.
CONSENT_MANIFEST = "consent.json"


class GalleryError(RuntimeError):
    """Raised when a gallery cannot be built, searched, or persisted."""


@dataclass(frozen=True, slots=True)
class GalleryEntry:
    """One enrolled reference embedding.

    An identity may have several entries.  Multiple enrolments per person --
    different lighting, pose, or capture date -- raise recall substantially,
    because a single reference embedding only covers one region of that
    person's appearance manifold.

    Attributes:
        identity: Label returned on a match.
        vector: L2-normalised ``float32`` array of shape ``(512,)``.
        source: Provenance of the enrolment, for the audit trail.
        metadata: Arbitrary extra fields carried through persistence.
    """

    identity: str
    vector: np.ndarray
    source: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the label and the embedding.

        Raises:
            GalleryError: If the identity is empty or the vector is malformed.
        """
        if not self.identity or not self.identity.strip():
            raise GalleryError("Gallery entries require a non-empty identity")
        if self.vector.shape != (EMBEDDING_DIM,):
            raise GalleryError(
                f"Entry {self.identity!r} has shape {self.vector.shape}, "
                f"expected ({EMBEDDING_DIM},)"
            )
        magnitude = float(np.linalg.norm(self.vector))
        if not np.isclose(magnitude, 1.0, atol=1e-3):
            raise GalleryError(
                f"Entry {self.identity!r} is not unit length ({magnitude:.6f}); "
                "inner-product search would not be cosine similarity"
            )


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One neighbour returned by a gallery search.

    Attributes:
        identity: Label of the matched entry.
        similarity: Cosine similarity in ``[-1, 1]``.
        entry_index: Index of the entry within the gallery.
        source: Provenance of the matched enrolment.
    """

    identity: str
    similarity: float
    entry_index: int
    source: str = ""

    def __repr__(self) -> str:
        """Return a compact debug representation."""
        return f"SearchHit({self.identity}, {self.similarity:.4f})"


class Gallery:
    """A searchable collection of enrolled identities.

    Args:
        config: Index type and search parameters.
        governance: Consent and retention policy.
    """

    def __init__(
        self, config: SearchConfig, governance: Optional[GovernanceConfig] = None
    ) -> None:
        self._config = config
        self._governance = governance or GovernanceConfig()
        self._entries: List[GalleryEntry] = []
        self._index = None
        self._built = False

    # -- properties -------------------------------------------------------- #
    @property
    def size(self) -> int:
        """Number of enrolled entries."""
        return len(self._entries)

    @property
    def identities(self) -> List[str]:
        """Distinct identity labels, sorted."""
        return sorted({entry.identity for entry in self._entries})

    @property
    def is_built(self) -> bool:
        """Whether the index is built and searchable."""
        return self._built

    @property
    def entries(self) -> List[GalleryEntry]:
        """A copy of the enrolled entries."""
        return list(self._entries)

    def __len__(self) -> int:
        """Number of enrolled entries."""
        return len(self._entries)

    # -- enrolment --------------------------------------------------------- #
    def add(self, entry: GalleryEntry) -> None:
        """Enrol one entry.

        Invalidates the index, which must be rebuilt before searching. Adding
        to a live index without rebuilding is possible for flat indices but not
        for IVF, and silently supporting one but not the other would make the
        gallery's behaviour depend on its size.

        Args:
            entry: The entry to enrol.
        """
        self._entries.append(entry)
        self._built = False

    def add_many(self, entries: Sequence[GalleryEntry]) -> None:
        """Enrol several entries.

        Args:
            entries: The entries to enrol.
        """
        self._entries.extend(entries)
        self._built = False

    def build(self) -> "Gallery":
        """Construct the FAISS index over the enrolled entries.

        Returns:
            ``self``, to allow fluent chaining.

        Raises:
            GalleryError: If the gallery is empty or FAISS is unavailable.
        """
        if not self._entries:
            raise GalleryError("Cannot build an index over an empty gallery")

        try:
            import faiss
        except ImportError as exc:
            raise GalleryError("faiss is required: pip install faiss-cpu") from exc

        matrix = np.ascontiguousarray(
            np.stack([entry.vector for entry in self._entries]).astype(np.float32)
        )

        index_type = self._config.index_type
        if index_type is IndexType.IVF_FLAT_IP and len(self._entries) < _MIN_IVF_IDENTITIES:
            LOGGER.info(
                "Only %d entries; using a flat index instead of IVF, whose "
                "clustering needs at least %d points to be meaningful",
                len(self._entries),
                _MIN_IVF_IDENTITIES,
            )
            index_type = IndexType.FLAT_IP

        if index_type is IndexType.FLAT_IP:
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
        elif index_type is IndexType.FLAT_L2:
            index = faiss.IndexFlatL2(EMBEDDING_DIM)
        else:
            quantiser = faiss.IndexFlatIP(EMBEDDING_DIM)
            index = faiss.IndexIVFFlat(
                quantiser, EMBEDDING_DIM, self._config.nlist, faiss.METRIC_INNER_PRODUCT
            )
            index.train(matrix)
            index.nprobe = self._config.nprobe

        index.add(matrix)
        self._index = index
        self._built = True

        LOGGER.info(
            "Built %s over %d entries covering %d identities",
            index_type.value,
            len(self._entries),
            len(self.identities),
        )
        return self

    # -- search ------------------------------------------------------------ #
    def search(self, query: np.ndarray, top_k: Optional[int] = None) -> List[SearchHit]:
        """Find the nearest enrolled entries to a query embedding.

        Args:
            query: L2-normalised ``float32`` array of shape ``(512,)``.
            top_k: Neighbours to return; defaults to the configured value.

        Returns:
            Hits ordered by descending similarity.  Shorter than ``top_k`` when
            the gallery holds fewer entries.

        Raises:
            GalleryError: If the index is not built or the query is malformed.
        """
        if not self._built:
            raise GalleryError("Gallery index not built; call build() first")
        if query.shape != (EMBEDDING_DIM,):
            raise GalleryError(f"Query must have shape ({EMBEDDING_DIM},), got {query.shape}")

        k = min(top_k or self._config.top_k, len(self._entries))
        vector = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32))
        similarities, indices = self._index.search(vector, k)

        hits: List[SearchHit] = []
        for similarity, entry_index in zip(similarities[0], indices[0]):
            # FAISS returns -1 for unfilled slots when k exceeds the index size.
            if entry_index < 0:
                continue
            entry = self._entries[int(entry_index)]
            hits.append(
                SearchHit(
                    identity=entry.identity,
                    similarity=float(similarity),
                    entry_index=int(entry_index),
                    source=entry.source,
                )
            )
        return hits

    def search_batch(
        self, queries: np.ndarray, top_k: Optional[int] = None
    ) -> List[List[SearchHit]]:
        """Search several queries in one FAISS call.

        Args:
            queries: ``float32`` array of shape ``(N, 512)`` with unit rows.
            top_k: Neighbours per query; defaults to the configured value.

        Returns:
            One hit list per query, in input order.

        Raises:
            GalleryError: If the index is not built or the shape is wrong.
        """
        if not self._built:
            raise GalleryError("Gallery index not built; call build() first")
        if queries.ndim != 2 or queries.shape[1] != EMBEDDING_DIM:
            raise GalleryError(
                f"Queries must have shape (N, {EMBEDDING_DIM}), got {queries.shape}"
            )

        k = min(top_k or self._config.top_k, len(self._entries))
        matrix = np.ascontiguousarray(queries.astype(np.float32))
        similarities, indices = self._index.search(matrix, k)

        results: List[List[SearchHit]] = []
        for row_similarities, row_indices in zip(similarities, indices):
            hits: List[SearchHit] = []
            for similarity, entry_index in zip(row_similarities, row_indices):
                if entry_index < 0:
                    continue
                entry = self._entries[int(entry_index)]
                hits.append(
                    SearchHit(
                        identity=entry.identity,
                        similarity=float(similarity),
                        entry_index=int(entry_index),
                        source=entry.source,
                    )
                )
            results.append(hits)
        return results

    # -- impostor statistics ----------------------------------------------- #
    def impostor_similarities(self, sample_limit: int = 4096) -> np.ndarray:
        """Collect similarities between entries of *different* identities.

        This is the empirical distribution of what a non-match looks like in
        this particular gallery, and it is what threshold calibration needs.
        A threshold chosen without it is a guess about a distribution that
        varies with encoder, camera and population.

        Args:
            sample_limit: Maximum entries used, to bound the pairwise cost.

        Returns:
            A one-dimensional array of impostor similarities, empty when the
            gallery holds fewer than two identities.
        """
        if len(self.identities) < 2:
            return np.zeros(0, dtype=np.float32)

        count = min(len(self._entries), sample_limit)
        indices = np.arange(count)
        if len(self._entries) > sample_limit:
            indices = np.random.default_rng(0).choice(
                len(self._entries), size=sample_limit, replace=False
            )

        matrix = np.stack([self._entries[i].vector for i in indices]).astype(np.float32)
        labels = np.asarray([self._entries[i].identity for i in indices])

        gram = matrix @ matrix.T
        different = labels[:, None] != labels[None, :]
        upper = np.triu(np.ones_like(different, dtype=bool), k=1)
        return gram[different & upper].astype(np.float32)

    # -- persistence ------------------------------------------------------- #
    def save(self, directory: Union[str, Path]) -> Path:
        """Persist the gallery to disk.

        Vectors are stored in a ``.npy`` file and labels in JSON, rather than
        via FAISS serialisation, so the gallery survives a FAISS version change
        and can be inspected without it.

        Args:
            directory: Destination directory, created if absent.

        Returns:
            The resolved directory.

        Raises:
            GalleryError: If the gallery is empty.
        """
        if not self._entries:
            raise GalleryError("Cannot save an empty gallery")

        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)

        vectors = np.stack([entry.vector for entry in self._entries]).astype(np.float32)
        np.save(directory / "vectors.npy", vectors)

        payload = {
            "embedding_dim": EMBEDDING_DIM,
            "entries": [
                {
                    "identity": entry.identity,
                    "source": entry.source,
                    "metadata": entry.metadata,
                }
                for entry in self._entries
            ],
        }
        (directory / "gallery.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

        LOGGER.info("Saved %d entries to %s", len(self._entries), directory)
        return directory

    @classmethod
    def load(
        cls,
        directory: Union[str, Path],
        config: SearchConfig,
        governance: Optional[GovernanceConfig] = None,
    ) -> "Gallery":
        """Load a gallery from disk and build its index.

        Args:
            directory: Directory previously written by :meth:`save`.
            config: Index type and search parameters.
            governance: Consent and retention policy.

        Returns:
            The loaded, built gallery.

        Raises:
            GalleryError: If files are missing or inconsistent.
        """
        directory = Path(directory).expanduser().resolve()
        vectors_path = directory / "vectors.npy"
        manifest_path = directory / "gallery.json"

        if not vectors_path.is_file() or not manifest_path.is_file():
            raise GalleryError(f"No gallery found in {directory}")

        vectors = np.load(vectors_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = payload.get("entries", [])

        if len(records) != len(vectors):
            raise GalleryError(
                f"Gallery is inconsistent: {len(records)} record(s) "
                f"but {len(vectors)} vector(s)"
            )

        gallery = cls(config, governance)
        gallery.add_many(
            [
                GalleryEntry(
                    identity=record["identity"],
                    vector=np.ascontiguousarray(vector, dtype=np.float32),
                    source=record.get("source", ""),
                    metadata=record.get("metadata", {}),
                )
                for record, vector in zip(records, vectors)
            ]
        )
        return gallery.build()

    # -- governance -------------------------------------------------------- #
    def verify_consent(self, directory: Union[str, Path]) -> None:
        """Check that enrolment data is accompanied by a consent manifest.

        Args:
            directory: Directory holding the enrolment data.

        Raises:
            GalleryError: If the manifest is required and missing, malformed,
                or does not cover every enrolled identity.
        """
        if not self._governance.require_consent_manifest:
            LOGGER.warning(
                "Consent manifest checking is disabled; enrolling biometric "
                "identifiers without a recorded lawful basis"
            )
            return

        manifest_path = Path(directory).expanduser().resolve() / CONSENT_MANIFEST
        if not manifest_path.is_file():
            raise GalleryError(
                f"No {CONSENT_MANIFEST} in {directory}. Enrolment records a "
                "biometric identifier, which needs a lawful basis under GDPR "
                "Art. 9 and India's DPDP Act 2023. Provide a manifest listing "
                "each identity and its basis, or set "
                "governance.require_consent_manifest=false to accept "
                "responsibility explicitly."
            )

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise GalleryError(f"{CONSENT_MANIFEST} is not valid JSON: {exc}") from exc

        consented = {record.get("identity") for record in manifest.get("identities", [])}
        missing = set(self.identities) - consented
        if missing:
            raise GalleryError(
                f"{len(missing)} identity(ies) have no consent record: "
                f"{sorted(missing)[:5]}"
            )
        LOGGER.info("Consent verified for %d identity(ies)", len(consented))