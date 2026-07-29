"""Tests for gallery search and open-set identification.

A note on the synthetic data
----------------------------
Embeddings are 512-dimensional unit vectors.  Adding per-component Gaussian
noise of standard deviation ``s`` produces a perturbation whose norm is
``s * sqrt(512)`` -- roughly ``22.6 s``.  So ``s = 0.25`` yields a noise vector
five times longer than the signal and destroys it entirely, while ``s = 0.04``
gives a genuine-pair similarity around 0.74, which matches what real face
encoders produce for the same person.  The constant below is not arbitrary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from configs import SurveillanceConfig
from recognition.fusion import FusedEmbedding
from search.gallery import Gallery, GalleryEntry, GalleryError
from search.open_set import OpenSetIdentifier

pytest.importorskip("faiss", reason="faiss-cpu is required for gallery search")

#: Intra-identity spread that reproduces realistic genuine similarity (~0.74).
SIGMA = 0.04


def unit(vector: np.ndarray) -> np.ndarray:
    """Normalise to unit length as float32."""
    return (vector / np.linalg.norm(vector)).astype(np.float32)


@pytest.fixture()
def config() -> SurveillanceConfig:
    """A stock configuration."""
    return SurveillanceConfig.default()


@pytest.fixture()
def identity_centres() -> Dict[str, np.ndarray]:
    """Fifty well-separated identity centroids."""
    rng = np.random.default_rng(0)
    return {f"person_{i:03d}": unit(rng.normal(size=512)) for i in range(50)}


@pytest.fixture()
def gallery(config, identity_centres) -> Gallery:
    """A built gallery of fifty identities with three enrolments each."""
    rng = np.random.default_rng(1)
    built = Gallery(config.search, config.governance)
    for name, centre in identity_centres.items():
        for enrolment in range(3):
            built.add(
                GalleryEntry(
                    identity=name,
                    vector=unit(centre + rng.normal(0, SIGMA, 512)),
                    source=f"enrol_{enrolment}.jpg",
                )
            )
    return built.build()


def make_fused(vector: np.ndarray, track_id: str, coherence: float = 0.95) -> FusedEmbedding:
    """Wrap a vector as a fused track embedding."""
    return FusedEmbedding(
        vector=unit(vector),
        track_id=track_id,
        sample_count=10,
        mean_quality=0.8,
        coherence=coherence,
        strategy="weighted_mean",
    )


# --------------------------------------------------------------------------- #
# Gallery entries
# --------------------------------------------------------------------------- #
def test_entry_requires_unit_vector() -> None:
    """Inner-product search is only cosine similarity for unit vectors."""
    with pytest.raises(GalleryError, match="unit length"):
        GalleryEntry(identity="x", vector=np.ones(512, dtype=np.float32))


def test_entry_requires_correct_dimensionality() -> None:
    """A wrong-sized vector indicates an encoder mismatch."""
    with pytest.raises(GalleryError, match="shape"):
        GalleryEntry(identity="x", vector=np.zeros(256, dtype=np.float32))


def test_entry_requires_identity() -> None:
    """An unlabelled entry could never be returned meaningfully."""
    with pytest.raises(GalleryError, match="identity"):
        GalleryEntry(identity="   ", vector=unit(np.ones(512)))


# --------------------------------------------------------------------------- #
# Gallery
# --------------------------------------------------------------------------- #
def test_empty_gallery_cannot_build(config) -> None:
    """Building over nothing must raise rather than produce an empty index."""
    with pytest.raises(GalleryError, match="empty"):
        Gallery(config.search, config.governance).build()


def test_search_requires_build(config, identity_centres) -> None:
    """Searching an unbuilt gallery must raise."""
    unbuilt = Gallery(config.search, config.governance)
    unbuilt.add(GalleryEntry(identity="a", vector=unit(np.ones(512))))
    with pytest.raises(GalleryError, match="not built"):
        unbuilt.search(unit(np.ones(512)))


def test_gallery_reports_size_and_identities(gallery: Gallery) -> None:
    """Entry and identity counts must be distinct and correct."""
    assert gallery.size == 150
    assert len(gallery.identities) == 50


def test_search_returns_ranked_hits(gallery: Gallery, identity_centres) -> None:
    """The correct identity must rank first for a genuine query."""
    rng = np.random.default_rng(7)
    query = unit(identity_centres["person_007"] + rng.normal(0, SIGMA, 512))
    hits = gallery.search(query, top_k=5)

    assert len(hits) == 5
    assert hits[0].identity == "person_007"
    assert hits == sorted(hits, key=lambda hit: -hit.similarity)


def test_top_k_is_capped_by_gallery_size(config) -> None:
    """Requesting more neighbours than exist must not emit padding entries."""
    small = Gallery(config.search, config.governance)
    small.add(GalleryEntry(identity="a", vector=unit(np.ones(512))))
    small.build()
    assert len(small.search(unit(np.ones(512)), top_k=10)) == 1


def test_batch_search_matches_single(gallery: Gallery, identity_centres) -> None:
    """Batched search must agree with per-query search."""
    queries = np.stack([identity_centres[f"person_{i:03d}"] for i in range(5)])
    batched = gallery.search_batch(queries, top_k=3)
    assert len(batched) == 5
    for index, hits in enumerate(batched):
        assert hits[0].identity == gallery.search(queries[index], top_k=3)[0].identity


def test_malformed_query_rejected(gallery: Gallery) -> None:
    """A wrong-sized query must raise rather than search garbage."""
    with pytest.raises(GalleryError, match="shape"):
        gallery.search(np.zeros(256, dtype=np.float32))


def test_impostor_similarities_exclude_same_identity(gallery: Gallery) -> None:
    """Impostor statistics must compare different people only."""
    impostors = gallery.impostor_similarities()
    assert impostors.size > 1000
    # Genuine pairs sit near 0.74; impostors must be far below that.
    assert float(impostors.max()) < 0.4


def test_ivf_falls_back_to_flat_when_small(config, identity_centres) -> None:
    """IVF clustering is meaningless on a small gallery and must not be used."""
    ivf_config = SurveillanceConfig.from_dict(
        {"search": {"index_type": "IndexIVFFlatIP"}}
    ).search
    small = Gallery(ivf_config, config.governance)
    for name, centre in list(identity_centres.items())[:5]:
        small.add(GalleryEntry(identity=name, vector=centre))
    small.build()
    assert small.search(identity_centres["person_000"])[0].identity == "person_000"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_save_and_load_round_trip(gallery: Gallery, config, tmp_path) -> None:
    """A reloaded gallery must return identical results."""
    gallery.save(tmp_path)
    reloaded = Gallery.load(tmp_path, config.search)

    assert reloaded.size == gallery.size
    assert reloaded.identities == gallery.identities

    query = gallery.entries[0].vector
    assert reloaded.search(query)[0].identity == gallery.search(query)[0].identity


def test_load_missing_gallery(config, tmp_path) -> None:
    """Loading from an empty directory must raise."""
    with pytest.raises(GalleryError, match="No gallery"):
        Gallery.load(tmp_path, config.search)


def test_load_detects_inconsistency(gallery: Gallery, config, tmp_path) -> None:
    """A vector/label count mismatch must be caught, not silently truncated."""
    gallery.save(tmp_path)
    payload = json.loads((tmp_path / "gallery.json").read_text())
    payload["entries"] = payload["entries"][:-5]
    (tmp_path / "gallery.json").write_text(json.dumps(payload))

    with pytest.raises(GalleryError, match="inconsistent"):
        Gallery.load(tmp_path, config.search)


# --------------------------------------------------------------------------- #
# Consent
# --------------------------------------------------------------------------- #
def test_consent_manifest_is_required(gallery: Gallery, tmp_path) -> None:
    """Enrolment records a biometric identifier and needs a lawful basis."""
    with pytest.raises(GalleryError, match="consent"):
        gallery.verify_consent(tmp_path)


def test_consent_manifest_accepted(gallery: Gallery, tmp_path) -> None:
    """A manifest covering every identity must pass."""
    (tmp_path / "consent.json").write_text(
        json.dumps(
            {"identities": [{"identity": n, "basis": "explicit"} for n in gallery.identities]}
        )
    )
    gallery.verify_consent(tmp_path)


def test_partial_consent_rejected(gallery: Gallery, tmp_path) -> None:
    """Consent for some identities must not authorise the rest."""
    (tmp_path / "consent.json").write_text(
        json.dumps({"identities": [{"identity": gallery.identities[0]}]})
    )
    with pytest.raises(GalleryError, match="no consent record"):
        gallery.verify_consent(tmp_path)


def test_consent_check_can_be_waived(gallery: Gallery, tmp_path) -> None:
    """Disabling the check must be possible but explicit."""
    waived = SurveillanceConfig.from_dict(
        {"governance": {"require_consent_manifest": False}}
    )
    gallery._governance = waived.governance  # noqa: SLF001
    gallery.verify_consent(tmp_path)


# --------------------------------------------------------------------------- #
# Open-set identification
# --------------------------------------------------------------------------- #
def test_enrolled_person_is_identified(gallery, config, identity_centres) -> None:
    """A genuine query must resolve to the right identity."""
    identifier = OpenSetIdentifier(gallery, config.search)
    rng = np.random.default_rng(7)
    query = make_fused(identity_centres["person_007"] + rng.normal(0, SIGMA, 512), "t")

    result = identifier.identify(query)
    assert result.accepted is True
    assert result.identity == "person_007"


def test_stranger_is_rejected(gallery, config) -> None:
    """Someone not in the gallery must return UNKNOWN.

    This is the central requirement: nearest-neighbour search always returns a
    neighbour, so without an explicit floor every stranger is labelled.
    """
    identifier = OpenSetIdentifier(gallery, config.search)
    rng = np.random.default_rng(99)
    result = identifier.identify(make_fused(rng.normal(size=512), "t"))

    assert result.accepted is False
    assert result.identity == "UNKNOWN"
    assert "threshold" in result.rejection_reason


def test_ambiguous_match_is_rejected_by_margin(gallery, config, identity_centres) -> None:
    """A query between two identities must be declined despite high similarity.

    The similarity floor alone accepts this: it clears 0.35 comfortably. Only
    the margin check catches that the decision is a coin flip.
    """
    identifier = OpenSetIdentifier(gallery, config.search)
    midpoint = identity_centres["person_001"] + identity_centres["person_002"]
    result = identifier.identify(make_fused(midpoint, "t"))

    assert result.similarity > config.search.similarity_threshold
    assert result.accepted is False
    assert "ambiguous" in result.rejection_reason


def test_incoherent_track_is_rejected(gallery, config, identity_centres) -> None:
    """A track holding two people fuses to a vector describing neither."""
    identifier = OpenSetIdentifier(gallery, config.search)
    result = identifier.identify(
        make_fused(identity_centres["person_007"], "t", coherence=0.45)
    )
    assert result.accepted is False
    assert "incoherent" in result.rejection_reason


def test_margin_ignores_the_same_identity(gallery, config, identity_centres) -> None:
    """Multiple enrolments of one person are not competing hypotheses.

    Treating them as such would penalise the multi-enrolment strategy that
    improves recall in the first place.
    """
    identifier = OpenSetIdentifier(gallery, config.search)
    rng = np.random.default_rng(3)
    result = identifier.identify(
        make_fused(identity_centres["person_003"] + rng.normal(0, SIGMA, 512), "t")
    )
    assert result.accepted is True
    assert result.margin > config.search.margin_threshold


def test_single_identity_gallery_has_infinite_margin(config, identity_centres) -> None:
    """With one enrolled identity there is nothing to be confused with."""
    solo = Gallery(config.search, config.governance)
    solo.add(GalleryEntry(identity="only", vector=identity_centres["person_000"]))
    solo.build()

    identifier = OpenSetIdentifier(solo, config.search)
    result = identifier.identify(make_fused(identity_centres["person_000"], "t"))
    assert np.isinf(result.margin)
    assert result.accepted is True


def test_no_false_identifications_across_many_strangers(gallery, config) -> None:
    """The whole point: strangers must not be named."""
    identifier = OpenSetIdentifier(gallery, config.search)
    rng = np.random.default_rng(11)
    accepted = sum(
        identifier.identify(make_fused(rng.normal(size=512), f"s{i}")).accepted
        for i in range(500)
    )
    assert accepted == 0


def test_enrolled_people_are_recognised(gallery, config, identity_centres) -> None:
    """Rejection must not come at the cost of recognising real matches."""
    identifier = OpenSetIdentifier(gallery, config.search)
    rng = np.random.default_rng(13)
    correct = sum(
        identifier.identify(make_fused(centre + rng.normal(0, SIGMA, 512), name)).identity
        == name
        for name, centre in identity_centres.items()
    )
    assert correct == len(identity_centres)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_calibration_uses_impostor_statistics(gallery, config) -> None:
    """The threshold must be derived from this gallery, not transplanted."""
    identifier = OpenSetIdentifier(gallery, config.search)
    assert identifier.is_calibrated is False
    identifier.calibrate()
    assert identifier.is_calibrated is True


def test_calibration_never_lowers_the_floor(gallery, config) -> None:
    """A well-separated gallery must not calibrate the threshold downward."""
    identifier = OpenSetIdentifier(gallery, config.search)
    identifier.calibrate()
    assert identifier.threshold >= config.search.similarity_threshold


def test_calibration_declines_on_insufficient_data(config, identity_centres) -> None:
    """Too few impostor pairs must leave the configured threshold alone."""
    tiny = Gallery(config.search, config.governance)
    for name in list(identity_centres)[:3]:
        tiny.add(GalleryEntry(identity=name, vector=identity_centres[name]))
    tiny.build()

    identifier = OpenSetIdentifier(tiny, config.search)
    before = identifier.threshold
    assert identifier.calibrate() == before
    assert identifier.is_calibrated is False


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_rejections_carry_their_candidates(gallery, config) -> None:
    """A declined decision must remain reviewable after the fact."""
    identifier = OpenSetIdentifier(gallery, config.search)
    rng = np.random.default_rng(21)
    result = identifier.identify(make_fused(rng.normal(size=512), "t"))
    assert result.candidates
    assert result.rejection_reason


def test_identification_serialises_for_audit(gallery, config, identity_centres) -> None:
    """Decisions must be recordable in the JSONL audit trail."""
    identifier = OpenSetIdentifier(gallery, config.search)
    result = identifier.identify(make_fused(identity_centres["person_000"], "t"))
    payload = result.to_dict()
    json.dumps(payload)
    assert payload["track_id"] == "t"
    assert "accepted" in payload


def test_statistics_track_decisions(gallery, config, identity_centres) -> None:
    """The identifier must report how often it declines."""
    identifier = OpenSetIdentifier(gallery, config.search)
    rng = np.random.default_rng(31)
    for name, centre in list(identity_centres.items())[:10]:
        identifier.identify(make_fused(centre + rng.normal(0, SIGMA, 512), name))
    for index in range(10):
        identifier.identify(make_fused(rng.normal(size=512), f"s{index}"))

    statistics = identifier.statistics()
    assert statistics["accepted"] == 10
    assert statistics["rejected"] == 10
    assert statistics["acceptance_rate"] == pytest.approx(0.5)