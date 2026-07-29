"""search package."""
"""Gallery search and open-set identification."""

from search.gallery import Gallery, GalleryEntry, GalleryError, SearchHit
from search.open_set import Identification, OpenSetIdentifier

__all__ = [
    "Gallery",
    "GalleryEntry",
    "GalleryError",
    "Identification",
    "OpenSetIdentifier",
    "SearchHit",
]