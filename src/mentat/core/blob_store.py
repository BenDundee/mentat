"""Local filesystem blob storage.

Provides a pluggable interface for storing and retrieving binary blobs.
All blobs are written to ``data/blobs/`` by default.  The interface
deliberately mirrors what an S3-compatible client would expose so that
swapping to remote storage later requires only a new implementation.
"""

from pathlib import Path

from mentat.core.logging import get_logger
from mentat.core.settings import settings

logger = get_logger(__name__)


class BlobStore:
    """Store and retrieve binary blobs on the local filesystem.

    Args:
        base_dir: Directory under which all blobs are stored.
                  Defaults to ``{settings.data_dir}/blobs``.
    """

    def __init__(self, base_dir: str | None = None) -> None:
        resolved = base_dir or f"{settings.data_dir}/blobs"
        self._base = Path(resolved)
        self._base.mkdir(parents=True, exist_ok=True)
        logger.info("BlobStore ready at %s", self._base)

    def put(self, key: str, data: bytes) -> None:
        """Write *data* to the blob store under *key*.

        Args:
            key: Unique identifier for the blob (e.g. ``upload_id``).
            data: Raw bytes to persist.
        """
        dest = self._base / key
        dest.write_bytes(data)
        logger.debug("BlobStore.put key=%s bytes=%d", key, len(data))

    def get(self, key: str) -> bytes:
        """Retrieve blob bytes for *key*.

        Args:
            key: The blob identifier previously passed to :meth:`put`.

        Returns:
            Raw bytes.

        Raises:
            FileNotFoundError: When no blob exists for *key*.
        """
        path = self._base / key
        if not path.exists():
            raise FileNotFoundError(f"No blob found for key '{key}'")
        data = path.read_bytes()
        logger.debug("BlobStore.get key=%s bytes=%d", key, len(data))
        return data

    def exists(self, key: str) -> bool:
        """Return True if a blob exists for *key*."""
        return (self._base / key).exists()
