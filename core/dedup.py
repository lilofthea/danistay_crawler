import redis
from config import redis_cfg
import logging

logger = logging.getLogger(__name__)


class DedupFilter:
    """
    Redis Bloom filter for document-level deduplication.
    False positive rate: 0.1% (we'd rather miss a rare duplicate
    than re-embed millions of already-indexed documents).
    """

    def __init__(self):
        self.r = redis.Redis(
            host=redis_cfg.host,
            port=redis_cfg.port,
            decode_responses=True
        )
        self._ensure_filter()

    def _ensure_filter(self):
        """Create bloom filter if it doesn't exist."""
        try:
            self.r.execute_command(
                "BF.RESERVE",
                redis_cfg.bloom_filter_key,
                redis_cfg.bloom_filter_error_rate,
                redis_cfg.bloom_filter_capacity,
                "NONSCALING"
            )
            logger.info("Bloom filter created.")
        except redis.ResponseError as e:
            if "EXISTS" in str(e).upper() or "item exists" in str(e):
                logger.info("Bloom filter already exists, reusing.")
            else:
                raise

    def is_seen(self, doc_id: str) -> bool:
        """Returns True if this doc_id was already processed."""
        return bool(self.r.execute_command(
            "BF.EXISTS", redis_cfg.bloom_filter_key, doc_id
        ))

    def mark_seen(self, doc_id: str) -> None:
        """Mark doc_id as processed."""
        self.r.execute_command(
            "BF.ADD", redis_cfg.bloom_filter_key, doc_id
        )

    def check_and_mark(self, doc_id: str) -> bool:
        """
        Atomic check-and-mark. Returns True if NEW (should process),
        False if already seen (should skip).
        """
        result = self.r.execute_command(
            "BF.ADD", redis_cfg.bloom_filter_key, doc_id
        )
        # BF.ADD returns 1 if newly added, 0 if already existed
        return result == 1
