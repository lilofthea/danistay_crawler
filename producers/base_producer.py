import json
import logging
from typing import Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError
from config import redpanda, raw_topic
from core.dedup import DedupFilter
from schemas.document import RawDocument

logger = logging.getLogger(__name__)


class LegalDocumentProducer:
    """
    Base producer. Wraps KafkaProducer with:
    - JSON serialization
    - Bloom filter dedup
    - Retry logic
    - Dead letter queue on failure

    One instance per source: LegalDocumentProducer(source="danistay") etc.
    """

    def __init__(self, source: str):
        self.source = source
        self.topic = raw_topic(source)
        self.producer = KafkaProducer(
            bootstrap_servers=redpanda.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks=redpanda.producer_acks,
            retries=redpanda.producer_retries,
            retry_backoff_ms=500,
            max_block_ms=10_000,
        )
        self.dedup = DedupFilter()

    def _make_doc_id(self, doc: RawDocument) -> str:
        """
        Deterministic doc ID for dedup. Prefer natural keys over UUID.
        Danıştay/Yargıtay: kaynak + esas_no + karar_no
        Mevzuat: kaynak + mevzuat_no
        Fallback: URL
        """
        if doc.esas_no and doc.karar_no:
            return f"{doc.kaynak}:{doc.esas_no}:{doc.karar_no}"
        if doc.mevzuat_no:
            return f"{doc.kaynak}:{doc.mevzuat_no}"
        return f"{doc.kaynak}:{doc.url}"

    def produce(self, doc: RawDocument) -> bool:
        """
        Produce a document to legal.raw topic.
        Returns True if produced, False if skipped (dedup).
        """
        doc_id = self._make_doc_id(doc)

        if not self.dedup.check_and_mark(doc_id):
            logger.debug(f"Skipping duplicate: {doc_id}")
            return False

        try:
            future = self.producer.send(
                topic=self.topic,
                key=doc_id,
                value=doc.model_dump(mode="json"),
            )
            future.get(timeout=10)   # block to confirm delivery
            logger.info(f"Produced: {doc_id} → {self.topic}")
            return True

        except KafkaError as e:
            logger.error(f"Failed to produce {doc_id}: {e}")
            self._send_to_dlq(doc, error=str(e))
            return False

    def _send_to_dlq(self, doc: RawDocument, error: str) -> None:
        """Send failed message to dead letter queue."""
        payload = doc.model_dump(mode="json")
        payload["_dlq_error"] = error
        try:
            self.producer.send(
                topic=redpanda.topic_dlq,
                value=payload
            )
        except Exception as e:
            logger.critical(f"DLQ send also failed: {e}")

    def flush(self):
        self.producer.flush()

    def close(self):
        self.producer.close()
