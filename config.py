from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class RedpandaConfig:
    bootstrap_servers: List[str] = field(
        default_factory=lambda: os.getenv(
            "REDPANDA_BROKERS", "localhost:19092"
        ).split(",")
    )
    # Per-source topics: each source is different data, so it gets its own
    # raw + chunked topic (clean lag/retention/monitoring per source).
    # See raw_topic() / chunked_topic() below.
    topic_dlq: str = "legal.dlq"
    # Producer
    producer_acks: str = "all"
    producer_retries: int = 5
    # Consumer groups — one per source, so each worker is fully isolated
    consumer_auto_offset_reset: str = "earliest"
    consumer_enable_auto_commit: bool = False  # manual commit for reliability


def chunker_group(source: str) -> str:
    return f"chunker-{source}"


def embedder_group(source: str) -> str:
    return f"embedder-{source}"


SOURCES = ("danistay", "yargitay", "mevzuat")


def raw_topic(source: str) -> str:
    return f"legal.{source}.raw"


def chunked_topic(source: str) -> str:
    return f"legal.{source}.chunked"


@dataclass
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    bloom_filter_key: str = "legal:seen_docs"
    bloom_filter_capacity: int = 5_000_000   # 5M docs
    bloom_filter_error_rate: float = 0.001   # 0.1% false positive rate


@dataclass
class QdrantConfig:
    # NOTE: this box runs Qdrant mapped to host port 6490 (docker: qdrant_server)
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6490"))
    collection_danistay: str = "danistay_kararlar"
    collection_yargitay: str = "yargitay_kararlar"
    collection_mevzuat: str = "mevzuat"
    # Qwen3-Embedding-4B truncated to 1024 dims (matches yargitay-hukuk/rag pipeline).
    # Use 1536 for BGE-M3, 2560 for full-dim Qwen3.
    vector_size: int = int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))


@dataclass
class EmbedderConfig:
    # "Qwen/Qwen3-Embedding-4B" (cached locally, used by yargitay RAG) | "BAAI/bge-m3"
    model_name: str = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
    # Optional explicit local path (skip HF lookup entirely)
    model_path: str = os.getenv("EMBED_MODEL_PATH", "")
    batch_size: int = 32
    device: str = os.getenv("EMBED_DEVICE", "cuda")  # or "cpu"
    max_length: int = 8192


@dataclass
class ChunkerConfig:
    chunk_size: int = 512        # tokens
    chunk_overlap: int = 64      # tokens
    min_chunk_size: int = 50     # discard tiny chunks


# Singleton instances
redpanda = RedpandaConfig()
redis_cfg = RedisConfig()
qdrant = QdrantConfig()
embedder = EmbedderConfig()
chunker = ChunkerConfig()
