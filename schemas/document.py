from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
import uuid


class RawDocument(BaseModel):
    """Message on legal.raw topic — output of fetchers."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    kaynak: Literal["danistay", "yargitay", "mevzuat"]
    url: str
    text: str                            # cleaned full text
    # Danıştay / Yargıtay specific
    esas_no: Optional[str] = None
    karar_no: Optional[str] = None
    daire: Optional[str] = None
    karar_tarihi: Optional[str] = None   # ISO date string
    # Mevzuat specific
    mevzuat_no: Optional[str] = None
    resmi_gazete_tarihi: Optional[str] = None
    mevzuat_turu: Optional[str] = None   # Kanun, Yönetmelik, Tebliğ, etc.
    # Housekeeping
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict)


class ChunkedDocument(BaseModel):
    """Message on legal.chunked topic — output of chunker worker."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str           # parent RawDocument.id
    kaynak: str
    text: str
    chunk_index: int
    total_chunks: int
    # Inherited metadata from parent
    esas_no: Optional[str] = None
    karar_no: Optional[str] = None
    daire: Optional[str] = None
    karar_tarihi: Optional[str] = None
    mevzuat_no: Optional[str] = None
    mevzuat_turu: Optional[str] = None
    url: str
    metadata: dict = Field(default_factory=dict)
