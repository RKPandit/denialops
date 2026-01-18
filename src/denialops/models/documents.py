"""Pydantic models for document handling."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Types of documents that can be uploaded."""

    DENIAL_LETTER = "denial_letter"
    EOB = "eob"
    SBC = "sbc"
    EOC = "eoc"


class UploadedDocument(BaseModel):
    """Metadata for an uploaded document."""

    document_id: str = Field(..., description="Unique identifier for this document")
    case_id: str = Field(..., description="Case this document belongs to")
    doc_type: DocumentType = Field(..., description="Type of document")
    original_filename: str = Field(..., description="Original filename from upload")
    stored_path: str = Field(..., description="Path where document is stored")
    uploaded_at: datetime = Field(..., description="When document was uploaded")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    content_type: str | None = Field(None, description="MIME type of the file")


class ExtractedPage(BaseModel):
    """Text extracted from a single page."""

    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    text: str = Field(..., description="Extracted text content")
    char_offset_start: int = Field(..., ge=0, description="Character offset in full text")
    char_offset_end: int = Field(..., ge=0, description="End character offset")


class ExtractedText(BaseModel):
    """Text extracted from a document."""

    document_id: str = Field(..., description="Source document ID")
    total_pages: int = Field(..., ge=1, description="Total number of pages")
    pages: list[ExtractedPage] = Field(..., description="Text by page")
    full_text: str = Field(..., description="Concatenated full text")
    extraction_method: str = Field(..., description="Method used (pdfminer, passthrough)")

    @property
    def text(self) -> str:
        """Alias for full_text for convenience."""
        return self.full_text


class ArtifactInfo(BaseModel):
    """Information about a generated artifact."""

    name: str = Field(..., description="Artifact filename")
    path: str = Field(..., description="Full path to artifact")
    size: int = Field(..., ge=0, description="File size in bytes")
    created_at: datetime = Field(..., description="When artifact was created")
    artifact_type: str = Field(..., description="Type of artifact (json, md)")
