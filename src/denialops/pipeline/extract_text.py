"""Text extraction from PDF and text documents."""

from pathlib import Path

from pdfminer.high_level import extract_text as pdfminer_extract
from pdfminer.pdfpage import PDFPage

from denialops.models.documents import ExtractedPage, ExtractedText


def extract_text(file_path: Path | str) -> ExtractedText:
    """
    Extract text from a PDF or text file.

    Args:
        file_path: Path to the document

    Returns:
        ExtractedText with pages and full text
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Handle text files directly
    if file_path.suffix.lower() == ".txt":
        return _extract_from_text(file_path)

    # Handle PDF files
    if file_path.suffix.lower() == ".pdf":
        return _extract_from_pdf(file_path)

    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def _extract_from_text(file_path: Path) -> ExtractedText:
    """Extract text from a plain text file."""
    content = file_path.read_text(encoding="utf-8", errors="replace")

    return ExtractedText(
        document_id=file_path.stem,
        total_pages=1,
        pages=[
            ExtractedPage(
                page_number=1,
                text=content,
                char_offset_start=0,
                char_offset_end=len(content),
            )
        ],
        full_text=content,
        extraction_method="passthrough",
    )


def _extract_from_pdf(file_path: Path) -> ExtractedText:
    """Extract text from a PDF file using pdfminer."""
    # Count pages first
    with open(file_path, "rb") as f:
        num_pages = sum(1 for _ in PDFPage.get_pages(f))

    # Extract text page by page
    pages: list[ExtractedPage] = []
    full_text_parts: list[str] = []
    char_offset = 0

    for page_num in range(1, num_pages + 1):
        # Extract single page
        page_text = pdfminer_extract(
            str(file_path),
            page_numbers=[page_num - 1],  # 0-indexed
        )

        # Clean up text
        page_text = page_text.strip()

        pages.append(
            ExtractedPage(
                page_number=page_num,
                text=page_text,
                char_offset_start=char_offset,
                char_offset_end=char_offset + len(page_text),
            )
        )

        full_text_parts.append(page_text)
        char_offset += len(page_text) + 1  # +1 for newline between pages

    full_text = "\n".join(full_text_parts)

    return ExtractedText(
        document_id=file_path.stem,
        total_pages=num_pages,
        pages=pages,
        full_text=full_text,
        extraction_method="pdfminer",
    )
