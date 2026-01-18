"""Storage utilities for case artifacts."""

import json
from datetime import datetime, timezone
from pathlib import Path

from denialops.models.documents import ArtifactInfo


class CaseStorage:
    """Handles storage of case documents and artifacts."""

    def __init__(self, base_path: Path) -> None:
        """Initialize storage with base path."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _case_path(self, case_id: str) -> Path:
        """Get path for a case directory."""
        # Sanitize case_id to prevent path traversal
        safe_id = Path(case_id).name
        if safe_id != case_id or ".." in case_id:
            raise ValueError(f"Invalid case_id: {case_id}")
        return self.base_path / safe_id

    def _documents_path(self, case_id: str) -> Path:
        """Get path for case documents."""
        return self._case_path(case_id) / "documents"

    def _artifacts_path(self, case_id: str) -> Path:
        """Get path for case artifacts."""
        return self._case_path(case_id) / "artifacts"

    def case_exists(self, case_id: str) -> bool:
        """Check if a case exists."""
        try:
            return self._case_path(case_id).exists()
        except ValueError:
            return False

    def create_case(self, case_id: str) -> Path:
        """Create a new case directory."""
        case_path = self._case_path(case_id)
        case_path.mkdir(parents=True, exist_ok=True)
        self._documents_path(case_id).mkdir(exist_ok=True)
        self._artifacts_path(case_id).mkdir(exist_ok=True)
        return case_path

    def store_document(
        self,
        case_id: str,
        document_id: str,
        filename: str,
        content: bytes,
        doc_type: str,
    ) -> str:
        """Store a document and return the storage path."""
        docs_path = self._documents_path(case_id)
        docs_path.mkdir(parents=True, exist_ok=True)

        # Sanitize filename
        safe_filename = Path(filename).name
        ext = safe_filename.rsplit(".", 1)[-1] if "." in safe_filename else "bin"
        stored_filename = f"{document_id}.{ext}"

        file_path = docs_path / stored_filename
        file_path.write_bytes(content)

        # Store document metadata
        metadata_path = docs_path / f"{document_id}.meta.json"
        metadata = {
            "document_id": document_id,
            "original_filename": filename,
            "stored_filename": stored_filename,
            "doc_type": doc_type,
            "size": len(content),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

        return str(file_path)

    def get_document_path(self, case_id: str, document_id: str) -> Path:
        """Get the path to a stored document."""
        docs_path = self._documents_path(case_id)

        # Find document by metadata
        meta_path = docs_path / f"{document_id}.meta.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            return docs_path / metadata["stored_filename"]

        raise FileNotFoundError(f"Document {document_id} not found")

    def list_documents(self, case_id: str) -> list[dict]:
        """List all documents for a case."""
        docs_path = self._documents_path(case_id)
        if not docs_path.exists():
            return []

        documents = []
        for meta_file in docs_path.glob("*.meta.json"):
            metadata = json.loads(meta_file.read_text())
            documents.append(metadata)

        return documents

    def store_artifact(
        self,
        case_id: str,
        filename: str,
        content: str | dict,
    ) -> Path:
        """Store an artifact (JSON or text)."""
        artifacts_path = self._artifacts_path(case_id)
        artifacts_path.mkdir(parents=True, exist_ok=True)

        # Sanitize filename
        safe_filename = Path(filename).name
        file_path = artifacts_path / safe_filename

        if isinstance(content, dict):
            file_path.write_text(json.dumps(content, indent=2, default=str))
        else:
            file_path.write_text(content)

        return file_path

    def get_artifact(self, case_id: str, filename: str) -> dict | str | None:
        """Get an artifact by filename."""
        artifacts_path = self._artifacts_path(case_id)
        safe_filename = Path(filename).name
        file_path = artifacts_path / safe_filename

        if not file_path.exists():
            return None

        content = file_path.read_text()

        # Try to parse as JSON
        if safe_filename.endswith(".json"):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass

        return content

    def list_artifacts(self, case_id: str) -> list[ArtifactInfo]:
        """List all artifacts for a case."""
        artifacts_path = self._artifacts_path(case_id)
        if not artifacts_path.exists():
            return []

        artifacts = []
        for file_path in artifacts_path.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                artifacts.append(
                    ArtifactInfo(
                        name=file_path.name,
                        path=str(file_path),
                        size=stat.st_size,
                        created_at=datetime.fromtimestamp(stat.st_ctime),
                        artifact_type=file_path.suffix.lstrip(".") or "unknown",
                    )
                )

        return sorted(artifacts, key=lambda a: a.name)

    def delete_case(self, case_id: str) -> bool:
        """Delete a case and all its contents."""
        import shutil

        case_path = self._case_path(case_id)
        if case_path.exists():
            shutil.rmtree(case_path)
            return True
        return False
