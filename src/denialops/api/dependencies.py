"""FastAPI dependencies for DenialOps."""

from typing import Annotated

from fastapi import Depends, HTTPException, Path, status

from denialops.config import Settings, get_settings
from denialops.utils.storage import CaseStorage


def get_storage(settings: Annotated[Settings, Depends(get_settings)]) -> CaseStorage:
    """Get storage instance."""
    return CaseStorage(settings.artifacts_path)


def validate_case_exists(
    case_id: Annotated[str, Path(description="Case ID")],
    storage: Annotated[CaseStorage, Depends(get_storage)],
) -> str:
    """Validate that a case exists, return case_id if valid."""
    if not storage.case_exists(case_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case {case_id} not found",
        )
    return case_id


# Type aliases for dependency injection
StorageDep = Annotated[CaseStorage, Depends(get_storage)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
CaseIdDep = Annotated[str, Depends(validate_case_exists)]
