"""API routes for DenialOps."""

import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from denialops.api.dependencies import CaseIdDep, SettingsDep, StorageDep
from denialops.models.case import CaseMode, UserContext
from denialops.models.documents import ArtifactInfo, DocumentType
from denialops.models.route import RouteType
from denialops.pipeline import (
    extract_case_facts,
    extract_plan_rules,
    extract_text,
    generate_action_plan,
    generate_document_pack,
    route_case,
)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class CreateCaseRequest(BaseModel):
    """Request to create a new case."""

    mode: CaseMode = Field(CaseMode.FAST, description="Processing mode")
    user_context: UserContext | None = Field(None, description="Optional user context")


class CreateCaseResponse(BaseModel):
    """Response from creating a case."""

    case_id: str = Field(..., description="Unique case identifier")
    mode: CaseMode = Field(..., description="Processing mode")
    created_at: datetime = Field(..., description="Creation timestamp")


class UploadDocumentResponse(BaseModel):
    """Response from uploading a document."""

    document_id: str = Field(..., description="Unique document identifier")
    case_id: str = Field(..., description="Case this document belongs to")
    doc_type: DocumentType = Field(..., description="Document type")
    stored_path: str = Field(..., description="Storage path")


class RunPipelineResponse(BaseModel):
    """Response from running the pipeline."""

    status: str = Field(..., description="Pipeline status")
    route: RouteType | None = Field(None, description="Selected route")
    confidence: float | None = Field(None, description="Routing confidence")
    artifacts: list[str] = Field(default_factory=list, description="Generated artifacts")
    error: str | None = Field(None, description="Error message if failed")


class ListArtifactsResponse(BaseModel):
    """Response listing case artifacts."""

    case_id: str
    artifacts: list[ArtifactInfo]


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/cases", response_model=CreateCaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(
    request: CreateCaseRequest,
    storage: StorageDep,
) -> CreateCaseResponse:
    """Create a new case for processing."""
    case_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    # Create case directory
    storage.create_case(case_id)

    # Store case metadata
    metadata = {
        "case_id": case_id,
        "mode": request.mode.value,
        "created_at": created_at.isoformat(),
        "user_context": request.user_context.model_dump() if request.user_context else None,
    }
    storage.store_artifact(case_id, "metadata.json", metadata)

    return CreateCaseResponse(
        case_id=case_id,
        mode=request.mode,
        created_at=created_at,
    )


@router.post("/cases/{case_id}/documents", response_model=UploadDocumentResponse)
async def upload_document(
    case_id: CaseIdDep,
    storage: StorageDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="Document file to upload")],
    doc_type: Annotated[DocumentType, Form(description="Type of document")],
) -> UploadDocumentResponse:
    """Upload a document to a case."""
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    # Check file extension
    allowed_extensions = {".pdf", ".txt"}
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension. Allowed: {allowed_extensions}",
        )

    # Check file size
    content = await file.read()
    if len(content) > settings.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_upload_size} bytes",
        )

    # Store document
    document_id = str(uuid.uuid4())
    stored_path = storage.store_document(
        case_id=case_id,
        document_id=document_id,
        filename=file.filename,
        content=content,
        doc_type=doc_type.value,
    )

    return UploadDocumentResponse(
        document_id=document_id,
        case_id=case_id,
        doc_type=doc_type,
        stored_path=stored_path,
    )


@router.post("/cases/{case_id}/run", response_model=RunPipelineResponse)
async def run_pipeline(
    case_id: CaseIdDep,
    storage: StorageDep,
    settings: SettingsDep,
) -> RunPipelineResponse:
    """Run the processing pipeline on a case."""
    try:
        # Load case metadata
        metadata = storage.get_artifact(case_id, "metadata.json")
        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Case metadata not found",
            )

        _mode = CaseMode(metadata.get("mode", "fast"))  # Used in Phase 2 for generation
        user_context = metadata.get("user_context")

        # Find denial letter
        documents = storage.list_documents(case_id)
        denial_doc = next(
            (d for d in documents if d.get("doc_type") == "denial_letter"),
            None,
        )

        if not denial_doc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No denial letter uploaded. Upload a denial letter first.",
            )

        # Stage 1: Extract text
        doc_path = storage.get_document_path(case_id, denial_doc["document_id"])
        extracted = extract_text(doc_path)
        storage.store_artifact(case_id, "extracted_text.txt", extracted.full_text)

        # Stage 2: Extract case facts
        facts = extract_case_facts(
            case_id=case_id,
            text=extracted,
            user_context=user_context,
            llm_api_key=settings.llm_api_key,
            llm_model=settings.llm_model,
            llm_provider=settings.llm_provider.value,
        )
        storage.store_artifact(case_id, "case_facts.json", facts.model_dump(mode="json"))

        # Stage 2b: Extract plan rules from SBC/EOC if uploaded (Verified mode)
        plan_rules = None
        sbc_doc = next(
            (d for d in documents if d.get("doc_type") in ("sbc", "eoc")),
            None,
        )
        if sbc_doc:
            sbc_path = storage.get_document_path(case_id, sbc_doc["document_id"])
            sbc_extracted = extract_text(sbc_path)
            plan_rules = extract_plan_rules(
                case_id=case_id,
                text=sbc_extracted,
                llm_api_key=settings.llm_api_key,
                llm_model=settings.llm_model,
                llm_provider=settings.llm_provider.value,
            )
            storage.store_artifact(
                case_id, "plan_rules.json", plan_rules.model_dump(mode="json")
            )

        # Stage 3: Route case
        route_decision = route_case(facts)
        storage.store_artifact(case_id, "route.json", route_decision.model_dump(mode="json"))

        # Stage 4: Generate action plan
        action_plan = generate_action_plan(
            facts=facts,
            route=route_decision,
            mode=_mode.value,
            plan_rules=plan_rules,
        )
        storage.store_artifact(case_id, "action_plan.json", action_plan.model_dump(mode="json"))

        # Stage 5: Generate document pack
        documents = generate_document_pack(
            facts=facts, plan=action_plan, plan_rules=plan_rules
        )
        for filename, content in documents.items():
            storage.store_artifact(case_id, filename, content)

        # List generated artifacts
        artifacts = storage.list_artifacts(case_id)
        artifact_names = [a.name for a in artifacts]

        return RunPipelineResponse(
            status="completed",
            route=route_decision.route,
            confidence=route_decision.confidence,
            artifacts=artifact_names,
        )

    except HTTPException:
        raise
    except Exception as e:
        return RunPipelineResponse(
            status="failed",
            error=str(e),
            artifacts=[],
        )


@router.get("/cases/{case_id}/artifacts", response_model=ListArtifactsResponse)
async def list_artifacts(
    case_id: CaseIdDep,
    storage: StorageDep,
) -> ListArtifactsResponse:
    """List all artifacts for a case."""
    artifacts = storage.list_artifacts(case_id)
    return ListArtifactsResponse(case_id=case_id, artifacts=artifacts)


@router.get("/cases/{case_id}/artifacts/{filename}")
async def get_artifact(
    case_id: CaseIdDep,
    filename: str,
    storage: StorageDep,
) -> dict | str:
    """Get a specific artifact."""
    artifact = storage.get_artifact(case_id, filename)
    if artifact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact {filename} not found",
        )
    return artifact
