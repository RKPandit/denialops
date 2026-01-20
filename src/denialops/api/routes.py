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
    extract_eob_facts,
    extract_plan_rules,
    extract_text,
    generate_action_plan,
    generate_document_pack,
    generate_personalized_summary,
    predict_success,
    route_case,
    validate_grounding,
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


class SuccessPredictionResponse(BaseModel):
    """Success prediction details."""

    likelihood: str = Field(..., description="low, medium, or high")
    score: float = Field(..., description="0.0 to 1.0")
    factors_for: list[str] = Field(default_factory=list)
    factors_against: list[str] = Field(default_factory=list)


class GroundingValidationResponse(BaseModel):
    """Grounding validation details."""

    is_grounded: bool = Field(..., description="Whether content is grounded")
    hallucinated_codes: list[str] = Field(default_factory=list)
    hallucinated_dates: list[str] = Field(default_factory=list)
    confidence: float = Field(0.5)


class RunPipelineResponse(BaseModel):
    """Response from running the pipeline."""

    status: str = Field(..., description="Pipeline status")
    route: RouteType | None = Field(None, description="Selected route")
    confidence: float | None = Field(None, description="Routing confidence")
    success_prediction: SuccessPredictionResponse | None = Field(
        None, description="Predicted success likelihood"
    )
    grounding_validation: GroundingValidationResponse | None = Field(
        None, description="Grounding validation results"
    )
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

        # Stage 2c: Extract EOB facts if uploaded (Multi-document support - Phase 4)
        eob_facts = None
        eob_doc = next(
            (d for d in documents if d.get("doc_type") == "eob"),
            None,
        )
        if eob_doc:
            eob_path = storage.get_document_path(case_id, eob_doc["document_id"])
            eob_extracted = extract_text(eob_path)
            eob_facts = extract_eob_facts(
                case_id=case_id,
                text=eob_extracted,
                llm_api_key=settings.llm_api_key,
                llm_model=settings.llm_model,
                llm_provider=settings.llm_provider.value,
            )
            storage.store_artifact(
                case_id, "eob_facts.json", eob_facts.model_dump(mode="json")
            )

            # Enrich case facts with EOB information
            if eob_facts.denial_codes and not facts.denial_codes:
                facts.denial_codes = [
                    {"code": code, "description": "From EOB"}
                    for code in eob_facts.denial_codes
                ]
            if (
                eob_facts.appeal_deadline
                and facts.dates
                and not facts.dates.appeal_deadline
            ):
                facts.dates.appeal_deadline = eob_facts.appeal_deadline

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

        # Stage 6: Generate personalized summary (Phase 4)
        personalized = generate_personalized_summary(
            facts=facts,
            route=route_decision,
            plan_rules=plan_rules,
            llm_api_key=settings.llm_api_key,
            llm_model=settings.llm_model,
            llm_provider=settings.llm_provider.value,
        )
        storage.store_artifact(
            case_id,
            "personalized_summary.json",
            {
                "situation_summary": personalized.situation_summary,
                "recommendation": personalized.recommendation,
                "key_points": personalized.key_points,
                "urgency": personalized.urgency,
                "success_factors": personalized.success_factors,
                "is_llm_generated": personalized.is_llm_generated,
            },
        )

        # Stage 7: Predict success likelihood (Phase 4)
        success = predict_success(
            facts=facts,
            route=route_decision,
            plan_rules=plan_rules,
            llm_api_key=settings.llm_api_key,
            llm_model=settings.llm_model,
            llm_provider=settings.llm_provider.value,
        )
        storage.store_artifact(
            case_id,
            "success_prediction.json",
            {
                "likelihood": success.likelihood,
                "score": success.score,
                "factors_for": success.factors_for,
                "factors_against": success.factors_against,
                "reasoning": success.reasoning,
            },
        )

        # Stage 8: Validate grounding of generated content (Phase 4)
        # Validate the appeal letter if it was generated
        appeal_letter = documents.get("appeal_letter.md", "")
        grounding = validate_grounding(
            content=appeal_letter,
            facts=facts,
            plan_rules=plan_rules,
            llm_api_key=settings.llm_api_key,
            llm_model=settings.llm_model,
            llm_provider=settings.llm_provider.value,
        )
        storage.store_artifact(
            case_id,
            "grounding_validation.json",
            {
                "is_grounded": grounding.is_grounded,
                "ungrounded_claims": grounding.ungrounded_claims,
                "hallucinated_codes": grounding.hallucinated_codes,
                "hallucinated_dates": grounding.hallucinated_dates,
                "hallucinated_amounts": grounding.hallucinated_amounts,
                "confidence": grounding.confidence,
            },
        )

        # List generated artifacts
        artifacts = storage.list_artifacts(case_id)
        artifact_names = [a.name for a in artifacts]

        return RunPipelineResponse(
            status="completed",
            route=route_decision.route,
            confidence=route_decision.confidence,
            success_prediction=SuccessPredictionResponse(
                likelihood=success.likelihood,
                score=success.score,
                factors_for=success.factors_for,
                factors_against=success.factors_against,
            ),
            grounding_validation=GroundingValidationResponse(
                is_grounded=grounding.is_grounded,
                hallucinated_codes=grounding.hallucinated_codes,
                hallucinated_dates=grounding.hallucinated_dates,
                confidence=grounding.confidence,
            ),
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
