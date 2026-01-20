"""Background pipeline task execution."""

import logging
from dataclasses import dataclass
from typing import Any

from denialops.config import Settings
from denialops.models.case import CaseMode
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
from denialops.tasks.manager import get_task_manager
from denialops.utils.storage import CaseStorage

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """Definition of a pipeline stage."""

    name: str
    weight: float  # Relative weight for progress calculation
    description: str


# Define pipeline stages with their relative weights
PIPELINE_STAGES = [
    PipelineStage("extract_text", 0.1, "Extracting text from documents"),
    PipelineStage("extract_facts", 0.15, "Extracting case facts"),
    PipelineStage("extract_plan_rules", 0.1, "Extracting plan rules"),
    PipelineStage("extract_eob", 0.1, "Extracting EOB information"),
    PipelineStage("route_case", 0.1, "Determining recommended action"),
    PipelineStage("generate_plan", 0.1, "Generating action plan"),
    PipelineStage("generate_documents", 0.15, "Generating documents"),
    PipelineStage("generate_summary", 0.1, "Generating personalized summary"),
    PipelineStage("validate", 0.1, "Validating outputs"),
]


class PipelineTask:
    """
    Background task for running the full processing pipeline.

    This runs the pipeline asynchronously and provides progress updates.
    """

    def __init__(
        self,
        case_id: str,
        storage: CaseStorage,
        settings: Settings,
    ) -> None:
        self.case_id = case_id
        self.storage = storage
        self.settings = settings
        self._task_manager = get_task_manager()
        self._task_id: str | None = None

    def start(self) -> str:
        """
        Start the pipeline execution in the background.

        Returns:
            The task ID for tracking progress
        """
        self._task_id = f"pipeline-{self.case_id}"

        # Submit the coroutine to the task manager
        self._task_manager.submit(
            self._run_pipeline(),
            task_id=self._task_id,
        )

        return self._task_id

    def _update_progress(self, stage_index: int, message: str) -> None:
        """Update progress based on completed stages."""
        if not self._task_id:
            return

        # Calculate progress from completed stages
        progress = sum(s.weight for s in PIPELINE_STAGES[:stage_index])
        self._task_manager.update_progress(self._task_id, progress, message)

    async def _run_pipeline(self) -> dict[str, Any]:
        """Run the full pipeline and return results."""
        result: dict[str, Any] = {
            "case_id": self.case_id,
            "stages_completed": [],
            "artifacts": [],
        }

        try:
            # Load case metadata
            metadata = self.storage.get_artifact(self.case_id, "metadata.json")
            if not metadata:
                raise ValueError("Case metadata not found")

            mode = CaseMode(metadata.get("mode", "fast"))
            user_context = metadata.get("user_context")

            # Find documents
            documents = self.storage.list_documents(self.case_id)
            denial_doc = next(
                (d for d in documents if d.get("doc_type") == "denial_letter"),
                None,
            )

            if not denial_doc:
                raise ValueError("No denial letter uploaded")

            # Stage 1: Extract text
            self._update_progress(0, PIPELINE_STAGES[0].description)
            doc_path = self.storage.get_document_path(
                self.case_id, denial_doc["document_id"]
            )
            extracted = extract_text(doc_path)
            self.storage.store_artifact(
                self.case_id, "extracted_text.txt", extracted.full_text
            )
            result["stages_completed"].append("extract_text")

            # Stage 2: Extract case facts
            self._update_progress(1, PIPELINE_STAGES[1].description)
            facts = extract_case_facts(
                case_id=self.case_id,
                text=extracted,
                user_context=user_context,
                llm_api_key=self.settings.llm_api_key,
                llm_model=self.settings.llm_model,
                llm_provider=self.settings.llm_provider.value,
            )
            self.storage.store_artifact(
                self.case_id, "case_facts.json", facts.model_dump(mode="json")
            )
            result["stages_completed"].append("extract_facts")

            # Stage 3: Extract plan rules (if SBC uploaded)
            self._update_progress(2, PIPELINE_STAGES[2].description)
            plan_rules = None
            sbc_doc = next(
                (d for d in documents if d.get("doc_type") in ("sbc", "eoc")),
                None,
            )
            if sbc_doc:
                sbc_path = self.storage.get_document_path(
                    self.case_id, sbc_doc["document_id"]
                )
                sbc_extracted = extract_text(sbc_path)
                plan_rules = extract_plan_rules(
                    case_id=self.case_id,
                    text=sbc_extracted,
                    llm_api_key=self.settings.llm_api_key,
                    llm_model=self.settings.llm_model,
                    llm_provider=self.settings.llm_provider.value,
                )
                self.storage.store_artifact(
                    self.case_id,
                    "plan_rules.json",
                    plan_rules.model_dump(mode="json"),
                )
            result["stages_completed"].append("extract_plan_rules")

            # Stage 4: Extract EOB (if uploaded)
            self._update_progress(3, PIPELINE_STAGES[3].description)
            eob_facts = None
            eob_doc = next(
                (d for d in documents if d.get("doc_type") == "eob"),
                None,
            )
            if eob_doc:
                eob_path = self.storage.get_document_path(
                    self.case_id, eob_doc["document_id"]
                )
                eob_extracted = extract_text(eob_path)
                eob_facts = extract_eob_facts(
                    case_id=self.case_id,
                    text=eob_extracted,
                    llm_api_key=self.settings.llm_api_key,
                    llm_model=self.settings.llm_model,
                    llm_provider=self.settings.llm_provider.value,
                )
                self.storage.store_artifact(
                    self.case_id,
                    "eob_facts.json",
                    eob_facts.model_dump(mode="json"),
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

            result["stages_completed"].append("extract_eob")

            # Stage 5: Route case
            self._update_progress(4, PIPELINE_STAGES[4].description)
            route_decision = route_case(facts)
            self.storage.store_artifact(
                self.case_id, "route.json", route_decision.model_dump(mode="json")
            )
            result["stages_completed"].append("route_case")
            result["route"] = route_decision.route.value
            result["confidence"] = route_decision.confidence

            # Stage 6: Generate action plan
            self._update_progress(5, PIPELINE_STAGES[5].description)
            action_plan = generate_action_plan(
                facts=facts,
                route=route_decision,
                mode=mode.value,
                plan_rules=plan_rules,
            )
            self.storage.store_artifact(
                self.case_id, "action_plan.json", action_plan.model_dump(mode="json")
            )
            result["stages_completed"].append("generate_plan")

            # Stage 7: Generate documents
            self._update_progress(6, PIPELINE_STAGES[6].description)
            generated_docs = generate_document_pack(
                facts=facts, plan=action_plan, plan_rules=plan_rules
            )
            for filename, content in generated_docs.items():
                self.storage.store_artifact(self.case_id, filename, content)
            result["stages_completed"].append("generate_documents")

            # Stage 8: Generate personalized summary
            self._update_progress(7, PIPELINE_STAGES[7].description)
            personalized = generate_personalized_summary(
                facts=facts,
                route=route_decision,
                plan_rules=plan_rules,
                llm_api_key=self.settings.llm_api_key,
                llm_model=self.settings.llm_model,
                llm_provider=self.settings.llm_provider.value,
            )
            self.storage.store_artifact(
                self.case_id,
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

            success = predict_success(
                facts=facts,
                route=route_decision,
                plan_rules=plan_rules,
                llm_api_key=self.settings.llm_api_key,
                llm_model=self.settings.llm_model,
                llm_provider=self.settings.llm_provider.value,
            )
            self.storage.store_artifact(
                self.case_id,
                "success_prediction.json",
                {
                    "likelihood": success.likelihood,
                    "score": success.score,
                    "factors_for": success.factors_for,
                    "factors_against": success.factors_against,
                    "reasoning": success.reasoning,
                },
            )
            result["stages_completed"].append("generate_summary")
            result["success_likelihood"] = success.likelihood

            # Stage 9: Validate outputs
            self._update_progress(8, PIPELINE_STAGES[8].description)
            appeal_letter = generated_docs.get("appeal_letter.md", "")
            grounding = validate_grounding(
                content=appeal_letter,
                facts=facts,
                plan_rules=plan_rules,
                llm_api_key=self.settings.llm_api_key,
                llm_model=self.settings.llm_model,
                llm_provider=self.settings.llm_provider.value,
            )
            self.storage.store_artifact(
                self.case_id,
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
            result["stages_completed"].append("validate")
            result["is_grounded"] = grounding.is_grounded

            # Get final artifact list
            artifacts = self.storage.list_artifacts(self.case_id)
            result["artifacts"] = [a.name for a in artifacts]
            result["status"] = "completed"

            return result

        except Exception as e:
            logger.error(f"Pipeline failed for case {self.case_id}: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            raise
