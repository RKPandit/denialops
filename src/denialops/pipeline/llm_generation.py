"""LLM-powered content generation with grounding validation.

This module provides:
1. Personalized summary generation using LLM
2. Grounding validation to prevent hallucination
3. Success likelihood estimation
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

from denialops.llm import create_llm_client
from denialops.models.case import CaseFacts
from denialops.models.plan_rules import PlanRules
from denialops.models.route import RouteDecision, RouteType

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PersonalizedSummary:
    """LLM-generated personalized summary."""

    situation_summary: str
    recommendation: str
    key_points: list[str]
    urgency: str  # "low", "medium", "high"
    success_factors: list[str]
    is_llm_generated: bool = True


@dataclass
class GroundingResult:
    """Result of grounding validation."""

    is_grounded: bool
    ungrounded_claims: list[dict[str, str]]
    hallucinated_codes: list[str]
    hallucinated_dates: list[str]
    hallucinated_amounts: list[str]
    confidence: float


@dataclass
class SuccessPrediction:
    """Prediction of appeal/action success likelihood."""

    likelihood: str  # "low", "medium", "high"
    score: float  # 0.0 to 1.0
    factors_for: list[str]
    factors_against: list[str]
    reasoning: str


# =============================================================================
# Prompts
# =============================================================================

PERSONALIZED_SUMMARY_SYSTEM = """You are an expert patient advocate helping people understand their insurance claim denial.

Your goal is to:
1. Explain their situation in simple, clear language
2. Provide empathetic but realistic guidance
3. Focus on what they can actually do to resolve this
4. Be specific to THEIR situation, not generic advice

Always respond with valid JSON. Ground your response ONLY in the provided facts."""

PERSONALIZED_SUMMARY_USER = """Generate a personalized summary for this patient's insurance denial situation.

<case_facts>
{facts_json}
</case_facts>

<route_decision>
Route: {route}
Confidence: {confidence}
Reasoning: {reasoning}
</route_decision>

<plan_rules>
{plan_rules_json}
</plan_rules>

Generate a JSON response:
{{
  "situation_summary": "A 2-3 sentence personalized summary of THEIR specific situation. Use their actual service, provider, and payer names. Be empathetic but factual.",
  "recommendation": "A clear 1-2 sentence recommendation for their specific next step. Be actionable and specific.",
  "key_points": ["3-5 key points specific to their case - include actual deadlines, amounts, and contact info if available"],
  "urgency": "low|medium|high - based on deadlines and situation severity",
  "success_factors": ["2-4 factors that will determine success in their specific case"]
}}

IMPORTANT RULES:
- Only reference information that appears in the case_facts
- If deadline is approaching (<30 days), set urgency to "high"
- If amounts are significant (>$1000), mention this in key_points
- Use the patient's actual payer name, service description, and dates
- If plan_rules are provided, reference specific policy citations

Return ONLY valid JSON, no other text."""


GROUNDING_VALIDATION_SYSTEM = """You are a fact-checker ensuring AI-generated content is grounded in source data.

Your job is to identify:
1. Claims that don't appear in the source facts
2. Codes, dates, or amounts that were invented
3. Assumptions presented as facts

Be strict - if something isn't explicitly in the source, flag it."""

GROUNDING_VALIDATION_USER = """Check if this generated content is properly grounded in the source facts.

<source_facts>
{facts_json}
</source_facts>

<plan_rules>
{plan_rules_json}
</plan_rules>

<generated_content>
{content}
</generated_content>

Return a JSON response:
{{
  "is_grounded": true/false,
  "ungrounded_claims": [
    {{"claim": "the ungrounded statement", "issue": "why it's not grounded"}}
  ],
  "hallucinated_codes": ["list of codes not in source - CPT, ICD, CARC codes"],
  "hallucinated_dates": ["list of dates not in source"],
  "hallucinated_amounts": ["list of dollar amounts not in source"],
  "confidence": 0.0-1.0
}}

Return ONLY valid JSON."""


SUCCESS_PREDICTION_SYSTEM = """You are an expert at predicting insurance appeal outcomes based on case factors.

Consider:
- Type of denial (PA, medical necessity, coding)
- Available documentation
- Deadline compliance
- Plan type and typical approval rates
- Strength of medical justification

Be realistic and evidence-based."""

SUCCESS_PREDICTION_USER = """Predict the likelihood of success for this insurance appeal/action.

<case_facts>
{facts_json}
</case_facts>

<route>
{route}
</route>

<plan_rules>
{plan_rules_json}
</plan_rules>

Return a JSON response:
{{
  "likelihood": "low|medium|high",
  "score": 0.0-1.0,
  "factors_for": ["2-4 factors that increase chances of success"],
  "factors_against": ["2-4 factors that decrease chances"],
  "reasoning": "1-2 sentence explanation of the prediction"
}}

Scoring guide:
- low: <30% chance (0.0-0.3)
- medium: 30-70% chance (0.3-0.7)
- high: >70% chance (0.7-1.0)

Return ONLY valid JSON."""


# =============================================================================
# Generation Functions
# =============================================================================


def generate_personalized_summary(
    facts: CaseFacts,
    route: RouteDecision,
    plan_rules: PlanRules | None = None,
    llm_api_key: str = "",
    llm_model: str = "",
    llm_provider: str = "openai",
) -> PersonalizedSummary:
    """
    Generate a personalized summary using LLM.

    Args:
        facts: Extracted case facts
        route: Routing decision
        plan_rules: Optional plan rules from SBC
        llm_api_key: API key for LLM
        llm_model: Model to use
        llm_provider: LLM provider

    Returns:
        PersonalizedSummary with LLM-generated content
    """
    # Fall back to template if no API key
    if not llm_api_key:
        return _generate_template_summary(facts, route, plan_rules)

    try:
        logger.info(f"Generating personalized summary using {llm_provider}/{llm_model}")

        client = create_llm_client(
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model if llm_model else None,
        )

        # Build prompt
        facts_json = facts.model_dump_json(indent=2)
        plan_rules_json = (
            plan_rules.model_dump_json(indent=2) if plan_rules else "null"
        )

        prompt = PERSONALIZED_SUMMARY_USER.format(
            facts_json=facts_json,
            route=route.route.value,
            confidence=route.confidence,
            reasoning=route.reasoning,
            plan_rules_json=plan_rules_json,
        )

        response = client.complete(
            prompt=prompt,
            system=PERSONALIZED_SUMMARY_SYSTEM,
            max_tokens=1000,
            temperature=0.3,  # Slight creativity for personalization
        )

        # Parse response
        data = _parse_json_response(response)

        return PersonalizedSummary(
            situation_summary=data.get("situation_summary", ""),
            recommendation=data.get("recommendation", ""),
            key_points=data.get("key_points", []),
            urgency=data.get("urgency", "medium"),
            success_factors=data.get("success_factors", []),
            is_llm_generated=True,
        )

    except Exception as e:
        logger.warning(f"LLM summary generation failed: {e}, using template")
        return _generate_template_summary(facts, route, plan_rules)


def validate_grounding(
    content: str,
    facts: CaseFacts,
    plan_rules: PlanRules | None = None,
    llm_api_key: str = "",
    llm_model: str = "",
    llm_provider: str = "openai",
) -> GroundingResult:
    """
    Validate that generated content is grounded in source facts.

    Args:
        content: Generated content to validate
        facts: Source case facts
        plan_rules: Optional plan rules
        llm_api_key: API key for LLM
        llm_model: Model to use
        llm_provider: LLM provider

    Returns:
        GroundingResult with validation details
    """
    if not llm_api_key:
        # Without LLM, do basic heuristic validation
        return _validate_grounding_heuristic(content, facts, plan_rules)

    try:
        logger.info("Validating grounding using LLM")

        client = create_llm_client(
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model if llm_model else None,
        )

        facts_json = facts.model_dump_json(indent=2)
        plan_rules_json = (
            plan_rules.model_dump_json(indent=2) if plan_rules else "null"
        )

        prompt = GROUNDING_VALIDATION_USER.format(
            facts_json=facts_json,
            plan_rules_json=plan_rules_json,
            content=content,
        )

        response = client.complete(
            prompt=prompt,
            system=GROUNDING_VALIDATION_SYSTEM,
            max_tokens=1000,
            temperature=0.0,  # Deterministic validation
        )

        data = _parse_json_response(response)

        return GroundingResult(
            is_grounded=data.get("is_grounded", False),
            ungrounded_claims=data.get("ungrounded_claims", []),
            hallucinated_codes=data.get("hallucinated_codes", []),
            hallucinated_dates=data.get("hallucinated_dates", []),
            hallucinated_amounts=data.get("hallucinated_amounts", []),
            confidence=data.get("confidence", 0.5),
        )

    except Exception as e:
        logger.warning(f"LLM grounding validation failed: {e}, using heuristics")
        return _validate_grounding_heuristic(content, facts, plan_rules)


def predict_success(
    facts: CaseFacts,
    route: RouteDecision,
    plan_rules: PlanRules | None = None,
    llm_api_key: str = "",
    llm_model: str = "",
    llm_provider: str = "openai",
) -> SuccessPrediction:
    """
    Predict the likelihood of success for the recommended action.

    Args:
        facts: Case facts
        route: Routing decision
        plan_rules: Optional plan rules
        llm_api_key: API key for LLM
        llm_model: Model to use
        llm_provider: LLM provider

    Returns:
        SuccessPrediction with likelihood and factors
    """
    if not llm_api_key:
        return _predict_success_heuristic(facts, route, plan_rules)

    try:
        logger.info("Predicting success using LLM")

        client = create_llm_client(
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model if llm_model else None,
        )

        facts_json = facts.model_dump_json(indent=2)
        plan_rules_json = (
            plan_rules.model_dump_json(indent=2) if plan_rules else "null"
        )

        prompt = SUCCESS_PREDICTION_USER.format(
            facts_json=facts_json,
            route=route.route.value,
            plan_rules_json=plan_rules_json,
        )

        response = client.complete(
            prompt=prompt,
            system=SUCCESS_PREDICTION_SYSTEM,
            max_tokens=800,
            temperature=0.2,
        )

        data = _parse_json_response(response)

        return SuccessPrediction(
            likelihood=data.get("likelihood", "medium"),
            score=data.get("score", 0.5),
            factors_for=data.get("factors_for", []),
            factors_against=data.get("factors_against", []),
            reasoning=data.get("reasoning", ""),
        )

    except Exception as e:
        logger.warning(f"LLM success prediction failed: {e}, using heuristics")
        return _predict_success_heuristic(facts, route, plan_rules)


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_json_response(response: str) -> dict[str, Any]:
    """Parse JSON from LLM response."""
    # Handle markdown code blocks
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        json_str = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        json_str = response[start:end].strip()
    else:
        json_str = response.strip()

    return json.loads(json_str)


def _generate_template_summary(
    facts: CaseFacts,
    route: RouteDecision,
    plan_rules: PlanRules | None = None,
) -> PersonalizedSummary:
    """Generate summary using templates (fallback)."""
    # Build situation based on route
    if route.route == RouteType.PRIOR_AUTH_NEEDED:
        situation = (
            f"Your claim was denied because prior authorization was not obtained. "
            f"The denial reason states: {facts.denial_reason}"
        )
        recommendation = (
            "Contact your healthcare provider to submit a prior authorization request, "
            "then request reconsideration of this claim."
        )
        key_points = [
            "Prior authorization was required but not obtained",
            "Your provider can submit a retroactive PA request",
            "Keep all documentation for appeal if PA is approved",
        ]
    elif route.route == RouteType.CLAIM_CORRECTION_RESUBMIT:
        situation = (
            f"Your claim was denied due to a coding or billing issue. "
            f"The denial reason states: {facts.denial_reason}"
        )
        recommendation = (
            "Contact your provider's billing department to correct and resubmit the claim."
        )
        key_points = [
            "This appears to be a billing/coding error",
            "Your provider's billing department can fix this",
            "Corrected claims have high success rates",
        ]
    else:  # MEDICAL_NECESSITY_APPEAL
        situation = (
            f"Your claim was denied based on medical necessity. "
            f"The denial reason states: {facts.denial_reason}"
        )
        recommendation = (
            "File a formal appeal with supporting clinical documentation "
            "from your healthcare provider."
        )
        key_points = [
            "Medical necessity appeals require clinical documentation",
            "Get a letter of medical necessity from your provider",
            "Include all relevant medical records",
        ]

    # Add deadline info if available
    if facts.dates and facts.dates.appeal_deadline:
        days_left = (facts.dates.appeal_deadline - facts.dates.date_of_denial).days
        key_points.append(f"Appeal deadline: {facts.dates.appeal_deadline} ({days_left} days)")
        urgency = "high" if days_left < 30 else "medium" if days_left < 60 else "low"
    else:
        urgency = "medium"

    # Add plan-specific info if available
    if plan_rules and plan_rules.appeal_rights:
        ar = plan_rules.appeal_rights
        if ar.internal_appeal_deadline_days:
            key_points.append(
                f"Your plan allows {ar.internal_appeal_deadline_days} days to appeal"
            )

    success_factors = [
        "Complete documentation submitted",
        "Appeal filed within deadline",
        "Clear medical justification provided",
    ]

    return PersonalizedSummary(
        situation_summary=situation,
        recommendation=recommendation,
        key_points=key_points,
        urgency=urgency,
        success_factors=success_factors,
        is_llm_generated=False,
    )


def _validate_grounding_heuristic(
    content: str,
    facts: CaseFacts,
    plan_rules: PlanRules | None = None,
) -> GroundingResult:
    """Validate grounding using heuristics (fallback)."""
    import re

    hallucinated_codes = []
    hallucinated_dates = []
    hallucinated_amounts = []

    # Extract codes from content
    content_codes = set(re.findall(r"\b(\d{5})\b", content))  # CPT codes
    content_codes.update(re.findall(r"\b(CO-\d+|PR-\d+|OA-\d+)\b", content))  # CARC codes

    # Get known codes from facts
    known_codes = set()
    for code in facts.denial_codes:
        known_codes.add(code.code)
    if facts.service and facts.service.cpt_codes:
        known_codes.update(facts.service.cpt_codes)

    # Find hallucinated codes
    for code in content_codes:
        if code not in known_codes:
            hallucinated_codes.append(code)

    # Extract dates from content (YYYY-MM-DD or MM/DD/YYYY)
    content_dates = set(re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", content))
    content_dates.update(re.findall(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", content))

    # Get known dates
    known_dates = set()
    if facts.dates:
        if facts.dates.date_of_service:
            known_dates.add(str(facts.dates.date_of_service))
        if facts.dates.date_of_denial:
            known_dates.add(str(facts.dates.date_of_denial))
        if facts.dates.appeal_deadline:
            known_dates.add(str(facts.dates.appeal_deadline))

    for date_str in content_dates:
        if date_str not in known_dates:
            hallucinated_dates.append(date_str)

    # Extract amounts from content
    content_amounts = set(re.findall(r"\$[\d,]+\.?\d*", content))

    # Get known amounts
    known_amounts = set()
    if facts.amounts:
        if facts.amounts.billed_amount:
            known_amounts.add(f"${facts.amounts.billed_amount:,.2f}")
        if facts.amounts.patient_responsibility:
            known_amounts.add(f"${facts.amounts.patient_responsibility:,.2f}")

    for amount in content_amounts:
        # Normalize amount format
        normalized = amount.replace(",", "")
        if amount not in known_amounts and normalized not in known_amounts:
            hallucinated_amounts.append(amount)

    is_grounded = (
        len(hallucinated_codes) == 0
        and len(hallucinated_dates) == 0
        and len(hallucinated_amounts) == 0
    )

    return GroundingResult(
        is_grounded=is_grounded,
        ungrounded_claims=[],  # Can't detect semantic claims with heuristics
        hallucinated_codes=hallucinated_codes,
        hallucinated_dates=hallucinated_dates,
        hallucinated_amounts=hallucinated_amounts,
        confidence=0.6 if is_grounded else 0.3,
    )


def _predict_success_heuristic(
    facts: CaseFacts,
    route: RouteDecision,
    plan_rules: PlanRules | None = None,
) -> SuccessPrediction:
    """Predict success using heuristics (fallback)."""
    factors_for = []
    factors_against = []
    score = 0.5  # Start at medium

    # Route-based factors
    if route.route == RouteType.CLAIM_CORRECTION_RESUBMIT:
        factors_for.append("Coding/billing issues have high success rates when corrected")
        score += 0.2
    elif route.route == RouteType.PRIOR_AUTH_NEEDED:
        factors_for.append("Retroactive PA approval is often possible")
        score += 0.1
    else:  # Medical necessity
        factors_against.append("Medical necessity appeals require strong clinical evidence")
        score -= 0.1

    # High confidence routing is a good sign
    if route.confidence > 0.7:
        factors_for.append("Clear denial reason makes response strategy clearer")
        score += 0.1

    # Check deadline compliance
    if facts.dates and facts.dates.appeal_deadline:
        from datetime import date

        days_left = (facts.dates.appeal_deadline - date.today()).days
        if days_left > 30:
            factors_for.append("Sufficient time remaining to file appeal")
            score += 0.1
        elif days_left < 7:
            factors_against.append("Very limited time to prepare appeal")
            score -= 0.2
        else:
            factors_against.append("Limited time remaining to file appeal")
            score -= 0.1

    # Plan rules availability
    if plan_rules:
        factors_for.append("Plan policy details available for reference")
        score += 0.1

    # Amount involved
    if (
        facts.amounts
        and facts.amounts.patient_responsibility
        and facts.amounts.patient_responsibility > 5000
    ):
        factors_for.append("Significant amount justifies thorough appeal effort")

    # Clamp score
    score = max(0.1, min(0.9, score))

    # Determine likelihood
    if score >= 0.7:
        likelihood = "high"
    elif score >= 0.4:
        likelihood = "medium"
    else:
        likelihood = "low"

    reasoning = (
        f"Based on the {route.route.value} route with {route.confidence:.0%} confidence, "
        f"the estimated success likelihood is {likelihood}."
    )

    return SuccessPrediction(
        likelihood=likelihood,
        score=score,
        factors_for=factors_for,
        factors_against=factors_against,
        reasoning=reasoning,
    )
