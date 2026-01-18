"""Case routing logic."""

from datetime import datetime, timezone

from denialops.models.case import CaseFacts
from denialops.models.route import (
    AlternativeRoute,
    RouteDecision,
    RouteSignal,
    RouteType,
    Urgency,
)


def route_case(facts: CaseFacts) -> RouteDecision:
    """
    Determine the best action route for a case.

    Uses rule-based routing with signal detection.

    Args:
        facts: Extracted case facts

    Returns:
        RouteDecision with selected route and reasoning
    """
    signals: list[RouteSignal] = []

    # Detect signals for each route
    pa_score, pa_signals = _detect_prior_auth_signals(facts)
    signals.extend(pa_signals)

    coding_score, coding_signals = _detect_coding_signals(facts)
    signals.extend(coding_signals)

    med_nec_score, med_nec_signals = _detect_medical_necessity_signals(facts)
    signals.extend(med_nec_signals)

    # Determine route based on scores
    scores = {
        RouteType.PRIOR_AUTH_NEEDED: pa_score,
        RouteType.CLAIM_CORRECTION_RESUBMIT: coding_score,
        RouteType.MEDICAL_NECESSITY_APPEAL: med_nec_score,
    }

    # Sort by score
    sorted_routes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected_route = sorted_routes[0][0]
    selected_score = sorted_routes[0][1]

    # Calculate confidence
    confidence = _calculate_confidence(selected_score, scores)

    # Build reasoning
    reasoning = _build_reasoning(selected_route, signals, facts)

    # Build alternative routes
    alternatives = [
        AlternativeRoute(
            route=route,
            confidence=round(score / 10, 2),  # Normalize to 0-1
            reason_not_selected=f"Lower signal score ({score:.1f} vs {selected_score:.1f})",
        )
        for route, score in sorted_routes[1:]
        if score > 0
    ]

    # Determine urgency
    urgency, urgency_reason = _determine_urgency(facts)

    # Check if Verified mode would help
    requires_verified = selected_route == RouteType.MEDICAL_NECESSITY_APPEAL

    return RouteDecision(
        case_id=facts.case_id,
        timestamp=datetime.now(timezone.utc),
        route=selected_route,
        confidence=confidence,
        reasoning=reasoning,
        signals=signals,
        alternative_routes=alternatives,
        requires_verified_mode=requires_verified,
        urgency=urgency,
        urgency_reason=urgency_reason,
    )


def _detect_prior_auth_signals(facts: CaseFacts) -> tuple[float, list[RouteSignal]]:
    """Detect signals indicating prior authorization issue."""
    score = 0.0
    signals: list[RouteSignal] = []
    denial_text = facts.denial_reason.lower()

    # Strong signals
    pa_keywords = [
        "prior authorization",
        "prior auth",
        "preauthorization",
        "pre-authorization",
        "pre-certification",
        "precertification",
        "not pre-certified",
        "authorization required",
        "pa required",
        "pa not obtained",
    ]

    for keyword in pa_keywords:
        if keyword in denial_text:
            score += 3.0
            signals.append(
                RouteSignal(
                    signal="prior_auth_keyword",
                    weight=0.9,
                    source="denial_reason",
                    evidence=keyword,
                )
            )
            break  # Only count once

    # Check PA info in facts
    if (
        facts.prior_authorization
        and facts.prior_authorization.was_required
        and not facts.prior_authorization.was_obtained
    ):
        score += 2.0
        signals.append(
            RouteSignal(
                signal="pa_required_not_obtained",
                weight=0.8,
                source="prior_authorization",
                evidence="PA was required but not obtained",
            )
        )

    # Negative signals
    if "medical necessity" in denial_text:
        score -= 1.0
        signals.append(
            RouteSignal(
                signal="medical_necessity_mentioned",
                weight=-0.3,
                source="denial_reason",
                evidence="medical necessity",
            )
        )

    return max(score, 0), signals


def _detect_coding_signals(facts: CaseFacts) -> tuple[float, list[RouteSignal]]:
    """Detect signals indicating coding/billing issue."""
    score = 0.0
    signals: list[RouteSignal] = []
    denial_text = facts.denial_reason.lower()

    # Strong signals
    coding_keywords = [
        "coding error",
        "invalid code",
        "incorrect code",
        "modifier",
        "wrong modifier",
        "missing modifier",
        "npi",
        "invalid npi",
        "duplicate claim",
        "already paid",
        "timely filing",
        "late submission",
        "eligibility",
        "not eligible",
        "coordination of benefits",
        "cob",
        "other insurance",
    ]

    for keyword in coding_keywords:
        if keyword in denial_text:
            score += 3.0
            signals.append(
                RouteSignal(
                    signal="coding_keyword",
                    weight=0.9,
                    source="denial_reason",
                    evidence=keyword,
                )
            )
            break

    # Check for CARC codes that indicate billing issues
    billing_carcs = {"CO-4", "CO-16", "CO-18", "CO-22", "PR-1", "CO-29"}
    for code in facts.denial_codes:
        if code.code in billing_carcs:
            score += 2.0
            signals.append(
                RouteSignal(
                    signal="billing_carc_code",
                    weight=0.7,
                    source="denial_codes",
                    evidence=code.code,
                )
            )

    return max(score, 0), signals


def _detect_medical_necessity_signals(facts: CaseFacts) -> tuple[float, list[RouteSignal]]:
    """Detect signals indicating medical necessity denial."""
    score = 1.0  # Base score (default route)
    signals: list[RouteSignal] = []
    denial_text = facts.denial_reason.lower()

    # Strong signals
    med_nec_keywords = [
        "medical necessity",
        "medically necessary",
        "not medically necessary",
        "experimental",
        "investigational",
        "clinical criteria",
        "clinical policy",
        "does not meet criteria",
        "medical review",
        "utilization review",
        "peer review",
    ]

    for keyword in med_nec_keywords:
        if keyword in denial_text:
            score += 3.0
            signals.append(
                RouteSignal(
                    signal="medical_necessity_keyword",
                    weight=0.9,
                    source="denial_reason",
                    evidence=keyword,
                )
            )
            break

    # Additional signals
    if "documentation" in denial_text:
        score += 1.0
        signals.append(
            RouteSignal(
                signal="documentation_mentioned",
                weight=0.5,
                source="denial_reason",
                evidence="documentation",
            )
        )

    if "clinical" in denial_text:
        score += 1.0
        signals.append(
            RouteSignal(
                signal="clinical_mentioned",
                weight=0.5,
                source="denial_reason",
                evidence="clinical",
            )
        )

    return max(score, 0), signals


def _calculate_confidence(selected_score: float, all_scores: dict[RouteType, float]) -> float:
    """Calculate confidence based on score differential."""
    if selected_score == 0:
        return 0.3  # Low confidence default

    total = sum(all_scores.values())
    if total == 0:
        return 0.3

    # Confidence based on how dominant the selected route is
    ratio = selected_score / total

    # Also consider absolute score
    absolute_factor = min(selected_score / 5, 1.0)  # Max out at score of 5

    confidence = ratio * 0.6 + absolute_factor * 0.4
    return round(min(max(confidence, 0.1), 0.95), 2)


def _build_reasoning(
    route: RouteType,
    signals: list[RouteSignal],
    facts: CaseFacts,
) -> str:
    """Build human-readable reasoning for the route selection."""
    route_descriptions = {
        RouteType.PRIOR_AUTH_NEEDED: "prior authorization issue",
        RouteType.CLAIM_CORRECTION_RESUBMIT: "coding or billing issue",
        RouteType.MEDICAL_NECESSITY_APPEAL: "medical necessity determination",
    }

    # Get positive signals for this route
    route_signals = [s for s in signals if s.weight > 0]

    if route_signals:
        evidence_list = [s.evidence for s in route_signals if s.evidence]
        evidence_str = ", ".join(evidence_list[:3])  # Limit to 3 examples
        return (
            f"The denial appears to be related to a {route_descriptions[route]}. "
            f"Key indicators: {evidence_str}."
        )
    else:
        return (
            f"Routed to {route_descriptions[route]} as the default action. "
            "No strong signals detected for other routes."
        )


def _determine_urgency(facts: CaseFacts) -> tuple[Urgency, str | None]:
    """Determine urgency level based on case facts."""
    if facts.dates and facts.dates.appeal_deadline_days:
        days = facts.dates.appeal_deadline_days
        if days <= 7:
            return Urgency.URGENT, f"Only {days} days until appeal deadline"
        elif days <= 14:
            return Urgency.EXPEDITED, f"{days} days until appeal deadline"

    if facts.dates and facts.dates.appeal_deadline:
        from datetime import date

        days_remaining = (facts.dates.appeal_deadline - date.today()).days
        if days_remaining <= 7:
            return Urgency.URGENT, f"Only {days_remaining} days until appeal deadline"
        elif days_remaining <= 14:
            return Urgency.EXPEDITED, f"{days_remaining} days until appeal deadline"

    return Urgency.STANDARD, None
