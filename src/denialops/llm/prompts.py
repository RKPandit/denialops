"""Prompt templates for DenialOps LLM interactions.

This module contains all prompt templates used for LLM calls, organized by function.
Each prompt follows best practices:
- Clear role setting in system prompts
- Structured output requirements (JSON)
- XML tags for input boundaries
- Explicit handling of missing data
"""

# =============================================================================
# Extraction Prompts
# =============================================================================

EXTRACT_FACTS_SYSTEM = """You are an expert at analyzing health insurance denial letters.
Extract structured information from the denial letter text provided.
Always respond with valid JSON. Be precise and accurate.
If information is not present in the letter, use null for that field."""

EXTRACT_FACTS_USER = """Analyze this insurance denial letter and extract the following information as JSON:

<denial_letter>
{text}
</denial_letter>

Extract and return a JSON object with these fields:
{{
  "denial_reason": "The full denial reason as stated in the letter",
  "denial_reason_summary": "A short 5-10 word summary of why the claim was denied",
  "denial_codes": [
    {{"code": "string", "code_type": "CPT|HCPCS|ICD_10_CM|ICD_10_PCS|CARC|RARC", "description": "optional description"}}
  ],
  "service": {{
    "description": "Description of the service that was denied",
    "cpt_codes": ["list of CPT codes for the service"],
    "diagnosis_codes": ["list of ICD-10 diagnosis codes"],
    "date_of_service": "YYYY-MM-DD or null",
    "provider_name": "Name of the provider",
    "facility_name": "Name of the facility"
  }},
  "payer": {{
    "name": "Insurance company name",
    "plan_name": "Specific plan name if mentioned",
    "plan_type": "employer|marketplace|medicare|medicaid|individual|unknown",
    "member_id": "Member/subscriber ID",
    "group_number": "Group number if mentioned",
    "claim_number": "Claim or reference number"
  }},
  "dates": {{
    "date_of_service": "YYYY-MM-DD or null",
    "date_of_denial": "YYYY-MM-DD - the date on the letter",
    "appeal_deadline_days": "Number of days allowed to appeal (integer)",
    "timely_filing_deadline": "YYYY-MM-DD or null"
  }},
  "amounts": {{
    "billed_amount": 0.00,
    "allowed_amount": 0.00,
    "paid_amount": 0.00,
    "patient_responsibility": 0.00
  }},
  "contact_info": {{
    "phone": "Phone number for appeals",
    "fax": "Fax number for appeals",
    "address": "Mailing address for appeals",
    "website": "Website if mentioned"
  }},
  "prior_auth_mentioned": true/false,
  "medical_necessity_mentioned": true/false,
  "coding_issue_mentioned": true/false,
  "timely_filing_mentioned": true/false
}}

Important:
- Only extract CPT codes that are explicitly procedure codes (5 digits like 72148), not zip codes or other numbers
- Distinguish between the denial date (letter date) and the date of service
- Extract the appeal deadline in days if mentioned (e.g., "within 180 days")
- For amounts, extract numeric values only (no $ signs in the output)

Return ONLY valid JSON, no other text."""


# Legacy prompts (kept for backwards compatibility)
EXTRACT_DENIAL_REASON = """You are an expert at analyzing insurance denial letters. Extract the denial reason from the following text.

<denial_letter>
{text}
</denial_letter>

Provide your response in the following JSON format:
```json
{{
  "denial_reason": "The exact or paraphrased denial reason from the letter",
  "denial_reason_summary": "A plain-language 1-2 sentence summary of why the claim was denied",
  "confidence": 0.0-1.0
}}
```

If you cannot find a clear denial reason, set confidence to a low value and explain in the summary.
"""

EXTRACT_PAYER_INFO = """You are an expert at analyzing insurance documents. Extract payer/insurance information from the following text.

<document>
{text}
</document>

Provide your response in the following JSON format:
```json
{{
  "payer_name": "Insurance company name or null",
  "plan_name": "Specific plan name or null",
  "plan_type": "HMO|PPO|EPO|POS|HDHP|Medicare|Medicaid|unknown",
  "member_id": "Member ID or null",
  "group_number": "Group number or null",
  "claim_number": "Claim reference number or null"
}}
```

Only include values you can confidently extract from the text.
"""


# =============================================================================
# Action Plan Generation Prompts
# =============================================================================

GENERATE_PLAN_SYSTEM = """You are an expert patient advocate helping people understand and respond to insurance claim denials.
Generate clear, actionable guidance that is:
- Specific to the patient's situation
- Grounded in the facts extracted from their denial letter
- Written in plain language a non-expert can understand
- Empowering without being overly optimistic

Always respond with valid JSON."""

GENERATE_PLAN_SUMMARY_USER = """Based on the following case facts, generate a personalized summary and recommendation.

<case_facts>
{facts_json}
</case_facts>

<route>
{route}
</route>

Generate a JSON response with:
{{
  "situation_summary": "A 2-3 sentence summary of the patient's situation in plain language",
  "recommendation": "A clear 1-2 sentence recommendation for what they should do",
  "key_points": ["3-5 key points they need to know"],
  "urgency": "low|medium|high",
  "success_factors": ["2-4 factors that will determine success"]
}}

Ground your response ONLY in the provided facts. Do not make assumptions about information not present.
If key information is missing, acknowledge it in your response.

Return ONLY valid JSON."""

# Legacy prompt
GENERATE_ACTION_PLAN = """You are an expert insurance appeals advisor. Generate an action plan for the following denial case.

<case_facts>
{facts_json}
</case_facts>

<route>
{route}
</route>

<mode>
{mode}
</mode>

Generate a detailed action plan with:
1. A clear summary of the situation
2. Step-by-step actions the patient should take
3. Evidence they need to gather
4. Important deadlines

IMPORTANT RULES:
- Only reference dates, codes, and amounts that appear in the case_facts
- If information is missing, explicitly note it as "Unknown" or "Not provided"
- Include disclaimers that this is not legal or medical advice
- Be specific and actionable

Provide your response as a valid JSON object matching the ActionPlan schema.
"""


# =============================================================================
# Document Generation Prompts
# =============================================================================

GENERATE_APPEAL_LETTER_SYSTEM = """You are an expert at writing insurance appeal letters.
Write clear, professional letters that:
- Are factual and specific
- Reference the denial reason directly
- Advocate effectively for the patient
- Maintain a respectful, professional tone

Include placeholders in brackets [like this] for information the patient needs to fill in."""

GENERATE_APPEAL_LETTER_USER = """Generate a medical necessity appeal letter based on these facts:

<case_facts>
{facts_json}
</case_facts>

The letter should:
1. Reference the specific denial (claim number, date, service)
2. Clearly state why the patient is appealing
3. Include space for the patient to add their personal situation
4. Reference the supporting documentation they should attach
5. End with a clear request for reconsideration

Use the patient's extracted information where available, and [BRACKETS] for information they need to provide.

Generate the letter in markdown format."""

# Legacy prompt
GENERATE_APPEAL_LETTER = """You are an expert at writing insurance appeal letters. Generate an appeal letter based on the following case.

<case_facts>
{facts_json}
</case_facts>

<action_plan>
{plan_json}
</action_plan>

Write a professional, persuasive appeal letter that:
1. Clearly identifies the claim being appealed
2. States the reason for the appeal
3. Provides a logical argument for why the denial should be overturned
4. Requests specific action (approval of the claim)

IMPORTANT RULES:
- Use only information from the case_facts - do not invent details
- Include placeholders like [YOUR NAME] for information that needs to be filled in
- Be professional but empathetic
- Keep it concise (under 2 pages)

Output the letter in markdown format.
"""

GENERATE_CALL_SCRIPT_SYSTEM = """You are an expert at helping patients navigate insurance phone calls.
Create scripts that:
- Are easy to follow step-by-step
- Anticipate common questions and responses
- Help the caller stay organized
- Include space to write down important information"""

GENERATE_CALL_SCRIPT_USER = """Generate a phone call script for contacting the insurance company about this denial:

<case_facts>
{facts_json}
</case_facts>

<purpose>
{purpose}
</purpose>

The script should include:
1. Information to have ready before calling
2. What to say when connected to a representative
3. Key questions to ask based on the denial type
4. What information to write down
5. Next steps after the call

Generate the script in markdown format with clear sections."""


# =============================================================================
# Grounding Validation Prompts
# =============================================================================

VALIDATE_GROUNDING_SYSTEM = """You are a fact-checker verifying that generated content is grounded in source data.
Identify any claims, codes, dates, or amounts that do not appear in the source facts."""

VALIDATE_GROUNDING_USER = """Check if this generated content is properly grounded in the source facts.

<source_facts>
{facts_json}
</source_facts>

<generated_content>
{content}
</generated_content>

Return a JSON response:
{{
  "is_grounded": true/false,
  "ungrounded_claims": [
    {{"claim": "the ungrounded statement", "issue": "why it's not grounded"}}
  ],
  "hallucinated_codes": ["list of codes not in source"],
  "hallucinated_dates": ["list of dates not in source"],
  "hallucinated_amounts": ["list of amounts not in source"]
}}

Return ONLY valid JSON."""


# =============================================================================
# Helper Functions
# =============================================================================


def format_extraction_prompt(text: str) -> tuple[str, str]:
    """Format the extraction prompt with the given text.

    Args:
        text: The denial letter text to analyze

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    return EXTRACT_FACTS_SYSTEM, EXTRACT_FACTS_USER.format(text=text)


def format_plan_summary_prompt(facts_json: str, route: str) -> tuple[str, str]:
    """Format the plan summary prompt.

    Args:
        facts_json: JSON string of case facts
        route: The selected route type

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    return GENERATE_PLAN_SYSTEM, GENERATE_PLAN_SUMMARY_USER.format(
        facts_json=facts_json, route=route
    )


def format_appeal_letter_prompt(facts_json: str) -> tuple[str, str]:
    """Format the appeal letter generation prompt.

    Args:
        facts_json: JSON string of case facts

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    return GENERATE_APPEAL_LETTER_SYSTEM, GENERATE_APPEAL_LETTER_USER.format(facts_json=facts_json)


def format_call_script_prompt(facts_json: str, purpose: str) -> tuple[str, str]:
    """Format the call script generation prompt.

    Args:
        facts_json: JSON string of case facts
        purpose: Purpose of the call (e.g., "follow up on prior auth")

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    return GENERATE_CALL_SCRIPT_SYSTEM, GENERATE_CALL_SCRIPT_USER.format(
        facts_json=facts_json, purpose=purpose
    )


def format_grounding_validation_prompt(facts_json: str, content: str) -> tuple[str, str]:
    """Format the grounding validation prompt.

    Args:
        facts_json: JSON string of source facts
        content: Generated content to validate

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    return VALIDATE_GROUNDING_SYSTEM, VALIDATE_GROUNDING_USER.format(
        facts_json=facts_json, content=content
    )


def format_action_plan_prompt(facts_json: str, route: str, mode: str) -> tuple[str, str]:
    """Format the action plan generation prompt.

    Args:
        facts_json: JSON string of case facts
        route: The selected route type
        mode: Processing mode (fast/verified)

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    return GENERATE_PLAN_SYSTEM, GENERATE_ACTION_PLAN.format(
        facts_json=facts_json, route=route, mode=mode
    )


class PromptLibrary:
    """Central registry of all prompts for easy access and modification."""

    # Extraction
    EXTRACT_FACTS_SYSTEM = EXTRACT_FACTS_SYSTEM
    EXTRACT_FACTS_USER = EXTRACT_FACTS_USER
    EXTRACT_DENIAL_REASON = EXTRACT_DENIAL_REASON
    EXTRACT_PAYER_INFO = EXTRACT_PAYER_INFO

    # Plan Generation
    GENERATE_PLAN_SYSTEM = GENERATE_PLAN_SYSTEM
    GENERATE_PLAN_SUMMARY_USER = GENERATE_PLAN_SUMMARY_USER
    GENERATE_ACTION_PLAN = GENERATE_ACTION_PLAN

    # Document Generation
    GENERATE_APPEAL_LETTER_SYSTEM = GENERATE_APPEAL_LETTER_SYSTEM
    GENERATE_APPEAL_LETTER_USER = GENERATE_APPEAL_LETTER_USER
    GENERATE_APPEAL_LETTER = GENERATE_APPEAL_LETTER
    GENERATE_CALL_SCRIPT_SYSTEM = GENERATE_CALL_SCRIPT_SYSTEM
    GENERATE_CALL_SCRIPT_USER = GENERATE_CALL_SCRIPT_USER

    # Validation
    VALIDATE_GROUNDING_SYSTEM = VALIDATE_GROUNDING_SYSTEM
    VALIDATE_GROUNDING_USER = VALIDATE_GROUNDING_USER

    @classmethod
    def get_all_prompts(cls) -> dict[str, str]:
        """Return all prompts as a dictionary."""
        return {
            name: value
            for name, value in vars(cls).items()
            if isinstance(value, str) and not name.startswith("_")
        }

    @classmethod
    def validate_prompt(cls, prompt: str, required_vars: list[str]) -> bool:
        """Validate that a prompt contains required format variables.

        Args:
            prompt: The prompt template string
            required_vars: List of required variable names (e.g., ["text", "facts_json"])

        Returns:
            True if all required variables are present
        """
        return all(f"{{{var}}}" in prompt for var in required_vars)
