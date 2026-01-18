"""Prompt templates for LLM interactions."""

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
