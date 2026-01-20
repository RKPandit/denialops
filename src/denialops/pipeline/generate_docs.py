"""Document generation for action plans."""

from datetime import date

from denialops.models.action_plan import ActionPlan
from denialops.models.case import CaseFacts
from denialops.models.plan_rules import PlanRules
from denialops.models.route import RouteType


def generate_document_pack(
    facts: CaseFacts,
    plan: ActionPlan,
    plan_rules: PlanRules | None = None,
) -> dict[str, str]:
    """
    Generate all documents for an action plan.

    Args:
        facts: Extracted case facts
        plan: Generated action plan
        plan_rules: Optional PlanRules from SBC (for verified mode with citations)

    Returns a dict mapping filename to content.
    """
    documents: dict[str, str] = {}

    # Always generate call script
    documents["call_script.md"] = _generate_call_script(facts, plan)

    # Route-specific documents
    if plan.route == RouteType.PRIOR_AUTH_NEEDED:
        documents["pa_checklist.md"] = _generate_pa_checklist(facts, plan, plan_rules)

    elif plan.route == RouteType.CLAIM_CORRECTION_RESUBMIT:
        documents["resubmit_checklist.md"] = _generate_resubmit_checklist(facts, plan)

    elif plan.route == RouteType.MEDICAL_NECESSITY_APPEAL:
        documents["appeal_letter.md"] = _generate_appeal_letter(facts, plan, plan_rules)

    return documents


def _generate_call_script(facts: CaseFacts, plan: ActionPlan) -> str:
    """Generate a call script for contacting insurance."""
    payer_name = facts.payer.name if facts.payer else "[Insurance Company]"
    member_id = facts.payer.member_id if facts.payer else "[Your Member ID]"
    claim_number = facts.payer.claim_number if facts.payer else "[Claim Number]"

    script = f"""# Call Script: {payer_name}

## Before You Call
- Have your denial letter ready
- Have a pen and paper to take notes
- Set aside 20-30 minutes

## Information to Have Ready
- Member ID: {member_id}
- Claim Number: {claim_number}
- Date of Service: {facts.dates.date_of_service if facts.dates and facts.dates.date_of_service else "[Date]"}
- Provider Name: {facts.service.provider_name if facts.service and facts.service.provider_name else "[Provider Name]"}

## When You Call

### 1. Get Representative Information
"Hello, I'm calling about a denied claim. Before we begin, may I have your name and a reference number for this call?"

**Write down:** _________________________

### 2. Verify Your Information
"My name is [YOUR NAME] and my member ID is {member_id}. I'm calling about claim number {claim_number}."

### 3. Ask About the Denial
"Can you explain why this claim was denied?"

**Write down their explanation:** _________________________

### 4. Ask About Next Steps
"""

    if plan.route == RouteType.PRIOR_AUTH_NEEDED:
        script += """
"What do I need to do to get prior authorization for this service?"
"Can my provider submit this retroactively?"
"Once I have the prior authorization, how do I get this claim reconsidered?"
"""
    elif plan.route == RouteType.CLAIM_CORRECTION_RESUBMIT:
        script += """
"What specifically needs to be corrected on this claim?"
"What is the timely filing deadline for resubmission?"
"Should my provider submit a corrected claim or a new claim?"
"""
    else:  # Appeal
        script += """
"What is the deadline to file an appeal?"
"What address should I send my appeal to?"
"What documentation do you need to support the appeal?"
"Is there a specific form I need to use?"
"""

    script += """
### 5. Confirm and Close
"Let me confirm what I need to do: [repeat back what they said]"
"Is there anything else I should know?"
"Thank you for your help, [Representative Name]."

## After the Call
- Date of call: __________
- Representative name: __________
- Reference number: __________
- Key information: __________
- Next steps: __________

---

"""
    script += _get_disclaimer()

    return script


def _generate_pa_checklist(
    facts: CaseFacts, plan: ActionPlan, plan_rules: PlanRules | None = None
) -> str:
    """Generate prior authorization checklist."""
    # Build overview with plan-specific citations if available
    overview = (
        "Your claim was denied because prior authorization was not obtained. "
        "Follow this checklist to get authorization and request reconsideration."
    )

    if plan_rules and plan_rules.prior_authorization_rules:
        # Find relevant PA rule
        service_desc = (
            facts.service.description if facts.service and facts.service.description else ""
        )
        pa_rule = plan_rules.get_pa_rule_for_service(service_desc)
        if pa_rule:
            citation = pa_rule.get_citation().format_citation()
            if pa_rule.conditions:
                overview += f"\n\n**Plan Policy:** {pa_rule.conditions} {citation}"

    checklist = f"""# Prior Authorization Checklist

## Overview
{overview}

## Step 1: Contact Your Provider
- [ ] Call your healthcare provider's office
- [ ] Inform them the claim was denied due to missing prior authorization
- [ ] Ask them to submit a prior authorization request
- [ ] Get a contact name and number for follow-up

Provider contact: _________________________
Date contacted: _________________________

## Step 2: Information Your Provider Needs
Make sure your provider has:
- [ ] Your current insurance information
- [ ] The specific procedure/service that needs authorization
- [ ] Supporting medical records
- [ ] Diagnosis codes (ICD-10)
- [ ] Procedure codes (CPT/HCPCS)

## Step 3: Track the PA Request
- [ ] Get the PA request submission date from your provider
- [ ] Ask for the PA reference number once submitted
- [ ] Note the expected decision timeframe

PA submitted date: _________________________
PA reference number: _________________________
Expected decision by: _________________________

## Step 4: Follow Up
- [ ] If no response in 5-7 business days, call insurance
- [ ] Use the call script provided
- [ ] Document each follow-up call

## Step 5: After PA Approval
- [ ] Get the PA approval letter/number
- [ ] Call insurance to request reconsideration of original claim
- [ ] Provide the PA number
- [ ] Ask for confirmation the claim will be reprocessed

PA approval number: _________________________
Reconsideration requested date: _________________________

## Important Deadlines
"""

    if plan.timeline and plan.timeline.appeal_deadline:
        checklist += f"- Appeal deadline: {plan.timeline.appeal_deadline}\n"
    else:
        checklist += "- Appeal deadline: [Check your denial letter]\n"

    checklist += """
## Tips for Success
1. Be persistent - follow up regularly
2. Keep copies of everything
3. Get names and reference numbers for every call
4. If the PA is taking too long, ask about expedited review

---

"""
    checklist += _get_disclaimer()

    return checklist


def _generate_resubmit_checklist(facts: CaseFacts, plan: ActionPlan) -> str:
    """Generate claim resubmission checklist."""
    checklist = """# Claim Resubmission Checklist

## Overview
Your claim was denied due to a coding or billing issue. Follow this checklist to get the claim corrected and resubmitted.

## Step 1: Identify the Problem
Review your denial letter for:
- [ ] Denial reason code
- [ ] Specific error mentioned
- [ ] What information was missing or incorrect

Denial reason: _________________________
Error identified: _________________________

## Step 2: Contact Provider Billing
- [ ] Call your healthcare provider's billing department
- [ ] Provide the denial letter information
- [ ] Ask them to review and identify the error
- [ ] Request they correct and resubmit the claim

Billing department phone: _________________________
Contact name: _________________________
Date contacted: _________________________

## Step 3: Common Issues to Check
- [ ] Correct procedure codes (CPT/HCPCS)
- [ ] Correct diagnosis codes (ICD-10)
- [ ] Required modifiers included
- [ ] Correct provider NPI
- [ ] Correct date of service
- [ ] Patient information matches insurance records
- [ ] Referring provider information (if required)

## Step 4: Resubmission
- [ ] Confirm billing department will resubmit
- [ ] Get the resubmission date
- [ ] Get a new claim/reference number if available

Resubmitted date: _________________________
New claim number: _________________________

## Step 5: Follow Up
- [ ] Wait 2-3 weeks for processing
- [ ] Call insurance to verify receipt
- [ ] Check claim status
- [ ] If denied again, ask for specific reason

Follow-up date: _________________________
Status: _________________________

## Important Deadlines
"""

    if facts.dates and facts.dates.timely_filing_deadline:
        checklist += f"- Timely filing deadline: {facts.dates.timely_filing_deadline}\n"
    else:
        checklist += "- Timely filing deadline: [Ask your insurance - typically 90-365 days]\n"

    checklist += """
## Tips for Success
1. Keep a copy of the original and corrected claims
2. Document all communications with billing department
3. If billing says they can't fix it, ask to speak with a supervisor
4. Consider requesting an itemized bill to review charges

---

"""
    checklist += _get_disclaimer()

    return checklist


def _generate_appeal_letter(
    facts: CaseFacts, plan: ActionPlan, plan_rules: PlanRules | None = None
) -> str:
    """Generate appeal letter template."""
    today = date.today().strftime("%B %d, %Y")
    payer_name = facts.payer.name if facts.payer else "[Insurance Company Name]"
    member_id = facts.payer.member_id if facts.payer else "[Your Member ID]"
    claim_number = facts.payer.claim_number if facts.payer else "[Claim Number]"
    dos = (
        facts.dates.date_of_service.strftime("%B %d, %Y")
        if facts.dates and facts.dates.date_of_service
        else "[Date of Service]"
    )
    service = (
        facts.service.description
        if facts.service and facts.service.description
        else "[Service/Procedure Name]"
    )
    denial_date = (
        facts.dates.date_of_denial.strftime("%B %d, %Y")
        if facts.dates and facts.dates.date_of_denial
        else "[Denial Date]"
    )

    # Build policy citations section if plan rules available
    policy_citations = ""
    if plan_rules:
        citations_parts = []

        # Add appeal rights citation
        if plan_rules.appeal_rights:
            ar = plan_rules.appeal_rights
            if ar.internal_appeal_deadline_days:
                citation = ar.get_citation().format_citation()
                citations_parts.append(
                    f"- You have {ar.internal_appeal_deadline_days} days to file an appeal {citation}"
                )
            if ar.expedited_appeal_available and ar.expedited_criteria:
                citations_parts.append(
                    f"- Expedited appeal available when: {ar.expedited_criteria}"
                )

        # Add medical necessity criteria citation
        if plan_rules.medical_necessity_criteria:
            mn = plan_rules.medical_necessity_criteria[0]
            if mn.definition:
                citation = mn.get_citation().format_citation()
                citations_parts.append(
                    f"- Plan defines medical necessity as: \"{mn.definition}\" {citation}"
                )

        if citations_parts:
            policy_citations = """
**Your Plan's Appeal Rights** (from your Summary of Benefits and Coverage)

""" + "\n".join(citations_parts) + "\n"

    letter = f"""# Appeal Letter Template

**Instructions:** Fill in the bracketed sections with your information. Attach all supporting documents.

---

[Your Name]
[Your Address]
[City, State ZIP]
[Your Phone Number]
[Your Email]

{today}

{payer_name}
Appeals Department
[Appeals Address from Denial Letter]
[City, State ZIP]

**RE: Appeal of Claim Denial**
- Member Name: [Your Name]
- Member ID: {member_id}
- Claim Number: {claim_number}
- Date of Service: {dos}
- Date of Denial: {denial_date}

Dear Appeals Committee:

I am writing to formally appeal the denial of coverage for {service} that I received on {dos}. I received the denial letter dated {denial_date}, which stated the reason for denial as:

> {facts.denial_reason}
{policy_citations}
**Why I Am Appealing**

I believe this denial should be overturned because [explain why you believe the service is medically necessary and should be covered. Be specific about your condition, symptoms, and why this treatment is needed].

**My Medical Situation**

[Describe your medical condition, how long you have had it, what symptoms you experience, and how it affects your daily life. Be specific but concise.]

**Treatment History**

[Describe what treatments you have already tried, if applicable. This shows the service isn't the first option tried.]

**Supporting Documentation**

I have enclosed the following documents to support this appeal:
- [ ] Copy of the denial letter
- [ ] Letter of medical necessity from my treating physician, Dr. [Provider Name]
- [ ] Relevant medical records
- [ ] [List any other supporting documents]

**Request**

I respectfully request that you reverse this denial and approve coverage for {service}. This treatment is medically necessary for my condition, and denial of coverage would [explain the impact - e.g., cause continued suffering, delay necessary treatment, create financial hardship].

Please contact me at [Your Phone] or [Your Email] if you need any additional information.

Thank you for your prompt attention to this matter.

Sincerely,

[Your Signature]

[Your Printed Name]

---

## Enclosures Checklist
- [ ] Copy of denial letter
- [ ] Letter of medical necessity from provider
- [ ] Relevant medical records
- [ ] Lab results/imaging reports (if applicable)
- [ ] [Other supporting documents]

## Mailing Tips
1. Make copies of everything before mailing
2. Send via certified mail with return receipt
3. Keep the tracking number
4. Note the date mailed: _________________________

---

"""
    letter += _get_disclaimer()

    return letter


def _get_disclaimer() -> str:
    """Get standard disclaimer text."""
    return """**IMPORTANT DISCLAIMERS**

This document was generated by an automated system and is provided for informational purposes only.

- This is NOT legal advice. For legal questions, consult a licensed attorney.
- This is NOT medical advice. For medical questions, consult your healthcare provider.
- Coverage details may vary. Verify all information with your insurance company.
- Deadlines are estimates. Confirm actual deadlines with your insurer.

The creators of this tool are not responsible for decisions made based on this information.
"""
