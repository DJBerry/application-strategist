"""Prompt templates for the job description extraction graph.

Each node type has a dedicated system prompt and user message template so they
can be read, tuned, and tested in isolation.  Keep extraction, validation, and
retry concerns fully separated here.
"""

# ---------------------------------------------------------------------------
# EXTRACT — initial structured extraction from the job description
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM_PROMPT = """\
You are a job description analyst. Extract structured information from the job \
description provided by the user.

Rules:
- Only use information that is explicitly stated in the job description.
- Extract the most relevant available text for each field, even if it is not \
  explicitly labeled as such (e.g., use a "we exist to..." statement as the \
  mission if no formal mission is given). Only use "N/A" when no relevant \
  information is present at all.
- "work_environment" must be exactly one of: "remote", "hybrid", "on-site", "N/A".
- Do not fabricate, infer, or guess any values.
- Output a raw JSON object only — no markdown fences, no prose, no explanation.

Return this exact structure:
{
  "company_info": {
    "company_name": "Company name as stated, or N/A",
    "company_description": "Brief description of what the company does, or N/A",
    "company_mission": "Company mission or values statement, or N/A"
  },
  "job_info": {
    "title": "Exact job title",
    "seniority_level": "e.g. Senior, Mid-level, Junior, Staff, Principal, or N/A",
    "location": "City, state, country, or N/A",
    "work_environment": "remote|hybrid|on-site|N/A"
  }
}"""

EXTRACT_USER_TEMPLATE = """\
Job Description:
{job_description}

Extract the structured information. Output JSON only."""


# ---------------------------------------------------------------------------
# CHECK — validate the extracted data against the original job description
# ---------------------------------------------------------------------------

CHECK_SYSTEM_PROMPT = """\
You are a data validator. You will receive extracted job description data and \
the original job description text.

Your task: classify each extracted field as correct, incorrect, or ambiguous.

Classifications:
- INCORRECT: the extracted value directly contradicts what is written in the \
  job description, is fabricated, or "N/A" was returned when relevant text is \
  clearly present. These fields must be re-extracted.
- AMBIGUOUS: the extracted value is a valid reading of the job description, but \
  the job description also contains conditions or caveats that qualify it \
  (e.g., "remote, but hybrid near HQ"). The extracted value is not wrong — it \
  just has nuance worth noting. Do NOT use ambiguous for fields where "N/A" was \
  used incorrectly; that is incorrect.
- CORRECT: the value accurately reflects what is stated and needs no action.

Rules:
- Never put the same field in both incorrect_fields and ambiguous_fields.
- Set all_correct to true when incorrect_fields is empty. Ambiguous fields \
  alone do not block the "all correct" result.
- "work_environment" must be one of: "remote", "hybrid", "on-site", "N/A".
- Do not penalise minor wording differences as long as the meaning is accurate.
- Output a raw JSON object only — no markdown fences, no prose.

Return this exact structure:
{
  "all_correct": true,
  "incorrect_fields": [],
  "ambiguous_fields": []
}

Example showing the distinction:
{
  "all_correct": false,
  "incorrect_fields": [
    {
      "field": "company_info.company_mission",
      "extracted_value": "N/A",
      "explanation": "JD contains 'we exist to empower developers' which should have been extracted as the mission"
    }
  ],
  "ambiguous_fields": [
    {
      "field": "job_info.work_environment",
      "extracted_value": "remote",
      "explanation": "Correct for most staff, but JD notes hybrid arrangement for candidates near HQ"
    }
  ]
}"""

CHECK_USER_TEMPLATE = """\
Original Job Description:
{job_description}

---
Extracted Data:
{extracted_data}

Validate each field. Output JSON only."""


# ---------------------------------------------------------------------------
# RETRY — re-extract only the fields that failed validation
# ---------------------------------------------------------------------------

# NOTE: {mistakes} is a pre-formatted human-readable string built in nodes.py,
# not raw JSON.  This avoids brace-escaping conflicts with str.format().
RETRY_SYSTEM_PROMPT_TEMPLATE = """\
You are a job description analyst. A previous extraction attempt produced \
incorrect values for some fields. Re-extract ONLY the listed incorrect fields.

Previous mistakes — do not repeat these:
{mistakes}

Rules:
- Only use information that is explicitly stated in the job description.
- If a field is genuinely absent, use "N/A".
- "work_environment" must be exactly one of: "remote", "hybrid", "on-site", "N/A".
- Output ONLY the corrected fields as a raw JSON object — no markdown fences, \
  no prose, no explanation.
- Use the same nested structure as the original extraction \
  (e.g. {{"company_info": {{"company_name": "..."}}}}).
- Do not include fields that were already correct."""

RETRY_USER_TEMPLATE = """\
Job Description:
{job_description}

---
Fields that need to be re-extracted:
{incorrect_fields}

Re-extract only these fields. Output JSON only."""


# ---------------------------------------------------------------------------
# EXTRACT_REQUIREMENTS — extract all requirements from the job description
# ---------------------------------------------------------------------------

# # v1 prompt
# EXTRACT_REQUIREMENTS_SYSTEM_PROMPT = """\
# You are a job description analyst. Extract every requirement mentioned in the \
# job description provided by the user.

# Rules:
# - Only extract requirements explicitly stated in the job description. Do not \
#   infer, assume, or add requirements that are not directly present in the text.
# - If the same skill or qualification appears in multiple places in the JD, \
#   produce ONE requirement that captures the fullest description — do not \
#   create separate duplicates.
# - If a JD bullet bundles multiple distinct qualifications together \
#   (e.g., "experience with Python and deploying ML models to production"), split \
#   them into separate requirements unless they are genuinely inseparable.
# - For each requirement's description: rephrase for clarity and consistency \
#   rather than copying JD language verbatim, but do NOT add qualifications, \
#   thresholds, or conditions that are not present in the source text. Capture \
#   all relevant nuance, conditions, and flexibility stated in the JD \
#   (e.g., if education can substitute for years of experience, include that).
# - Determine priority from the JD's own language cues:
#     "must", "required", "minimum", "mandatory"  → "minimum_requirement"
#     "preferred", "desired", "ideal", "important" → "preferred_requirement"
#     "bonus", "plus", "nice to have", "a plus"   → "nice_to_have"
#     unclear or contradictory language            → "ambiguous"
# - Do not fabricate, infer, or guess any values.
# - Output a raw JSON object only — no markdown fences, no prose, no explanation.

# Return this exact structure:
# {
#   "requirements": [
#     {
#       "label": "Short human-readable label, e.g. 'Python proficiency'",
#       "description": "Full description capturing all JD nuance and conditions",
#       "priority": "minimum_requirement|preferred_requirement|nice_to_have|ambiguous"
#     }
#   ]
# }"""

# v2 concise prompt
EXTRACT_REQUIREMENTS_SYSTEM_PROMPT = """\
You are a job description analyst. Extract **all explicitly stated \
requirements/qualifications** from the user-provided job description.

Extract **only** candidate requirements or qualifications. Do **not** \
extract responsibilities, duties, or infer unstated requirements.

Definitions:

- **Requirement/qualification**: Something the candidate must already \
  possess or satisfy before or as a condition of hire, such as a skill, \
  credential, experience level, knowledge area, attribute, work \
  authorization, clearance, travel/location constraint, language, \
  certification, license, physical requirement, or availability condition.
- **Responsibility/duty**: Something the candidate will do in the role, such \
  as tasks, activities, deliverables, or outcomes.

Examples:

- "Design and implement microservices architectures" → responsibility
- "Experience designing microservices architectures" → requirement
- "Collaborate with cross-functional teams" → responsibility
- "Strong communication skills and ability to work across teams" → requirement
- "You will manage a portfolio of enterprise accounts" → responsibility
- "3+ years of enterprise account management" → requirement

Rules:

- Extract only requirements explicitly stated in the JD.
- Requirements may appear anywhere in the JD, not only in qualifications \
  sections.
- Do not convert responsibilities into requirements unless the JD explicitly \
  frames them as qualifications or prerequisites.
- If an activity is listed in a qualifications section or explicitly treated \
  as a prerequisite, extract it as a requirement.
- If the same requirement appears multiple times, output one requirement \
  with the fullest explicitly stated description.
- Do not merge distinct tools, credentials, thresholds, or domains unless \
  the JD treats them as one inseparable qualification.
- Split bundled qualifications into separate requirements unless they are \
  genuinely inseparable.
- Rephrase for clarity and consistency, but do not add, infer, generalize, \
  or broaden the source meaning. Preserve all explicitly stated nuance, \
  thresholds, substitutions, and conditions.

Priority:

- Use both wording and section context.
- `"minimum_requirement"`: cues like "must," "required," "minimum," \
  "mandatory," or items under "Basic/Minimum/Required Qualifications"
- `"preferred_requirement"`: cues like "preferred," "desired," "ideal," or \
  items under "Preferred Qualifications"
- `"nice_to_have"`: cues like "bonus," "plus," "nice to have," "a plus," or \
  items under similar sections
- `"ambiguous"`: unclear or conflicting wording/section context

Output:

- Return raw JSON only.
- No markdown, prose, or explanation.
- If no explicit requirements are stated, return `{"requirements":[]}`

Return this exact structure:
{
  "requirements": [
    {
      "label": "Short human-readable label, e.g. 'Python proficiency'",
      "description": "Full description capturing all JD nuance and conditions",
      "priority": "minimum_requirement|preferred_requirement|nice_to_have|ambiguous"
    }
  ]
}"""

# # v2 verbose, clarity
# EXTRACT_REQUIREMENTS_SYSTEM_PROMPT = """\
# You are a job description analyst. Extract **ALL stated requirements** in \
# the job description provided by the user. Only extract requirements or \
# qualifications. Do NOT extract job responsibilities or duties, or attempt to \
# infer requirements not directly stated.

# For the purposes of this task, requirements/qualifications and \
# responsibilities/duties are defined as follows:
# - **Requirement/qualification**: Something the candidate is expected to already \
#   possess: a skill, credential, experience level, knowledge area, or \
#   attribute they are expected to bring before or as a condition of hire.
# - **Responsibility/duty**: Something the candidate will do, perform, or be \
#   tasked with once in the role: activities, tasks, deliverables, or outcomes \
#   they'll be responsible for.

# Some examples include:
# - "Design and implement microservices architectures" → responsibility (what \
#   you'll do), not a requirement
# - "Experience designing microservices architectures" → requirement (what you \
#   bring)
# - "Collaborate with cross-functional teams" → responsibility
# - "Strong communication skills and ability to work across teams" → \
#   requirement
# - "You will manage a portfolio of enterprise accounts" → responsibility
# - "3+ years of enterprise account management" → requirement

# Keep in mind that while wording matters, it is not the only consideration; \
# placement and framing should also be considered. If something is phrased as \
# an activity, but it appears in the job description's qualifications block, \
# or the job description explicitly treats the item as a credential or \
# prerequisite, then extract the item as a requirement/qualification anyway.

# Rules:
# - Only extract requirements explicitly stated in the job description. Do not \
#   infer, assume, or add requirements that are not directly present in the text.
# - Do not convert responsibilities or duties into requirements unless the job \
#   description frames them as qualifications or prerequisites.
# - Requirements can be found in any part of the job description, including \
#   qualifications sections, summary paragraphs, candidate profile text, and \
#   any other text explicitly framing something as required, preferred, or a \
#   condition of hire.
# - Treat explicitly stated work authorization, clearance, travel, location, \
#   onsite/hybrid, language, certification, license, physical, and \
#   availability conditions as requirements when present.
# - If the same skill or qualification appears in multiple places in the JD, \
#   produce ONE requirement that captures the fullest description. Do not \
#   merge distinct tools, credentials, thresholds, or domains into one \
#   requirement unless the source treats them as a single inseparable \
#   qualification.
# - Do not generalize specific technologies, credentials, or domain areas into \
#   broader categories.
# - If a JD bullet bundles multiple distinct qualifications together \
#   (e.g., "experience with Python and deploying ML models to production"), split \
#   them into separate requirements unless they are genuinely inseparable.
# - For each requirement's description: rephrase for clarity and consistency \
#   rather than copying JD language verbatim, but do NOT add qualifications, \
#   thresholds, or conditions that are not present in the source text. Capture \
#   all relevant nuance, conditions, and flexibility stated in the JD \
#   (e.g., if education can substitute for years of experience, include that).
# - Determine priority using a combination of the job description's own \
#   language cues and organization, such as section headers:
#     - Language:
#         - "must", "required", "minimum", "mandatory" → "minimum_requirement"
#         - "preferred", "desired", "ideal", "important" → \
#           "preferred_requirement"
#         - "bonus", "plus", "nice to have", "a plus" → "nice_to_have"
#         - unclear or contradictory language → "ambiguous"
#     - Section headers:
#         - bullets under “Basic Qualifications,” “Minimum Qualifications,” or \
#           “Required Qualifications” → "minimum_requirement"
#         - bullets under “Preferred Qualifications” → "preferred_requirement"
#         - bullets under “Nice to Have,” “Bonus,” or similar → "nice_to_have"
#     - If wording and section conflict → "ambiguous"
# - Do not fabricate, infer, or guess any values.
# - Output a raw JSON object only — no markdown fences, no prose, no explanation.
# - If no explicit requirements are stated, return {"requirements":[]}

# Return this exact JSON structure:
# {
#   "requirements": [
#     {
#       "label": "Short human-readable label, e.g. 'Python proficiency'",
#       "description": "Full description capturing all JD nuance and conditions",
#       "priority": "minimum_requirement|preferred_requirement|nice_to_have|ambiguous"
#     }
#   ]
# }"""

EXTRACT_REQUIREMENTS_USER_TEMPLATE = """\
Job Description:
{job_description}

---
Company and Job Context (from earlier extraction — use for context only):
Company Info:
{company_info}

Job Info:
{job_info}

Extract all requirements from the job description. Output JSON only."""


# ---------------------------------------------------------------------------
# VALIDATE_REQUIREMENTS — review extracted requirements for quality
# ---------------------------------------------------------------------------

VALIDATE_REQUIREMENTS_SYSTEM_PROMPT = """\
You are a requirements validator. You will receive an extracted requirements \
list and the original job description. Review the extraction for four things:

1. COMPLETENESS: Are any requirements from the JD missing from the list?
2. ACCURACY: Does each requirement's description faithfully reflect what the \
   JD says, without hallucination or invention? Flag any description that adds \
   information (qualifications, thresholds, conditions) not present in the \
   source text.
3. PRIORITY CORRECTNESS: Is the priority label reasonable given the language \
   used in the JD?
4. DUPLICATES: Are any requirements duplicates or substantially overlapping? \
   If so, the correction must MERGE them into a single requirement with the \
   most complete description — not simply delete one.

Behavioural guidelines — be pragmatic, not pedantic:
- Only flag CLEAR problems. Do not nitpick borderline cases.
- Tolerate ambiguity: if a priority could reasonably go either way, that is \
  NOT an error. If the JD language is genuinely unclear, "ambiguous" is an \
  acceptable and correct label — do not flag it.
- Do NOT trigger an issue over disagreements that are matters of interpretation. \
  If the extraction captures nuance faithfully (even in different words), do \
  not flag it.
- Set all_correct to true when issues is empty.
- Output a raw JSON object only — no markdown fences, no prose.

Issue types:
- "missing"       — a requirement is present in the JD but absent from the list
- "inaccurate"    — a requirement's description adds or misrepresents information
- "wrong_priority"— the priority label is clearly inconsistent with JD language
- "duplicate"     — two requirements are substantially overlapping; correction merges them

Return this exact structure:
{
  "all_correct": true,
  "issues": []
}

When issues exist:
{
  "all_correct": false,
  "issues": [
    {
      "type": "missing|inaccurate|wrong_priority|duplicate",
      "label": "label of the problematic requirement, or null for missing",
      "duplicate_of": "label of the other requirement (duplicate type only), or null",
      "problem": "clear explanation of the problem",
      "correction": {
        "label": "corrected label",
        "description": "corrected description",
        "priority": "corrected priority"
      }
    }
  ]
}"""

VALIDATE_REQUIREMENTS_USER_TEMPLATE = """\
Original Job Description:
{job_description}

---
Extracted Requirements:
{requirements}

Validate the requirements. Output JSON only."""


# ---------------------------------------------------------------------------
# CORRECT_REQUIREMENTS — apply validator corrections to the requirements list
# ---------------------------------------------------------------------------

# NOTE: {issues} is a pre-formatted human-readable string built in nodes.py,
# not raw JSON.  This avoids brace-escaping conflicts with str.format().
CORRECT_REQUIREMENTS_SYSTEM_PROMPT_TEMPLATE = """\
You are a job description analyst. A previous extraction of requirements \
contained some issues. Apply ONLY the listed corrections to the requirements \
list — do not re-extract from scratch.

Issues to fix:
{issues}

Rules for applying corrections:
- "missing": add the correction object as a new requirement to the list.
- "inaccurate" or "wrong_priority": replace the named requirement with the \
  correction object. Keep all other requirements unchanged.
- "duplicate": replace BOTH named requirements with the single merged correction \
  object. Do not keep either of the originals.
- Do NOT change any requirement that is not named in the issues list.
- Do not fabricate, infer, or add any information beyond what the JD contains.
- Output a raw JSON object only — no markdown fences, no prose, no explanation.

Return the full updated requirements list in this exact structure:
{{
  "requirements": [
    {{
      "label": "...",
      "description": "...",
      "priority": "minimum_requirement|preferred_requirement|nice_to_have|ambiguous"
    }}
  ]
}}"""

CORRECT_REQUIREMENTS_USER_TEMPLATE = """\
Job Description:
{job_description}

---
Current Requirements List:
{requirements}

Apply the corrections and return the full updated requirements list. \
Output JSON only."""


# ---------------------------------------------------------------------------
# EXTRACT_IMPLICIT_REQUIREMENTS — infer unstated requirements from JD evidence
# ---------------------------------------------------------------------------

EXTRACT_IMPLICIT_REQUIREMENTS_SYSTEM_PROMPT = """\
You are a job description analyst. Your task is to infer unstated but strongly \
implied requirements from the job description and the already-extracted explicit \
requirements list.

An *implicit* requirement is one that:
- Is NOT explicitly stated in the JD as a requirement or qualification.
- CAN be inferred from direct textual evidence in the JD: the job title, stated \
  responsibilities, domain vocabulary, toolchain mentioned in context, company \
  description, or stakeholder interactions named in the JD.
- Would be considered obvious by any practitioner in the field reading this JD.

Rules:
- Every requirement you produce MUST be grounded in specific textual evidence from \
  the JD. Cite the evidence in the description field.
- Do NOT duplicate any requirement that already appears in the explicit requirements \
  list provided. If a concept overlaps with an explicit requirement, skip it.
- Do NOT produce generic soft skills (e.g., "communication", "teamwork") unless the \
  JD contains specific, non-generic evidence that implies them (e.g., "present roadmap \
  to C-suite weekly" → executive communication).
- Do NOT speculate beyond what the text supports.
- Do NOT use priority "ambiguous". If priority cannot be determined from the evidence, \
  default to "preferred_requirement".
- Set "is_implicit": true on every item in the output.
- If no implicit requirements can be grounded in the JD text, return {"requirements":[]}.
- Output raw JSON only — no markdown fences, no prose, no explanation.

Priority guidance for implicit requirements:
- "minimum_requirement": the role clearly cannot function without this \
  (implied by core stated responsibilities or the job title itself)
- "preferred_requirement": strongly implied but role could proceed with less; \
  use this as the default when uncertain
- "nice_to_have": helpful given context but clearly secondary

Return this exact structure:
{
  "requirements": [
    {
      "label": "Short human-readable label",
      "description": "Full description citing the specific JD text that grounds this inference",
      "priority": "minimum_requirement|preferred_requirement|nice_to_have",
      "is_implicit": true
    }
  ]
}"""

EXTRACT_IMPLICIT_REQUIREMENTS_USER_TEMPLATE = """\
Job Description:
{job_description}

---
Company and Job Context (from earlier extraction — use for context only):
Company Info:
{company_info}

Job Info:
{job_info}

---
Already-extracted explicit requirements (do NOT duplicate these):
{explicit_requirements}

Infer any implicit requirements supported by specific JD text. Output JSON only."""


# ---------------------------------------------------------------------------
# VALIDATE_IMPLICIT_REQUIREMENTS — review implicit requirements for quality
# ---------------------------------------------------------------------------

VALIDATE_IMPLICIT_REQUIREMENTS_SYSTEM_PROMPT = """\
You are a requirements validator. You will receive an extracted implicit \
requirements list and the original job description. Review the extraction for \
five things:

1. COMPLETENESS: Are there clearly implied requirements that are missing?
2. ACCURACY: Does each requirement's description faithfully reflect what the \
   JD implies, without hallucination? Flag descriptions that add information \
   not supportable by the JD text.
3. PRIORITY CORRECTNESS: Is the priority label reasonable given how central \
   the implication is to the role?
4. DUPLICATES: Are any requirements substantially overlapping? \
   Correction must MERGE them — not simply delete one.
5. GROUNDING: Is each requirement traceable to specific text in the JD? \
   Flag any requirement that is generic, speculative, or not evidenced by a \
   concrete signal in the JD. For a grounding issue, set correction to null — \
   the requirement should be removed entirely, not replaced.

Behavioural guidelines — be pragmatic, not pedantic:
- Only flag CLEAR problems. Do not nitpick borderline cases.
- Set all_correct to true when issues is empty.
- Output a raw JSON object only — no markdown fences, no prose.

Issue types:
- "missing"        — a clearly implied requirement is absent from the list
- "inaccurate"     — a description adds or misrepresents information
- "wrong_priority" — the priority label is clearly inconsistent with the evidence
- "duplicate"      — two requirements are substantially overlapping; correction merges them
- "ungrounded"     — requirement is speculative or not traceable to JD text; \
  correction MUST be null (removal, no replacement)

Return this exact structure:
{
  "all_correct": true,
  "issues": []
}

When issues exist:
{
  "all_correct": false,
  "issues": [
    {
      "type": "missing|inaccurate|wrong_priority|duplicate|ungrounded",
      "label": "label of the problematic requirement, or null for missing",
      "duplicate_of": "label of the other requirement (duplicate type only), or null",
      "problem": "clear explanation of the problem",
      "correction": {
        "label": "corrected label",
        "description": "corrected description",
        "priority": "corrected priority",
        "is_implicit": true
      }
    }
  ]
}

For "ungrounded" issues, correction must be null:
{
  "type": "ungrounded",
  "label": "Communication skills",
  "duplicate_of": null,
  "problem": "Generic; no specific JD text implies this",
  "correction": null
}"""

VALIDATE_IMPLICIT_REQUIREMENTS_USER_TEMPLATE = """\
Original Job Description:
{job_description}

---
Extracted Implicit Requirements:
{requirements}

Validate the implicit requirements. Output JSON only."""


# ---------------------------------------------------------------------------
# CORRECT_IMPLICIT_REQUIREMENTS — apply corrections to the implicit list
# ---------------------------------------------------------------------------

# NOTE: {issues} is a pre-formatted human-readable string built in nodes.py,
# not raw JSON.  This avoids brace-escaping conflicts with str.format().
CORRECT_IMPLICIT_REQUIREMENTS_SYSTEM_PROMPT_TEMPLATE = """\
You are a job description analyst. A previous extraction of implicit requirements \
contained some issues. Apply ONLY the listed corrections to the requirements \
list — do not re-extract from scratch.

Issues to fix:
{issues}

Rules for applying corrections:
- "missing": add the correction object as a new requirement to the list.
- "inaccurate" or "wrong_priority": replace the named requirement with the \
  correction object. Keep all other requirements unchanged.
- "duplicate": replace BOTH named requirements with the single merged correction \
  object. Do not keep either of the originals.
- "ungrounded": REMOVE the named requirement entirely. Do not add any replacement.
- Do NOT change any requirement that is not named in the issues list.
- Do not fabricate, infer, or add any information beyond what the JD contains.
- Every requirement in the output must have "is_implicit": true.
- Output a raw JSON object only — no markdown fences, no prose, no explanation.

Return the full updated requirements list in this exact structure:
{{
  "requirements": [
    {{
      "label": "...",
      "description": "...",
      "priority": "minimum_requirement|preferred_requirement|nice_to_have",
      "is_implicit": true
    }}
  ]
}}"""

CORRECT_IMPLICIT_REQUIREMENTS_USER_TEMPLATE = """\
Job Description:
{job_description}

---
Current Implicit Requirements List:
{requirements}

Apply the corrections and return the full updated requirements list. \
Output JSON only."""


# ---------------------------------------------------------------------------
# DEDUPLICATE_REQUIREMENTS — cross-list deduplication of implicit vs explicit
# ---------------------------------------------------------------------------

DEDUPLICATE_REQUIREMENTS_SYSTEM_PROMPT = """\
You are reviewing two lists of job requirements extracted from a job description.

List A contains EXPLICIT requirements — those stated directly in the job description.
List B contains IMPLICIT requirements — those inferred from the job description context.

Identify which implicit requirements (List B) overlap significantly with explicit
requirements (List A) and should be removed.

Remove an implicit requirement if:
- It is fully duplicated by an explicit requirement, OR
- The key parts or most substantive elements of the implicit requirement are already
  captured by an explicit requirement (even if the wording differs)

Keep an implicit requirement if it captures something genuinely distinct that is not
substantially covered by any explicit requirement.

Return a JSON object with a single key "remove" — a list of labels from List B that
should be removed. If nothing should be removed, return {"remove": []}.

Output only valid JSON. No markdown fences, no prose, no explanation."""

DEDUPLICATE_REQUIREMENTS_USER_TEMPLATE = """\
EXPLICIT REQUIREMENTS:
{explicit_requirements}

IMPLICIT REQUIREMENTS:
{implicit_requirements}

Which implicit requirements should be removed because they substantially overlap \
with explicit requirements?"""
