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

EXTRACT_REQUIREMENTS_SYSTEM_PROMPT = """\
You are a job description analyst. Extract every requirement mentioned in the \
job description provided by the user.

Rules:
- Only extract requirements explicitly stated in the job description. Do not \
  infer, assume, or add requirements that are not directly present in the text.
- If the same skill or qualification appears in multiple places in the JD, \
  produce ONE requirement that captures the fullest description — do not \
  create separate duplicates.
- If a JD bullet bundles multiple distinct qualifications together \
  (e.g., "experience with Python and deploying ML models to production"), split \
  them into separate requirements unless they are genuinely inseparable.
- For each requirement's description: rephrase for clarity and consistency \
  rather than copying JD language verbatim, but do NOT add qualifications, \
  thresholds, or conditions that are not present in the source text. Capture \
  all relevant nuance, conditions, and flexibility stated in the JD \
  (e.g., if education can substitute for years of experience, include that).
- Determine priority from the JD's own language cues:
    "must", "required", "minimum", "mandatory"  → "minimum_requirement"
    "preferred", "desired", "ideal", "important" → "preferred_requirement"
    "bonus", "plus", "nice to have", "a plus"   → "nice_to_have"
    unclear or contradictory language            → "ambiguous"
- Do not fabricate, infer, or guess any values.
- Output a raw JSON object only — no markdown fences, no prose, no explanation.

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
