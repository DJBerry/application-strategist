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
