# OpenAI API support and code extensibility
_Exported on 3/1/2026 at 14:28:36 PST from Cursor (2.5.26)_

---

**User**

I would like to add support for OpenAI's API. The current code base should be engineered to allow this extensibility, although some re-engineering may be necessary - for example, some default options are hardcoded for Anthropic.

---

**Cursor**


Exploring the codebase to map the LLM provider architecture and identify where Anthropic-specific options are hardcoded.



Creating a plan to add OpenAI support.


---

**User**

In this plan, how will the user specify which LLM provider to use?

---

**Cursor**


In the plan, the user chooses the LLM provider via the **`LLM_PROVIDER` environment variable**.

- Possible values: `anthropic` or `openai`
- Default: `anthropic` if the variable is not set

**Where to set it**

1. **In `.env`** (recommended, since the project uses `python-dotenv`):

   ```env
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-...
   ```

2. **In the shell**, before running the app:

   ```powershell
   $env:LLM_PROVIDER = "openai"
   $env:OPENAI_API_KEY = "sk-..."
   app-strategist analyze -r resume.txt -j job.txt
   ```

The plan does not add a CLI flag or config file for provider selection; only the environment variable is used. If you prefer a CLI option (e.g. `--provider openai`) or config file support, that can be added to the plan.

---

**User**

Would this plan be easily extensible to use a CLI flag in the future?

---

**Cursor**


Yes. The plan is structured so a CLI flag can be added later without much rework.

1. **Provider creation is centralized** – `get_llm_provider()` is the single factory. Adding an optional parameter is straightforward:

   ```python
   def get_llm_provider(provider: str | None = None) -> LLMProvider:
       name = (provider or os.getenv("LLM_PROVIDER", "anthropic")).lower()
       # ... rest unchanged
   ```

   When `provider` is passed, it takes precedence over the env var.

2. **Consumers already use dependency injection** – `AnalysisService(llm=...)` and the REPL can receive a provider instance. In `analyze`, you’d do:

   ```python
   # Hypothetical future: --provider flag
   llm = get_llm_provider(provider=provider_flag) if provider_flag else get_llm_provider()
   service = AnalysisService(llm=llm)
   ```

   The REPL would need to receive the same `llm` or a provider name instead of constructing its own, which is a small change.

3. **No changes to providers** – `AnthropicProvider` and `OpenAIProvider` stay as-is; only the factory and CLI wiring change.

So extending the plan later to support a `--provider openai` flag is a small addition: an optional `provider` argument on `get_llm_provider()` plus wiring the Typer option through. The current design supports that cleanly.

---

**User**

What if in the future we want to use different models for different tasks within an MCP environment? For example "anthropic" might be used for some tools and "openai" used for others. Does this plan allow for adding this functionality in the future without too much rework?

---

**Cursor**


Yes. The plan’s design leaves room for task-specific providers without major changes.

**What already supports this**

1. **Provider factory with an explicit name** – If `get_llm_provider(provider: str | None = None)` takes a provider name, MCP tools can request specific backends:

   ```python
   # Tool "analyze_resume" handler
   llm = get_llm_provider(provider="anthropic")
   service = AnalysisService(llm=llm)
   
   # Tool "generate_cover_letter" handler  
   llm = get_llm_provider(provider="openai")
   # ...
   ```

2. **Dependency injection** – `AnalysisService`, scorers, and REPL already accept `llm: LLMProvider`. Each tool can construct its own service with a different provider. No changes needed to scorers or other consumers.

3. **API keys** – `get_api_key(provider)` gives per-provider keys. With `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` set, both providers can be used in the same process.

4. **LLMProvider protocol** – All providers expose the same `complete(system_prompt, messages)` interface, so callers stay provider-agnostic.

**What would need to be added**

- **Task → provider mapping** – A way to choose the provider per tool, e.g.:
  - Env: `ANALYZE_PROVIDER=anthropic`, `COVER_LETTER_PROVIDER=openai`
  - MCP server config mapping tool IDs to providers
- **Wiring in tool handlers** – Each MCP tool handler would look up its provider and pass it into `get_llm_provider()` / `AnalysisService(llm=...)`.

**Possible future refinements**

- Caching providers by `(provider, model)` if many tools reuse the same client.
- Allowing per-provider overrides (e.g. model name) in config if needed.

So the plan stays extensible: the central factory and DI pattern mean different tools can use different providers by calling `get_llm_provider(provider=...)` and passing the result in. The extra work would mainly be the configuration layer that maps tools to providers, not changes to the provider implementations or the core architecture.

---

**User**

Please review this plan and highlight any potential concerns or pitfalls with the approach. Particularly pay attention to possible maintainability, extensibility, and security concerns.

---

**Cursor**


Here’s a focused review of the plan:

---

## Critical: Circular Import Risk

**Issue:** `config.py` would need to instantiate `AnthropicProvider` and `OpenAIProvider`. Those providers import `config` for `get_api_key()`. If config imports llm at import time:

```
config (loading) → imports llm → anthropic_provider → imports config (not fully loaded)
```

**Mitigation:** Use lazy imports inside `get_llm_provider()`:

```python
def get_llm_provider(provider: str | None = None):
    name = (provider or os.getenv("LLM_PROVIDER", "anthropic")).lower()
    if name == "anthropic":
        from app_strategist.llm import AnthropicProvider
        return AnthropicProvider(api_key=get_api_key("anthropic"))
    # ...
```

The plan does not mention this. Without lazy imports, circular imports are likely.

---

## Maintainability

| Concern | Details | Recommendation |
|--------|---------|----------------|
| **Provider branching** | `get_llm_provider()` will grow `if/elif` per provider. Fine for 2–3 providers; for 5+ it becomes clumsy. | Mentioned as "Optional: registry pattern." Acceptable for initial scope. |
| **Provider string normalization** | `"OPENAI"`, `"OpenAI"`, `"openai"` should all work. | Normalize with `.lower()` on the provider name before comparisons. |
| **Config scope** | `config.py` will hold both env loading and provider creation. | Reasonable for a small app; consider splitting "LLM setup" into a dedicated module later. |

---

## Extensibility

| Concern | Details | Recommendation |
|--------|---------|----------------|
| **Adding a third provider** | Requires updating `get_api_key()`, `get_llm_provider()`, and adding env vars. | Document these touchpoints for future providers. |
| **Optional dependencies** | `openai` is added as a core dependency. Users who only need Anthropic will still install it. | For a small app this is usually fine; optional deps could be added later for leaner installs. |
| **Provider-specific options** | Model names, `max_tokens`, temperature, etc. are not configurable via env. | Noted as out of scope; worth documenting in the plan as a known limitation. |

---

## Security

| Concern | Details | Status |
|--------|---------|--------|
| **`.env` in version control** | `.env` is in `.gitignore`. | OK |
| **API keys in logs** | Current `logger.debug` uses `self.model`, not the key. | OK; keep ensuring keys are never logged. |
| **API keys in errors** | `get_api_key()` errors mention the var name, not the value. | OK; keep this pattern. |
| **Key stripping** | Plan keeps `key.strip()` to remove accidental whitespace. | OK |
| **Error message clarity** | `ValueError` for unknown provider should not leak internal details. | OK |

**Recommendation:** Add a brief implementation note: never log or include API keys in exceptions or messages.

---

## Other Pitfalls

1. **Timing of fail-fast check**  
   Replacing `get_api_key()` with `get_llm_provider()` is fine, but both resolve at import/startup. Ensure validation happens before heavy work so misconfiguration fails quickly.

2. **Test isolation**  
   Tests inject `MockLLMProvider`, so they do not need env vars and should remain isolated. No change needed.

3. **REPL sharing a provider**  
   REPL and `AnalysisService` both use `get_llm_provider()`, so they use the same provider. That matches expected behavior.

4. **Provider name mapping**  
   The plan uses `"anthropic"` and `"openai"`. Add validation so invalid values raise a clear `ValueError` rather than returning a broken object.

---

## Summary

Most important adjustment: use lazy imports in `get_llm_provider()` to avoid circular imports. The rest of the plan is sound; the main follow-ups are normalizing provider names (e.g. `.lower()`), documenting the explicit fail-fast behavior, and clarifying optional-dependency and env-based model/token options for later.

---

**User**

Let's tweak the plan to include the following.

Documentation changes:
- Document the need to edit get_api_key(), get_llm_provider(), and other touchpoints in order to add additional providers.
- Document changing OpenAI to an optional dependency in the future, or possibly setting the app up so one provider is necessary, but it doesn't matter which one (this way non-Anthropic users can install without having to include Anthropic)
- Document known limitations of configuring model names, max_tokens, etc. via env.

Implementation notes:
- CRITICAL: Use lazy imports in get_llm_provider() to avoid circular imports. 
- Add note that API keys shoud never be logged or included in exceptions or messages, and should never be hardcoded in the code.
- Since get_api_key() and get_llm_provider() both resolve at import/startup, ensure validation happens before heavy work so misconfiguration fails quickly.
- For provider name mapping, add validation so invalid values raise a clear ValueError rather than returning broken objects.

---

**Cursor**


Ask mode prevents edits. Providing the updated plan content for you to apply manually or by switching to Agent mode:



I can't edit files in Ask mode, so here is the content you can add to the plan. Switch to Agent mode if you want me to apply these edits automatically.

---

## 1. Add an "Implementation Notes" section (after Implementation Plan, before File Changes Summary)

```markdown
---

## Implementation Notes

- **CRITICAL — Lazy imports:** `get_llm_provider()` MUST use lazy imports (import providers inside the function body, not at module level) to avoid circular imports: `config` → `llm` → providers → `config`.

- **API key security:** API keys must never be logged, included in exception messages, or hardcoded in source. Use environment variables only; error messages may reference the variable name (e.g. `ANTHROPIC_API_KEY`) but never the value.

- **Fail-fast validation:** `get_api_key()` and `get_llm_provider()` resolve at import or startup. Call validation (e.g. `get_llm_provider()` or a dedicated `validate_llm_config()`) at the start of CLI commands, before heavy work (parsing files, calling APIs), so misconfiguration fails immediately with a clear error.

- **Provider name validation:** Normalize provider names with `.lower()` before comparison. For invalid values, raise a clear `ValueError` (e.g. `"Unknown LLM provider: {name}. Supported: anthropic, openai"`) rather than returning broken or defaulted objects.
```

---

## 2. Update Section 1 (Generalize config) with implementation details

Add these bullets under the `get_llm_provider()` description:

- **CRITICAL: Use lazy imports** — import providers inside the function body (e.g. `from app_strategist.llm import AnthropicProvider`) to avoid circular imports.
- **Validate provider name** — if unknown, raise a clear `ValueError` rather than returning broken objects.

---

## 3. Add a "Documentation for Maintainers" section

Place it after "Optional Enhancements" and before "Message Format Compatibility":

```markdown
---

## Documentation for Maintainers

### Adding a New Provider

To add a third provider (e.g. Gemini), update these touchpoints:

1. **`config.py`**
   - `get_api_key(provider)` — add mapping for the new provider's env var (e.g. `"gemini"` → `GEMINI_API_KEY`)
   - `get_llm_provider()` — add branch for the new provider; update the `ValueError` message listing supported providers

2. **`llm/`**
   - Create new provider module implementing `LLMProvider` protocol
   - Add to `llm/__init__.py` exports

3. **`pyproject.toml`** — add dependency (or optional dependency group if using optional deps)

4. **`.env.example`** and **README** — document new env vars and setup

### Optional Dependencies / Lean Install (Future)

The initial plan adds `openai` as a core dependency. Future options to consider:

- **Optional dependency groups** — e.g. `[project.optional-dependencies]` with `openai` and `anthropic` as separate groups; require at least one. This lets non-Anthropic users install without Anthropic, and vice versa.
- **Runtime check** — on provider selection, verify the corresponding package is installed and raise a clear error if not, pointing the user to `pip install app-strategist[openai]` (or similar).

### Known Limitations

- **Model names and max_tokens** are not configurable via environment variables in the initial implementation. Each provider uses its own hardcoded defaults (e.g. `gpt-4o`, `claude-sonnet-4-6`). Per-provider env vars (e.g. `OPENAI_MODEL`, `ANTHROPIC_MODEL`, `*_MAX_TOKENS`) are deferred to a later enhancement.
```

---

## 4. Update "Optional Enhancements"

Add this bullet:

- Document the optional-dependency and lean-install approaches in the "Documentation for Maintainers" section for future implementation.

---

I can apply these edits directly if you switch to Agent mode.

---

**User**

Please apply these edits directly to the plan.

---

**Cursor**


All requested updates are in place. Summary:

1. **Section 1 (Generalize config)** — Added:
   - **CRITICAL: Use lazy imports** to avoid circular imports
   - Normalize provider name with `.lower()`
   - **Validate provider name** and raise a clear `ValueError` when unknown

2. **Implementation Notes (new section)** — Added:
   - Lazy imports requirement
   - API key security (no logging, no hardcoding, never in exceptions)
   - Fail-fast validation before heavy work
   - Provider name validation with clear `ValueError`

3. **Documentation for Maintainers (new section)** — Added:
   - Steps for adding a new provider: touchpoints (`get_api_key`, `get_llm_provider`, `llm/`, `pyproject.toml`, `.env.example`, README)
   - Optional dependencies / lean-install approach (future work)
   - Limitations: model names and `max_tokens` not configurable via env

4. **Optional Enhancements** — Added a bullet referencing the Documentation for Maintainers for the optional-dependency approach.

---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**


Summary of the changes:

## Summary of Implementation

### 1. Config (`config.py`)
- **`get_api_key(provider: str = "anthropic")`** — Supports `"anthropic"` and `"openai"`; validates provider and raises clear errors for unknown/missing keys.
- **`get_llm_provider(provider: str | None = None)`** — Uses lazy imports to avoid circular imports; reads `LLM_PROVIDER` from env (default `"anthropic"`); validates provider name and raises `ValueError` for unknown values.

### 2. `OpenAIProvider` (`llm/openai_provider.py`)
- Implements `LLMProvider` via `complete(system_prompt, messages)`.
- Uses `openai` client; maps system prompt to `{"role": "system", "content": system_prompt}`.
- Defaults: `gpt-4o`, `max_tokens=4096`.

### 3. AnthropicProvider
- Updated to use `get_api_key("anthropic")`.

### 4. Wiring
- **AnalysisService** — Uses `get_llm_provider()` when `llm` is not provided.
- **main.py** — Uses `get_llm_provider()` for startup validation and REPL.

### 5. Dependencies and docs
- Added `openai>=1.0.0` to `pyproject.toml`.
- Updated `.env.example` with `LLM_PROVIDER`, `OPENAI_API_KEY`.
- Updated README with setup for Anthropic and OpenAI.

### 6. LLM exports
- `OpenAIProvider` exported from `llm/__init__.py`.

All 27 tests pass. Use `LLM_PROVIDER=openai` and `OPENAI_API_KEY` to run with OpenAI.

