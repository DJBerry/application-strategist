"""Configuration and environment loading."""

import os

from dotenv import load_dotenv

# Load .env from current directory or parents (project root)
load_dotenv()

_SUPPORTED_PROVIDERS = ("anthropic", "openai")

_ENV_VAR_BY_PROVIDER = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def get_api_key(provider: str = "anthropic") -> str:
    """Get API key for the given provider from environment. Raises if missing.

    Args:
        provider: One of "anthropic", "openai". Case-insensitive.

    Raises:
        ValueError: If provider is unknown or API key is not set.
    """
    name = provider.strip().lower()
    if name not in _ENV_VAR_BY_PROVIDER:
        raise ValueError(
            f"Unknown LLM provider: {provider}. Supported: {', '.join(_SUPPORTED_PROVIDERS)}"
        )
    env_var = _ENV_VAR_BY_PROVIDER[name]
    key = os.getenv(env_var)
    if not key or not key.strip():
        urls = {
            "anthropic": "https://console.anthropic.com/",
            "openai": "https://platform.openai.com/api-keys",
        }
        raise ValueError(
            f"{env_var} is not set. Create a .env file with {env_var}=your-key "
            f"or set the environment variable. Get your key at {urls.get(name, 'provider docs')}"
        )
    return key.strip()


def get_llm_provider(provider: str | None = None):
    """Return the configured LLM provider. Uses lazy imports to avoid circular imports.

    Args:
        provider: Override the default. One of "anthropic", "openai". If None,
            reads LLM_PROVIDER from environment (default: "anthropic").

    Returns:
        An LLMProvider instance.

    Raises:
        ValueError: If provider is unknown or API key is missing.
    """
    name = (provider or os.getenv("LLM_PROVIDER", "anthropic")).strip().lower()
    if name not in _SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider: {name}. Supported: {', '.join(_SUPPORTED_PROVIDERS)}"
        )

    if name == "anthropic":
        from app_strategist.llm import AnthropicProvider

        return AnthropicProvider(api_key=get_api_key("anthropic"))
    if name == "openai":
        from app_strategist.llm import OpenAIProvider

        return OpenAIProvider(api_key=get_api_key("openai"))

    raise ValueError(
        f"Unknown LLM provider: {name}. Supported: {', '.join(_SUPPORTED_PROVIDERS)}"
    )
