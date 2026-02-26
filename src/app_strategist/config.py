"""Configuration and environment loading."""

import os

from dotenv import load_dotenv

# Load .env from current directory or parents (project root)
load_dotenv()


def get_api_key() -> str:
    """Get Anthropic API key from environment. Raises if missing."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key or not key.strip():
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Create a .env file with ANTHROPIC_API_KEY=your-key "
            "or set the environment variable. Get your key at https://console.anthropic.com/"
        )
    return key.strip()
