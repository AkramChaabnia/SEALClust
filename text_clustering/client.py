"""
client.py — OpenAI-compatible client factory for OpenRouter (and any compatible provider).

Thin wrapper that loads .env and returns a configured openai.OpenAI client.

The original pipeline scripts call OpenAI() with no arguments and rely on
OPENAI_API_KEY / OPENAI_BASE_URL being set in the environment. This module
sets those variables from .env before any script runs, so the rest of the
code works without modification whether you are using OpenRouter, Groq, or
a direct OpenAI key.

Usage
-----
    from text_clustering.client import make_client

    client = make_client()
    response = client.chat.completions.create(model=..., ...)

Run as a script to smoke-test the configuration:

    python -m text_clustering.client
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# __file__ is text_clustering/client.py — go one level up to reach the project root
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=False)


def make_client() -> OpenAI:
    """Build and return an OpenAI-compatible client.

    Provider detection logic:
      1. If LLM_PROVIDER is explicitly set, use that.
      2. If OPENAI_BASE_URL contains 'openrouter', assume openrouter.
      3. Otherwise, assume direct openai.

    For OpenRouter, the optional HTTP-Referer and X-Title headers are
    injected as required by OpenRouter's attribution guidelines.
    """
    base_url = os.getenv("OPENAI_BASE_URL") or None
    explicit_provider = os.getenv("LLM_PROVIDER", "").lower()

    # Auto-detect provider from base URL when not explicitly set
    if explicit_provider:
        provider = explicit_provider
    elif base_url and "openrouter" in base_url:
        provider = "openrouter"
    else:
        provider = "openai"

    extra_headers: dict[str, str] = {}
    if provider == "openrouter":
        if site_url := os.getenv("OR_SITE_URL", ""):
            extra_headers["HTTP-Referer"] = site_url
        if app_name := os.getenv("OR_APP_NAME", "SEALClust"):
            extra_headers["X-Title"] = app_name

    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=base_url,
        default_headers=extra_headers or None,
    )


# Resolved at import time so other modules can do:
#   from text_clustering.client import MODEL
# instead of reading the env var themselves.
MODEL: str = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-001")
TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))


if __name__ == "__main__":
    from text_clustering.config import USE_RESPONSES_API

    base_url = os.getenv("OPENAI_BASE_URL", "(openai default)")
    provider_env = os.getenv("LLM_PROVIDER", "")
    detected = (
        provider_env
        if provider_env
        else ("openrouter" if "openrouter" in str(base_url) else "openai")
    )
    print(f"Provider  : {detected}")
    print(f"Base URL  : {base_url}")
    print(f"Model     : {MODEL}")
    print(f"API mode  : {'Responses API' if USE_RESPONSES_API else 'Chat Completions'}")
    print(f"Temp      : {TEMPERATURE}  |  Max tokens: {MAX_TOKENS}")
    print("\nSending smoke-test request...")
    client = make_client()

    if USE_RESPONSES_API:
        resp = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
        print(f"Response  : {resp.output_text}")
    else:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=64,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
        print(f"Response  : {resp.choices[0].message.content}")
    print("Smoke test passed.")
