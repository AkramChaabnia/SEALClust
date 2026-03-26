"""
llm.py — LLM client initialisation and chat helper shared by all pipeline steps.

Functions
---------
ini_client()
    Return a configured OpenAI-compatible client via openrouter_adapter.

_strip_fenced_json(text)
    Strip markdown code fences that some models wrap their JSON output in.

chat(prompt, client)
    Send a single-turn chat prompt and return the raw text response.
    Retries up to 5 times on HTTP 429 with linear back-off.

get_token_usage() / reset_token_usage()
    Retrieve / reset cumulative input/output token counters.
"""

import logging
import re
import threading
import time

from text_clustering.config import (
    FORCE_JSON_MODE,
    MAX_TOKENS,
    MODEL,
    REASONING_EFFORT,
    REQUEST_DELAY,
    TEMPERATURE,
    USE_RESPONSES_API,
)

logger = logging.getLogger(__name__)

# ── Token-usage tracking ──────────────────────────────────────────────────
_token_lock = threading.Lock()
_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0}


def get_token_usage() -> dict:
    """Return a snapshot of cumulative token usage since last reset."""
    with _token_lock:
        return dict(_token_usage)


def reset_token_usage() -> None:
    """Reset all token counters to zero."""
    with _token_lock:
        for k in _token_usage:
            _token_usage[k] = 0


def _record_usage(completion) -> None:
    """Extract and accumulate token counts from an API completion response.

    Handles both Chat Completions (prompt_tokens/completion_tokens) and
    Responses API (input_tokens/output_tokens) usage formats.
    """
    usage = getattr(completion, "usage", None)
    if usage is None:
        return
    # Responses API uses input_tokens/output_tokens directly;
    # Chat Completions uses prompt_tokens/completion_tokens.
    inp = getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0
    tot = getattr(usage, "total_tokens", 0) or (inp + out)
    with _token_lock:
        _token_usage["input_tokens"] += inp
        _token_usage["output_tokens"] += out
        _token_usage["total_tokens"] += tot
        _token_usage["api_calls"] += 1


def ini_client():
    from text_clustering.client import make_client

    return make_client()


def _strip_fenced_json(text: str) -> str:
    """Remove markdown code fences if the model wrapped its JSON output in them."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def chat(prompt, client, max_tokens: int | None = None):
    if USE_RESPONSES_API:
        return _chat_responses(prompt, client, max_tokens)
    return _chat_completions(prompt, client, max_tokens)


def _chat_responses(prompt, client, max_tokens: int | None = None):
    """Call the OpenAI Responses API (gpt-5.x / o-series models)."""
    kwargs: dict = dict(
        model=MODEL,
        input=[
            {
                "role": "developer",
                "content": "You are a helpful assistant designed to output JSON.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    # Reasoning models use reasoning.effort instead of temperature
    kwargs["reasoning"] = {"effort": REASONING_EFFORT}
    if max_tokens is not None or MAX_TOKENS:
        kwargs["max_output_tokens"] = max_tokens if max_tokens is not None else MAX_TOKENS

    if FORCE_JSON_MODE:
        kwargs["text"] = {"format": {"type": "json_object"}}

    for attempt in range(5):
        try:
            response = client.responses.create(**kwargs)
            _record_usage(response)
            raw = response.output_text
            if REQUEST_DELAY > 0:
                time.sleep(REQUEST_DELAY)
            return _strip_fenced_json(raw)
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                wait = 20 * (attempt + 1)
                logger.warning("rate limit — attempt %d/5, retrying in %ds...", attempt + 1, wait)
                time.sleep(wait)
            else:
                logger.error("api error: %s", e)
                return None
    return None


def _chat_completions(prompt, client, max_tokens: int | None = None):
    """Call the Chat Completions API (gpt-4o-mini, OpenRouter models, etc.)."""
    kwargs = dict(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else MAX_TOKENS,
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    if FORCE_JSON_MODE:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(5):
        try:
            completion = client.chat.completions.create(**kwargs)
            _record_usage(completion)
            raw = completion.choices[0].message.content
            if REQUEST_DELAY > 0:
                time.sleep(REQUEST_DELAY)
            return _strip_fenced_json(raw)
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                wait = 20 * (attempt + 1)
                logger.warning("rate limit — attempt %d/5, retrying in %ds...", attempt + 1, wait)
                time.sleep(wait)
            else:
                logger.error("api error: %s", e)
                return None
    return None
