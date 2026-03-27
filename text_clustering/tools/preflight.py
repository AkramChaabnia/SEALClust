"""
preflight.py
------------
Pre-flight check before running the pipeline. Validates the environment,
configuration, API connectivity, and model behaviour in ~30 seconds.

Checks
------
  C1  .env file        — exists and is not the template
  C2  API key          — OPENAI_API_KEY is set and non-empty
  C3  Config echo      — print resolved config values
  C4  Reachability     — model responds without error
  C5  JSON output      — label-gen prompt returns parseable JSON list
  C6  Merge output     — merge prompt returns parseable JSON (dict or list)
  C7  Merge count      — merged list is smaller than input (consolidation check)
  C8  Classification   — classify prompt returns a valid label name
  C9  Token budget     — no reasoning tokens (thinking models excluded)
  C10 Seed labels      — runs/chosen_labels.json exists and has entries

Usage
-----
    python -m text_clustering.tools.preflight
    tc-preflight            # if installed via `uv pip install -e .`
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: load .env before importing anything from text_clustering
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / ".env", override=False)

from text_clustering.client import make_client  # noqa: E402
from text_clustering.config import (  # noqa: E402
    FORCE_JSON_MODE,
    MAX_TOKENS,
    MODEL,
    REQUEST_DELAY,
    TEMPERATURE,
)
from text_clustering.prompts import (  # noqa: E402
    prompt_construct_classify,
    prompt_construct_generate_label,
    prompt_construct_merge_label,
)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

OK = f"{_GREEN}✓ OK{_RESET}"
FAIL = f"{_RED}✗ FAIL{_RESET}"
WARN = f"{_YELLOW}~ WARN{_RESET}"
SKIP = f"{_YELLOW}· SKIP{_RESET}"


def ok(msg: str) -> None:
    print(f"  {OK}   {msg}")


def fail(msg: str) -> None:
    print(f"  {FAIL}  {msg}")


def warn(msg: str) -> None:
    print(f"  {WARN}  {msg}")


def skip(msg: str) -> None:
    print(f"  {SKIP}  {msg}")


# ---------------------------------------------------------------------------
# Fixtures — same texts used in probe_models.py (massive_intent/small)
# ---------------------------------------------------------------------------
_SENTENCES = [
    "wake me up at five am this week",
    "quiet",
    "pink is all we need",
    "olly turn the lights off in the bedroom",
    "what's the time in australia",
    "cancel my seven am alarm",
    "i like sinatra songs",
    "play some jazz for me",
    "set a reminder for my dentist appointment",
    "what will the weather be like tomorrow",
    "add milk to my shopping list",
    "call mum on whatsapp",
]

_SEED_LABELS = [
    "alarm set", "audio volume mute", "datetime query",
    "iot hue lightoff", "music likeness", "general greet",
    "shopping list add", "weather query", "reminder set",
    "play music", "calling contact",
]

# Realistic merged label set — what classification runs against
_CLASSIFY_LABELS = [
    "alarm set", "alarm remove", "alarm query",
    "audio volume mute", "audio volume up", "audio volume down",
    "iot hue lightoff", "iot hue lighton", "iot cleaning",
    "general greet", "datetime query", "datetime convert",
    "music likeness", "play music", "reminder set",
    "shopping list add", "weather query", "calling contact",
]

# A deliberately large label list to check real consolidation ability
# (mirrors what Step 1 produces for massive_scenario before merge)
_MERGE_LABELS = [
    "alarm", "alarm set", "alarm remove", "alarm query", "alarms",
    "audio volume", "audio volume mute", "audio volume up", "audio volume down",
    "volume control", "mute", "sound control", "media control",
    "iot", "iot hue lightoff", "iot hue lighton", "smart home",
    "home automation", "lighting control", "smart plug",
    "general", "general greet", "general quirky", "general knowledge",
    "datetime", "datetime query", "datetime convert", "time", "date",
    "time zone", "time conversion",
    "music", "music likeness", "play music", "music playback",
    "podcast", "radio", "audiobook",
    "reminder", "reminder set", "reminder query",
    "shopping", "shopping list add", "shopping list query",
    "weather", "weather query", "weather forecast",
    "calling", "calling contact", "call contact",
    "news", "news query", "breaking news",
    "calendar", "calendar query", "calendar set",
    "email", "email send", "email query", "email check",
    "transport", "transport query", "train", "taxi",
    "cooking", "recipe", "cooking instructions",
    "lists", "list", "to-do list", "task management", "task",
    "takeaway", "food order", "food delivery", "restaurant locate",
    "social", "social media", "social post",
    "recommendation", "local events", "local recommendation",
]

_CLASSIFY_SENTENCE = "remind me to take my medicine at 8pm"
_EXPECTED_LABEL = "reminder set"


# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------
def _call(client, prompt: str, max_tokens: int = 512) -> dict:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to output JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        choice = resp.choices[0]
        usage = resp.usage
        rt = 0
        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            rt = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        return {
            "content": choice.message.content or "",
            "finish": choice.finish_reason,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "reasoning_tokens": rt,
            "error": None,
        }
    except Exception as exc:
        return {"content": "", "finish": "error", "prompt_tokens": 0,
                "completion_tokens": 0, "reasoning_tokens": 0, "error": str(exc)}


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    return m.group(1).strip() if m else text


def _parse(raw: str):
    if not raw:
        return None
    clean = _strip_fences(raw)
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return eval(clean)  # noqa: S307 — mirrors pipeline
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main preflight
# ---------------------------------------------------------------------------
def run_preflight() -> int:
    """Run all checks. Returns exit code (0 = all OK, 1 = at least one FAIL)."""
    failures = 0
    env_path = _ROOT / ".env"

    print(f"\n{_BOLD}Pre-flight check — text-clustering-llm{_RESET}")
    print(f"{'─' * 52}")

    # ------------------------------------------------------------------
    # C1 — .env file
    # ------------------------------------------------------------------
    print("\n  C1  .env file")
    if not env_path.exists():
        fail(f".env not found at {env_path}  (copy .env.example and fill in your key)")
        failures += 1
    else:
        content = env_path.read_text()
        if "your-openrouter-key-here" in content:
            fail(".env still contains the placeholder key — edit OPENAI_API_KEY")
            failures += 1
        elif not os.getenv("OPENAI_API_KEY"):
            fail("OPENAI_API_KEY is not set in the environment after loading .env")
            failures += 1
        else:
            key = os.environ["OPENAI_API_KEY"]
            ok(f".env loaded — key ends …{key[-6:]}")

    # ------------------------------------------------------------------
    # C2 — Base URL
    # ------------------------------------------------------------------
    print("\n  C2  API endpoint")
    base_url = os.getenv("OPENAI_BASE_URL", "(openai default)")
    ok(f"OPENAI_BASE_URL = {base_url}")

    # ------------------------------------------------------------------
    # C3 — Config echo
    # ------------------------------------------------------------------
    print("\n  C3  Resolved config")
    ok(f"MODEL           = {MODEL}")
    ok(f"MAX_TOKENS      = {MAX_TOKENS}")
    ok(f"TEMPERATURE     = {TEMPERATURE}")
    ok(f"REQUEST_DELAY   = {REQUEST_DELAY}s")
    ok(f"FORCE_JSON_MODE = {FORCE_JSON_MODE}")
    if MAX_TOKENS < 2048:
        warn(
            f"MAX_TOKENS={MAX_TOKENS} is low — the merge step needs ≥2300 tokens "
            "for large label sets"
        )
    if REQUEST_DELAY < 1:
        warn(f"REQUEST_DELAY={REQUEST_DELAY}s is very low — may trigger 429 rate limits")

    # Bail early if no API key — remaining checks all need it
    if failures > 0:
        print(
            f"\n{_RED}{_BOLD}Aborting — fix the above errors before running the pipeline."
            f"{_RESET}\n"
        )
        return failures

    client = make_client()

    # ------------------------------------------------------------------
    # C4 — Reachability
    # ------------------------------------------------------------------
    print("\n  C4  API reachability")
    r = _call(client, "Reply with exactly: OK", max_tokens=10)
    if r["error"]:
        fail(f"API error: {r['error'][:120]}")
        failures += 1
        print(f"\n{_RED}{_BOLD}Cannot reach the API — stopping.{_RESET}\n")
        return failures
    elif not r["content"]:
        fail(f"Empty response (finish={r['finish']})")
        failures += 1
        print(f"\n{_RED}{_BOLD}Cannot reach the API — stopping.{_RESET}\n")
        return failures
    else:
        ok(f"Model responded ({r['completion_tokens']} tokens, finish={r['finish']})")
    time.sleep(REQUEST_DELAY)

    # ------------------------------------------------------------------
    # C5 — Label generation
    # ------------------------------------------------------------------
    print("\n  C5  Label generation (Step 1 prompt)")
    prompt = prompt_construct_generate_label(_SENTENCES, _SEED_LABELS)
    r = _call(client, prompt, max_tokens=512)
    gen_labels: list[str] = []
    if r["error"]:
        fail(f"API error: {r['error'][:120]}")
        failures += 1
    elif not r["content"]:
        fail(f"Empty response (finish={r['finish']})")
        failures += 1
    else:
        parsed = _parse(r["content"])
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    gen_labels = v
                    break
        if gen_labels:
            ok(f"Got {len(gen_labels)} labels: {gen_labels}")
        else:
            fail(f"No string list in response. Raw: {repr(r['content'][:120])}")
            failures += 1
    time.sleep(REQUEST_DELAY)

    # ------------------------------------------------------------------
    # C6 + C7 — Merge (small probe, then count check)
    # ------------------------------------------------------------------
    print("\n  C6  Merge output (Step 1 merge prompt)")
    merge_input = list(dict.fromkeys(_MERGE_LABELS))  # deduplicate, preserve order
    n_input = len(merge_input)
    # No target_k — paper behaviour: let the model consolidate naturally.
    # target_k is only for weaker models that under-consolidate.
    prompt = prompt_construct_merge_label(merge_input)
    r = _call(client, prompt, max_tokens=4096)
    merged: list[str] = []
    if r["error"]:
        fail(f"API error: {r['error'][:120]}")
        failures += 1
    elif not r["content"]:
        fail(f"Empty response (finish={r['finish']})")
        failures += 1
    else:
        parsed = _parse(r["content"])
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            merged = parsed
        elif isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    merged.extend(v)
        if merged:
            ok(f"Merge response is valid JSON — {n_input} → {len(merged)} labels")
        else:
            fail(f"Cannot parse merge response. Raw: {repr(r['content'][:120])}")
            failures += 1

    print("\n  C7  Merge consolidation quality")
    if not merged:
        skip("Skipped — C6 failed")
    elif len(merged) >= n_input:
        fail(
            f"Merge produced {len(merged)} labels from {n_input} input — no consolidation. "
            "This model cannot perform the merge step."
        )
        failures += 1
    elif len(merged) > 30:
        warn(
            f"Merge reduced {n_input} → {len(merged)} labels (target ≈18). "
            "Acceptable, but may still produce fragmented clusters. "
            "Check n_pred_clusters after Step 1."
        )
    else:
        ok(f"Merge reduced {n_input} → {len(merged)} labels (target ≈18) ✓")
    time.sleep(REQUEST_DELAY)

    # ------------------------------------------------------------------
    # C8 — Classification
    # ------------------------------------------------------------------
    print("\n  C8  Classification (Step 2 prompt)")
    prompt = prompt_construct_classify(_CLASSIFY_LABELS, _CLASSIFY_SENTENCE)
    r = _call(client, prompt, max_tokens=64)
    if r["error"]:
        fail(f"API error: {r['error'][:120]}")
        failures += 1
    elif not r["content"]:
        fail(f"Empty response (finish={r['finish']})")
        failures += 1
    else:
        parsed = _parse(r["content"])
        label_name = None
        if isinstance(parsed, dict):
            label_name = next(iter(parsed.values()), None)
        if label_name and isinstance(label_name, str):
            match = label_name.strip().lower() == _EXPECTED_LABEL.lower()
            msg = f'Got label: "{label_name}"'
            if match:
                ok(f"{msg} ✓")
            else:
                warn(f'{msg} (expected "{_EXPECTED_LABEL}" — different label returned)')
        else:
            fail(f"No label_name in response. Raw: {repr(r['content'][:120])}")
            failures += 1
    time.sleep(REQUEST_DELAY)

    # ------------------------------------------------------------------
    # C9 — Token budget (no reasoning tokens)
    # ------------------------------------------------------------------
    print("\n  C9  Token budget (no reasoning tokens)")
    r = _call(client, "Reply with exactly: OK", max_tokens=10)
    if r["reasoning_tokens"] > 0:
        fail(
            f"Model returned {r['reasoning_tokens']} reasoning tokens — "
            "this is a thinking/reasoning model and is not suitable for this pipeline."
        )
        failures += 1
    else:
        ok("No reasoning tokens detected ✓")

    # ------------------------------------------------------------------
    # C10 — Seed labels
    # ------------------------------------------------------------------
    print("\n  C10 Seed labels (runs/chosen_labels.json)")
    seed_path = _ROOT / "runs" / "chosen_labels.json"
    if not seed_path.exists():
        warn(
            "runs/chosen_labels.json not found — run `tc-seed-labels` before Step 1. "
            "(Safe to ignore if you haven't done Step 0 yet.)"
        )
    else:
        try:
            seeds = json.loads(seed_path.read_text())
            total = sum(len(v) for v in seeds.values()) if isinstance(seeds, dict) else len(seeds)
            ok(f"Found seed labels ({total} entries across {len(seeds)} datasets)")
        except Exception as exc:
            warn(f"Could not parse chosen_labels.json: {exc}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'─' * 52}")
    if failures == 0:
        print(f"  {_GREEN}{_BOLD}All checks passed — ready to run the pipeline.{_RESET}")
        print("\n  Next steps:")
        print("    tc-seed-labels                          # if not done yet")
        print("    tc-label-gen --data massive_scenario")
        print("    tc-classify  --data massive_scenario --run_dir runs/<ts>/")
        print("    tc-evaluate  --data massive_scenario --run_dir runs/<ts>/")
    else:
        print(f"  {_RED}{_BOLD}{failures} check(s) failed — fix the above before running.{_RESET}")
    print()
    return failures


def main() -> None:
    sys.exit(run_preflight())


if __name__ == "__main__":
    main()
