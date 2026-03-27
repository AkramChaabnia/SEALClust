"""
probe_models.py
---------------
Quick eligibility check for a candidate LLM before committing it to a full
pipeline run. Sends the same prompts the pipeline actually uses, then reports
whether the model produces valid JSON, stays consistent, and doesn't waste the
token budget on internal chain-of-thought.

Usage
-----
    python -m text_clustering.tools.probe_models --model <openrouter-model-id>
    python -m text_clustering.tools.probe_models --model google/gemini-2.0-flash-001 --verbose

Tests
-----
    T1  Reachability    — model responds without error
    T2  Label gen       — pipeline label-generation prompt → {"labels": [...]}
    T3  Label merge     — pipeline merge prompt → any JSON list
    T4  Classification  — pipeline classification prompt → valid label from the list
    T5  Consistency     — T4 repeated 3× → same answer each time
    T6  Token budget    — zero reasoning tokens (thinking models are not suitable)

Each test prints PASS / WARN / FAIL with a one-line explanation.
A final score and verdict (RECOMMENDED / USABLE / SKIP) are printed at the end.

NOTE: Passing T1–T6 is necessary but NOT sufficient. A model must also pass the
merge-at-scale test (≥100 proposed labels → ≈ true class count) before being used
for a full pipeline run. Use the --merge-test flag or run manually:
    python -m text_clustering.tools.probe_models --model <id> --verbose
and inspect the T3 output count. Models that pass probe but fail merge at scale:
    - arcee-ai/trinity-large-preview:free  (stalls at 144/158 labels)
    - openai/gpt-4o-mini                   (167 → 105, poor consolidation)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from text_clustering.client import make_client

load_dotenv(Path(__file__).parent.parent / ".env", override=False)

# ---------------------------------------------------------------------------
# Test fixtures
# Real texts sampled from massive_intent/small.jsonl (first 15 records).
# Seed labels are those that select_part_labels.py picked for massive_intent
# (11 out of 59, the 20% given to the LLM as a starting point).
# ---------------------------------------------------------------------------

SAMPLE_SENTENCES = [
    "wake me up at five am this week",
    "quiet",
    "pink is all we need",
    "olly turn the lights off in the bedroom",
    "hoover the hallway",
    "what's up olly",
    "what's the time in australia",
    "cancel my seven am alarm",
    "tell me about my alarms",
    "i like sinatra songs",
    "play some jazz for me",
    "set a reminder for my dentist appointment",
    "what will the weather be like tomorrow",
    "add milk to my shopping list",
    "call mum on whatsapp",
]

SEED_LABELS = [
    "alarm set", "audio volume mute", "datetime query",
    "iot hue lightoff", "music likeness", "general greet",
    "shopping list add", "weather query", "reminder set",
    "play music", "calling contact",
]

# A realistic merged label set — what classification runs against.
CANDIDATE_LABELS = [
    "alarm set", "alarm remove", "alarm query",
    "audio volume mute", "audio volume up", "audio volume down",
    "iot hue lightoff", "iot hue lighton", "iot hue lightchange", "iot cleaning",
    "general greet", "general quirky",
    "datetime query", "datetime convert",
    "music likeness", "play music",
    "reminder set", "shopping list add",
    "weather query", "calling contact",
    "news query", "calendar query", "transport query", "cooking recipe",
    "email send", "email query", "social post",
    "currency convert", "translate", "definition",
]

TEST_SENTENCE = "remind me to take my medicine at 8pm"
EXPECTED_LABEL = "reminder set"


# ---------------------------------------------------------------------------
# Prompt builders — identical to the functions in label_generation.py and
# given_label_classification.py so we test exactly what the pipeline sends.
# ---------------------------------------------------------------------------

def build_label_gen_prompt(sentences: list[str], given_labels: list[str]) -> str:
    json_example = {"labels": ["label name", "label name"]}
    return (
        "Given the labels, under a text classicifation scenario, can all these text "
        "match the label given? If the sentence does not match any of the label, "
        "please generate a meaningful new label name.\n "
        f"           Labels: {given_labels}\n "
        f"           Sentences: {sentences} \n "
        "           You should NOT return meaningless label names such as "
        "'new_label_1' or 'unknown_topic_1' and only return the new label names, "
        f"please return in json format like: {json_example}"
    )


def build_merge_prompt(label_list: list[str]) -> str:
    json_example = {"merged_labels": ["label name", "label name"]}
    prompt = (
        "Please analyze the provided list of labels to identify entries that are "
        "similar or duplicate, considering synonyms, variations in phrasing, and "
        "closely related terms that essentially refer to the same concept. "
        "Your task is to merge these similar entries into a single representative "
        "label for each unique concept identified. The goal is to simplify the list "
        "by reducing redundancies without organizing it into subcategories or "
        "altering its fundamental structure. \n"
    )
    prompt += f"Here is the list of labels for analysis and simplification::{label_list}.\n"
    prompt += (
        "Produce the final, simplified list in a flat, JSON-formatted structure "
        f"without any substructures or hierarchical categorization like: {json_example}"
    )
    return prompt


def build_classify_prompt(label_list: list[str], sentence: str) -> str:
    json_example = {"label_name": "label"}
    prompt = (
        "Given the label list and the sentence, please categorize the sentence "
        "into one of the labels.\n"
    )
    prompt += f"Label list: {label_list}.\n"
    prompt += f"Sentence:{sentence}.\n"
    prompt += (
        "You should only return the label name, please return in json format "
        f"like: {json_example}"
    )
    return prompt


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def call_api(client: OpenAI, model: str, prompt: str, max_tokens: int = 512) -> dict:
    """Send one chat completion request and return a normalized result dict."""
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
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
        reasoning_tokens = 0
        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning_tokens = (
                getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
            )
        return {
            "content": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "error": None,
        }
    except Exception as exc:
        return {
            "content": "",
            "finish_reason": "error",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "error": str(exc),
        }


def strip_fences(text: str) -> str:
    """Remove markdown code fences if present (some models wrap JSON in them)."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def parse_json(raw: str) -> dict | list | None:
    """
    Try to parse JSON from a model response.
    Order: strip fences → json.loads on first {...} block → eval fallback.
    The eval fallback mirrors what the original pipeline scripts do.
    """
    if not raw:
        return None
    clean = strip_fences(raw)
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return eval(clean)  # noqa: S307 — intentional, mirrors original pipeline
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

_ICONS = {PASS: "✓", FAIL: "✗", WARN: "~"}


def fmt(status: str, msg: str) -> str:
    return f"  [{_ICONS[status]} {status}] {msg}"


def test_reachability(client: OpenAI, model: str, verbose: bool) -> tuple[str, str]:
    """T1 — check that the model responds at all."""
    result = call_api(client, model, "Reply with exactly: OK", max_tokens=10)
    if result["error"]:
        return FAIL, f"API error: {result['error'][:120]}"
    if not result["content"]:
        return FAIL, f"Empty response (finish_reason={result['finish_reason']})"
    if verbose:
        print(f"       raw: {repr(result['content'][:80])}")
    ct, fr = result["completion_tokens"], result["finish_reason"]
    return PASS, f"{ct} tokens, finish={fr}"


def test_label_generation(
    client: OpenAI, model: str, verbose: bool
) -> tuple[str, str, list[str] | None]:
    """T2 — label-generation prompt must return parseable JSON with a list of strings."""
    prompt = build_label_gen_prompt(SAMPLE_SENTENCES, SEED_LABELS)
    result = call_api(client, model, prompt, max_tokens=512)

    if result["error"]:
        return FAIL, f"API error: {result['error'][:120]}", None
    if not result["content"]:
        hint = "reasoning model?" if result["reasoning_tokens"] > 0 else "empty content"
        return FAIL, f"No content ({hint}, finish={result['finish_reason']})", None
    if verbose:
        print(f"       raw: {repr(result['content'][:200])}")

    parsed = parse_json(result["content"])
    if parsed is None:
        return FAIL, f"Not valid JSON. Raw: {repr(result['content'][:120])}", None

    labels = None
    if isinstance(parsed, dict):
        for val in parsed.values():
            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                labels = val
                break
    if labels is None:
        return FAIL, f"No string list found in response. Parsed: {parsed}", None

    placeholders = [
        lbl for lbl in labels
        if "new_label" in lbl.lower() or "unknown_topic" in lbl.lower()
    ]
    if placeholders:
        return WARN, f"Placeholder labels in output: {placeholders}", labels

    return PASS, f"Got {len(labels)} labels: {labels}", labels


def test_label_merge(
    client: OpenAI,
    model: str,
    labels_from_t2: list[str] | None,
    verbose: bool,
) -> tuple[str, str]:
    """T3 — merge prompt must return parseable JSON containing a list."""
    input_labels = list(
        dict.fromkeys((labels_from_t2 or []) + SEED_LABELS + CANDIDATE_LABELS[:10])
    )
    prompt = build_merge_prompt(input_labels)
    result = call_api(client, model, prompt, max_tokens=512)

    if result["error"]:
        return FAIL, f"API error: {result['error'][:120]}"
    if not result["content"]:
        return FAIL, f"Empty response (finish={result['finish_reason']})"
    if verbose:
        print(f"       raw: {repr(result['content'][:200])}")

    parsed = parse_json(result["content"])
    if parsed is None:
        return FAIL, f"Not valid JSON. Raw: {repr(result['content'][:120])}"

    if isinstance(parsed, dict):
        for key, val in parsed.items():
            if isinstance(val, list):
                return PASS, f"Key '{key}' → {len(val)} merged labels"
    if isinstance(parsed, list):
        return PASS, f"Flat list of {len(parsed)} labels"

    return WARN, f"Unexpected structure: {type(parsed).__name__}"


def test_classification(
    client: OpenAI, model: str, verbose: bool
) -> tuple[str, str, str | None]:
    """T4 — classification prompt must return a label that exists in the candidate list."""
    prompt = build_classify_prompt(CANDIDATE_LABELS, TEST_SENTENCE)
    result = call_api(client, model, prompt, max_tokens=128)

    if result["error"]:
        return FAIL, f"API error: {result['error'][:120]}", None
    if not result["content"]:
        return FAIL, f"Empty response (finish={result['finish_reason']})", None
    if verbose:
        print(f"       raw: {repr(result['content'][:200])}")

    parsed = parse_json(result["content"])
    predicted = None

    if isinstance(parsed, dict):
        for val in parsed.values():
            if isinstance(val, str):
                predicted = val
                break
    elif isinstance(parsed, str):
        predicted = parsed

    # fallback: scan raw text for any known label
    if not predicted or predicted not in CANDIDATE_LABELS:
        for label in CANDIDATE_LABELS:
            if label in (result["content"] or ""):
                predicted = label
                break

    if not predicted:
        return FAIL, f"Could not extract a label. Raw: {repr(result['content'][:120])}", None
    if predicted not in CANDIDATE_LABELS:
        return WARN, f"'{predicted}' is not in the candidate list (hallucinated?)", predicted

    correct = predicted == EXPECTED_LABEL
    quality = PASS if correct else WARN
    suffix = "correct" if correct else f"expected '{EXPECTED_LABEL}'"
    return quality, f"→ '{predicted}' ({suffix})", predicted


def test_consistency(client: OpenAI, model: str, verbose: bool) -> tuple[str, str]:
    """T5 — classification repeated 3× must return the same answer each time."""
    prompt = build_classify_prompt(CANDIDATE_LABELS, TEST_SENTENCE)
    answers = []
    for i in range(3):
        result = call_api(client, model, prompt, max_tokens=128)
        parsed = parse_json(result["content"] or "")
        predicted = None
        if isinstance(parsed, dict):
            for val in parsed.values():
                if isinstance(val, str):
                    predicted = val
                    break
        if not predicted:
            for label in CANDIDATE_LABELS:
                if label in (result["content"] or ""):
                    predicted = label
                    break
        answers.append(predicted or "?")
        if verbose:
            print(f"       run {i + 1}: {repr(predicted)}")
        if i < 2:
            time.sleep(2)

    unique = list(dict.fromkeys(answers))
    if len(unique) == 1:
        return PASS, f"All 3 runs agree on '{unique[0]}'"
    if len(unique) == 2:
        return WARN, f"2 different answers across 3 runs: {answers}"
    return FAIL, f"Fully inconsistent across 3 runs: {answers}"


def test_token_efficiency(client: OpenAI, model: str, verbose: bool) -> tuple[str, str]:
    """T6 — the model must not produce reasoning tokens (thinking models are disqualified)."""
    prompt = build_classify_prompt(CANDIDATE_LABELS[:15], TEST_SENTENCE)
    result = call_api(client, model, prompt, max_tokens=256)

    if result["error"]:
        return WARN, f"Could not check (API error): {result['error'][:80]}"

    rt, ct = result["reasoning_tokens"], result["completion_tokens"]
    if rt > 50:
        ratio = rt / max(ct, 1)
        return FAIL, f"Thinking model: {rt} reasoning tokens vs {ct} output tokens ({ratio:.1f}×)"
    if rt > 0:
        return WARN, f"{rt} reasoning tokens (marginal — may need higher max_tokens)"
    return PASS, f"0 reasoning tokens, {ct} output tokens"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_probe(model: str, verbose: bool) -> None:
    print(f"\n{'=' * 70}")
    print(f"  Probing: {model}")
    print(f"{'=' * 70}")

    client = make_client()

    results: list[tuple[str, str]] = []

    print("\n  T1 — Reachability")
    status, msg = test_reachability(client, model, verbose)
    print(fmt(status, msg))
    results.append((status, "T1"))
    if status == FAIL:
        print("\n  Cannot reach model — skipping remaining tests.")
        _print_verdict(results)
        return
    time.sleep(2)

    print("\n  T2 — Label generation")
    status, msg, labels = test_label_generation(client, model, verbose)
    print(fmt(status, msg))
    results.append((status, "T2"))
    time.sleep(2)

    print("\n  T3 — Label merge")
    status, msg = test_label_merge(client, model, labels, verbose)
    print(fmt(status, msg))
    results.append((status, "T3"))
    time.sleep(2)

    print("\n  T4 — Classification")
    status, msg, _ = test_classification(client, model, verbose)
    print(fmt(status, msg))
    results.append((status, "T4"))
    time.sleep(2)

    print("\n  T5 — Consistency (3 runs)")
    status, msg = test_consistency(client, model, verbose)
    print(fmt(status, msg))
    results.append((status, "T5"))
    time.sleep(2)

    print("\n  T6 — Token efficiency")
    status, msg = test_token_efficiency(client, model, verbose)
    print(fmt(status, msg))
    results.append((status, "T6"))

    _print_verdict(results)


def _print_verdict(results: list[tuple[str, str]]) -> None:
    passed = sum(1 for s, _ in results if s == PASS)
    warned = sum(1 for s, _ in results if s == WARN)
    failed = sum(1 for s, _ in results if s == FAIL)
    total = len(results)

    print(f"\n  {'─' * 50}")
    print(f"  {passed} PASS  {warned} WARN  {failed} FAIL  (out of {total})")
    print(f"  {'─' * 50}")

    critical = {"T1", "T2", "T4", "T6"}
    critical_fail = any(s == FAIL and t in critical for s, t in results)

    if critical_fail or failed >= 2:
        verdict = "SKIP — not suitable for this pipeline"
    elif failed == 1 or warned >= 3:
        verdict = "USABLE — works but has issues worth noting"
    else:
        verdict = "RECOMMENDED"

    print(f"  Verdict: {verdict}")
    print(f"  {'─' * 50}\n")


def main() -> None:
    """CLI entry-point for ``tc-probe-models``."""
    parser = argparse.ArgumentParser(
        description="Test a model's compatibility with the text-clustering pipeline."
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="OpenRouter model ID (e.g. meta-llama/llama-3.3-70b-instruct:free)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show raw model responses for each test",
    )
    args = parser.parse_args()
    run_probe(args.model, args.verbose)


if __name__ == "__main__":
    main()
