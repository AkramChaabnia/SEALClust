"""
remerge_labels.py — Re-merge proposed labels to a target K.

When the initial label generation produces far more labels than needed
(e.g. 972 → 59), a single LLM call cannot reliably reduce them.  This
tool uses an **iterative chunked reduction** strategy:

  1. Split labels into large chunks (~200 labels each).
  2. Ask the LLM to aggressively merge each chunk to ~20% of its size.
  3. Collect the results into a new, smaller label list.
  4. Repeat until small enough for a single final merge to target_k.

  972 labels → ~5 chunks → ~200 total → final merge → 59.
  Total: ~6 LLM calls, 2 rounds.

Usage::

    python -m text_clustering.tools.remerge_labels <run_dir> <target_k>
    python -m text_clustering.tools.remerge_labels ./runs/massive_intent_small_20260319_151040 59
"""

import json
import math
import re
import sys
import time

from text_clustering.llm import chat, ini_client
from text_clustering.prompts import prompt_construct_merge_label


def _aggressive_merge_prompt(label_list: list[str], target_k: int) -> str:
    """Prompt that forcefully demands exactly target_k merged labels."""
    return (
        "You are a label-merging machine. Your ONLY job is to reduce "
        f"the following {len(label_list)} labels down to EXACTLY "
        f"{target_k} labels by merging similar, overlapping, and "
        "related labels into broad categories.\n\n"
        "RULES:\n"
        f"1. You MUST return EXACTLY {target_k} labels. Not more, "
        "not fewer.\n"
        "2. Merge aggressively — combine anything remotely related "
        "into a single broad label.\n"
        "3. Use short, general category names.\n"
        "4. Every original label must fit under one of your merged "
        "labels.\n\n"
        f"Labels to merge:\n{label_list}\n\n"
        "Return ONLY valid JSON. No explanation, no markdown.\n"
        "Format exactly like this:\n"
        '{"merged_labels": ["label1", "label2", "..."]}\n'
    )

# ── Parsing helpers ────────────────────────────────────────────────────

def safe_parse(response: str) -> list[str] | None:
    """Extract a flat list of label strings from LLM output."""
    if response is None:
        return None

    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", response).strip()

    # Try direct JSON parse
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON block
        match = re.search(r"\{.*\}|\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None

    # Normalise to flat list
    if isinstance(obj, list):
        return [str(x) for x in obj if isinstance(x, str)]
    if isinstance(obj, dict):
        for val in obj.values():
            if isinstance(val, list):
                return [str(x) for x in val if isinstance(x, str)]
    return None


# ── Merge helpers ──────────────────────────────────────────────────────

def _merge_one_chunk(
    labels: list[str],
    target: int,
    client,
    attempt_label: str = "",
    aggressive: bool = True,
) -> list[str]:
    """Merge a single chunk of labels towards *target* via one LLM call.

    When *aggressive* is True (default for intermediate rounds), uses a
    forceful prompt that demands exactly *target* labels.  For the final
    pass, set aggressive=False to use the softer standard prompt.

    Returns the merged list, or the original list on failure.
    """
    if aggressive:
        prompt = _aggressive_merge_prompt(labels, target)
    else:
        prompt = prompt_construct_merge_label(labels, target_k=target)
    response = chat(prompt, client, max_tokens=4096)
    parsed = safe_parse(response)
    if parsed and len(parsed) < len(labels):
        return parsed
    # If parsing failed or no reduction, return original
    return labels


def iterative_merge(
    labels: list[str],
    target_k: int,
    client,
    *,
    chunk_size: int = 250,
    max_rounds: int = 10,
) -> list[str]:
    """Iteratively reduce *labels* to *target_k* via chunked LLM merges.

    Strategy — aggressive & fast
    ----------------------------
    Each round:
      - If len(labels) ≤ chunk_size → single final merge to target_k.
      - Otherwise split into big chunks (~250), aggressively reduce each
        to a proportional share of target_k, collect results, repeat.

    With 250-label chunks, 972 labels = 4 chunks. Each chunk gets told
    to merge to ~(target_k / n_chunks) labels. So round 1 goes from
    972 → ~59 directly in 4 parallel-ish calls.
    """
    current = list(dict.fromkeys(labels))  # deduplicate preserving order
    print(f"Starting iterative merge: {len(current)} → {target_k}")

    for round_num in range(1, max_rounds + 1):
        n = len(current)

        # Already at or below target
        if n <= target_k:
            print(f"  Round {round_num}: already at {n} ≤ {target_k}")
            break

        # Small enough for a single final merge
        if n <= chunk_size:
            print(
                f"  Round {round_num}: final merge {n} → {target_k} "
                f"(single call)"
            )
            best = current
            best_dist = abs(len(current) - target_k)
            for attempt in range(1, 4):
                # Close to target → gentle trim; far → aggressive
                use_aggressive = (n > target_k * 1.5)
                result = _merge_one_chunk(
                    current, target_k, client,
                    attempt_label=f"final-{attempt}",
                    aggressive=use_aggressive,
                )
                dist = abs(len(result) - target_k)
                print(
                    f"    attempt {attempt}: {len(result)} labels "
                    f"(off by {dist})"
                )
                if dist < best_dist:
                    best = result
                    best_dist = dist
                if dist == 0:
                    break
            current = best
            break

        # ── Chunked aggressive reduction round ──
        # Each chunk gets a proportional share of target_k.
        # Overshoot by 30% to compensate for cross-chunk dedup.
        n_chunks = math.ceil(n / chunk_size)
        per_chunk_target = max(3, int(target_k * 1.3) // n_chunks)
        merged_round: list[str] = []

        print(
            f"  Round {round_num}: {n} labels → {n_chunks} chunks "
            f"of ~{chunk_size}  (each → ~{per_chunk_target})"
        )

        for i in range(n_chunks):
            chunk = current[i * chunk_size : (i + 1) * chunk_size]
            # Scale target proportionally for smaller last chunk
            chunk_target = max(
                3, int(per_chunk_target * len(chunk) / chunk_size)
            )

            result = _merge_one_chunk(
                chunk, chunk_target, client,
                attempt_label=f"r{round_num}-c{i+1}",
                aggressive=True,
            )
            merged_round.extend(result)
            print(
                f"    chunk {i+1}/{n_chunks}: "
                f"{len(chunk)} → {len(result)}"
            )

        # Deduplicate across chunks
        current = list(dict.fromkeys(merged_round))
        print(
            f"  Round {round_num} done: {n} → {len(current)} "
            f"(target: {target_k})"
        )

        # Safety: if no progress, break to avoid infinite loop
        if len(current) >= n:
            print(
                f"  ⚠ No reduction in round {round_num} — stopping"
            )
            break

    print(f"Iterative merge complete: {len(current)} labels")
    return current


# ── Main ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(
            "Usage: tc-remerge-labels <run_dir> <target_k>\n"
            "       python -m text_clustering.tools.remerge_labels <run_dir> <target_k>"
        )
        sys.exit(1)

    run_dir = sys.argv[1]
    target_k = int(sys.argv[2])

    client = ini_client()

    with open(f"{run_dir}/labels_proposed.json") as f:
        all_labels = json.load(f)

    print(f"Proposed labels: {len(all_labels)}")
    print(f"Target K:        {target_k}")
    print()

    t0 = time.time()
    final = iterative_merge(all_labels, target_k, client)
    elapsed = time.time() - t0

    print()
    print(f"Final label count: {len(final)}  (target was {target_k})")
    print(f"Time: {elapsed:.1f}s")
    print()
    for i, label in enumerate(final, 1):
        print(f"  {i:3d}. {label}")

    with open(f"{run_dir}/labels_merged.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved → {run_dir}/labels_merged.json")


if __name__ == "__main__":
    main()
