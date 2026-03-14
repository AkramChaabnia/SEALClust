"""
prompts.py — Prompt-construction helpers for the LLM pipeline.

All functions are pure (no I/O, no side-effects) and return a prompt string.

Functions
---------
prompt_construct_generate_label(sentence_list, given_labels)
    Prompt for Step 1 — propose new label names for unchategorised texts.

prompt_construct_merge_label(label_list, target_k=None)
    Prompt for Step 1 — deduplicate/merge near-synonymous label names.

    The base prompt is identical to the original paper (ECNU-Text-Computing).
    target_k is an optional escape hatch: if provided, an extra sentence
    instructs the model to produce approximately that many final labels.
    This should NOT be used with capable models (gemini, GPT-4) — it forces
    the model to fill all k slots with spurious categories instead of
    producing a naturally consolidated set.  It exists only as a fallback
    for weaker models that under-consolidate without guidance.

prompt_construct_classify(label_list, sentence)
    Prompt for Step 2 — classify a single text into one of the known labels.
"""


def prompt_construct_generate_label(sentence_list, given_labels):
    json_example = {"labels": ["label name", "label name"]}
    prompt = f"Given the labels, under a text classicifation scenario, can all these text match the label given? If the sentence does not match any of the label, please generate a meaningful new label name.\n \
            Labels: {given_labels}\n \
            Sentences: {sentence_list} \n \
            You should NOT return meaningless label names such as 'new_label_1' or 'unknown_topic_1' and only return the new label names, please return in json format like: {json_example}"
    return prompt


def prompt_construct_merge_label(label_list, target_k: int | None = None):
    json_example = {"merged_labels": ["label name", "label name"]}
    prompt = (
        "Please analyze the provided list of labels to identify entries that are similar or "
        "duplicate, considering synonyms, variations in phrasing, and closely related terms "
        "that essentially refer to the same concept. Your task is to merge these similar entries "
        "into a single representative label for each unique concept identified. The goal is to "
        "simplify the list by reducing redundancies without organizing it into subcategories or "
        "altering its fundamental structure.\n"
    )
    if target_k is not None:
        prompt += (
            f"The final list should contain approximately {target_k} labels — "
            f"merge aggressively until you reach roughly that number.\n"
        )
    prompt += f"Here is the list of labels for analysis and simplification: {label_list}.\n"
    prompt += f"Produce the final, simplified list in a flat, JSON-formatted structure without any substructures or hierarchical categorization like: {json_example}"
    return prompt


def prompt_construct_classify(label_list, sentence):
    json_example = {"label_name": "label"}
    prompt = "Given the label list and the sentence, please categorize the sentence into one of the labels.\n"
    prompt += f"Label list: {label_list}.\n"
    prompt += f"Sentence:{sentence}.\n"
    prompt += f"You should only return the label name, please return in json format like: {json_example}"
    return prompt


# ---------------------------------------------------------------------------
# SEALClust-specific prompts
# ---------------------------------------------------------------------------

def prompt_discover_labels(representative_texts: list[str]) -> str:
    """Stage 5 — Label Discovery.

    Send a batch of representative documents to the LLM and ask it to
    propose semantic labels/topics that describe the data.

    Unlike the original ``prompt_construct_generate_label`` which uses seed
    labels and incrementally grows a label set, this prompt is *seed-free*:
    the LLM sees only representative documents and freely discovers topics.
    """
    json_example = {"labels": ["label name 1", "label name 2", "..."]}
    prompt = (
        "You are an expert text analyst. Below are representative documents sampled "
        "from a large text dataset. Each document represents a cluster of similar texts.\n\n"
        "Your task is to read all these documents and propose a list of meaningful, "
        "descriptive topic labels that capture the main themes present in the data.\n\n"
        "Guidelines:\n"
        "- Each label should be a short, descriptive phrase (2-5 words).\n"
        "- Cover all the themes you observe — it is better to propose too many labels than too few.\n"
        "- Do NOT use generic labels like 'other', 'miscellaneous', or 'unknown'.\n"
        "- Labels should be mutually exclusive when possible.\n\n"
        f"Documents:\n"
    )
    for i, text in enumerate(representative_texts, 1):
        prompt += f"{i}. {text}\n"
    prompt += (
        f"\nReturn the complete list of proposed labels in JSON format like: {json_example}"
    )
    return prompt


def prompt_consolidate_labels(label_list: list[str], target_k: int) -> str:
    """Stage 7 — Final Label Consolidation.

    Given a list of candidate labels and the statistically optimal number of
    clusters K*, ask the LLM to merge them into exactly K* final labels.
    """
    json_example = {"merged_labels": ["label name 1", "label name 2"]}
    prompt = (
        f"You are an expert text analyst. You have been given a list of {len(label_list)} "
        f"candidate topic labels discovered from a text dataset.\n\n"
        f"A statistical analysis has determined that the optimal number of clusters is "
        f"**exactly {target_k}**.\n\n"
        f"Your task is to merge the candidate labels into exactly {target_k} final labels "
        f"by combining similar, overlapping, or redundant labels into broader categories.\n\n"
        f"Guidelines:\n"
        f"- You MUST produce exactly {target_k} labels — no more, no less.\n"
        f"- Each final label should be a short, descriptive phrase (2-5 words).\n"
        f"- Merge semantically similar labels together.\n"
        f"- Every candidate label should be covered by one of the final labels.\n"
        f"- Do NOT use generic labels like 'other' or 'miscellaneous'.\n\n"
        f"Candidate labels:\n{label_list}\n\n"
        f"Return exactly {target_k} merged labels in JSON format like: {json_example}"
    )
    return prompt
