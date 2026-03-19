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


def prompt_construct_classify_batch(label_list: list[str], sentences: list[str]) -> str:
    """Batched classification prompt — classify multiple sentences in a single call.

    Sends N indexed sentences and asks the LLM to return a JSON object mapping
    each index to its predicted label.  This reduces LLM calls by the batch
    size factor (e.g. batch_size=10 → 10× fewer calls).

    Parameters
    ----------
    label_list : list[str]
        Available labels to classify into.
    sentences : list[str]
        Batch of sentences to classify.

    Returns
    -------
    str
        Prompt string requesting indexed JSON output.
    """
    json_example = {"1": "label_a", "2": "label_b", "3": "label_c"}
    prompt = (
        "Given the label list, categorize EACH of the following sentences "
        "into exactly one of the labels.\n\n"
        f"Label list: {label_list}\n\n"
        "Sentences:\n"
    )
    for i, sentence in enumerate(sentences, 1):
        prompt += f'{i}. {sentence}\n'
    prompt += (
        f"\nReturn a JSON object mapping each sentence number to its label, "
        f"like: {json_example}\n"
        f"You MUST classify ALL {len(sentences)} sentences. "
        f"Use ONLY labels from the label list above."
    )
    return prompt


# ---------------------------------------------------------------------------
# Hybrid pipeline prompts
# ---------------------------------------------------------------------------

def prompt_hybrid_generate_labels(sentence_list: list[str]) -> str:
    """Step 1 of the Hybrid pipeline — generate one-word general labels.

    Given a batch of documents, ask the LLM to assign one general one-word
    label to each document.  No seed labels are provided — the LLM freely
    discovers topics.

    Returns a prompt string that asks for a JSON response mapping each
    document index to its one-word label.
    """
    json_example = {"labels": ["Technology", "Sports", "Finance"]}
    prompt = (
        "You are an expert text classifier. For each of the following documents, "
        "assign exactly ONE label.\n\n"
        "Rules:\n"
        "- Each label must be exactly ONE word.\n"
        "- Labels must be as GENERAL as possible (broad categories, not specific topics).\n"
        "- Reuse the same label for documents that belong to the same category.\n"
        "- Do NOT use generic labels like 'Other', 'Miscellaneous', or 'Unknown'.\n\n"
        "Documents:\n"
    )
    for i, text in enumerate(sentence_list, 1):
        prompt += f"{i}. {text}\n"
    prompt += (
        f"\nReturn a JSON object with a 'labels' key containing a list of labels "
        f"(one per document, in the same order) like: {json_example}"
    )
    return prompt


def prompt_hybrid_reduce_labels(label_list: list[str]) -> str:
    """Step 3 of the Hybrid pipeline — merge semantically similar labels.

    Given K0 labels (potentially hundreds), ask the LLM to merge synonyms,
    variations, and closely related labels into a reduced set K1.
    """
    json_example = {"merged_labels": ["Technology", "Sports", "Finance"]}
    prompt = (
        "You are an expert text analyst. You are given a list of one-word labels "
        "that were generated for a text classification task.\n\n"
        "Your task is to merge labels that are semantically similar, synonymous, "
        "or refer to the same concept into a single representative label.\n\n"
        "Rules:\n"
        "- Each final label must be exactly ONE word.\n"
        "- Merge aggressively — combine all synonyms and near-synonyms.\n"
        "- Keep labels as general as possible.\n"
        "- Do NOT create new labels that weren't implied by the originals.\n"
        "- Do NOT use 'Other', 'Miscellaneous', or 'Unknown'.\n\n"
        f"Labels to merge:\n{label_list}\n\n"
        f"Return the merged list in JSON format like: {json_example}"
    )
    return prompt


def prompt_hybrid_align_labels(
    label_list: list[str], target_k: int,
) -> str:
    """Step 5 of the Hybrid pipeline — force labels to exactly target_k.

    Given a set of labels and the target number of categories, ask the LLM
    to merge/split until exactly target_k labels remain.
    """
    json_example = {"aligned_labels": ["label1", "label2"]}
    prompt = (
        f"You are an expert text analyst. You have {len(label_list)} labels "
        f"but need exactly {target_k} final labels.\n\n"
        f"Current labels:\n{label_list}\n\n"
        f"Your task is to merge or group these labels into exactly {target_k} "
        f"final labels.\n\n"
        f"Rules:\n"
        f"- You MUST produce exactly {target_k} labels — no more, no less.\n"
        f"- Each label should be a short, descriptive phrase (1-3 words).\n"
        f"- Merge semantically similar labels together.\n"
        f"- Every original label should be covered by one of the final labels.\n"
        f"- Do NOT use 'Other', 'Miscellaneous', or 'Unknown'.\n\n"
        f"Return exactly {target_k} labels in JSON format like: {json_example}"
    )
    return prompt


def prompt_hybrid_classify_medoid(
    label_list: list[str], document: str,
) -> str:
    """Step 7 of the Hybrid pipeline — assign a label to a medoid document.

    Given the final label set and a single medoid document, ask the LLM to
    pick the most relevant label.
    """
    json_example = {"label": "Technology"}
    prompt = (
        "You are an expert text classifier. Assign exactly one label from the "
        "list below to the given document.\n\n"
        f"Available labels: {label_list}\n\n"
        f"Document: {document}\n\n"
        "Rules:\n"
        "- You MUST choose a label from the list above.\n"
        "- Pick the label that best describes the document's main topic.\n"
        f"- Return in JSON format like: {json_example}"
    )
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
