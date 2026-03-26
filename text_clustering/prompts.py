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
    This should NOT be used with capable models (e.g. gemini-2.0-flash) — it forces
    the model to fill all k slots with spurious categories instead of
    producing a naturally consolidated set.  It exists only as a fallback
    for weaker models that under-consolidate without guidance.

prompt_construct_classify(label_list, sentence)
    Prompt for Step 2 — classify a single text into one of the known labels.
"""


def prompt_construct_generate_label(sentence_list, given_labels):
    json_example = {"labels": ["label name", "label name"]}
    prompt = f"Given the labels, under a text classification scenario, can all these text match the label given? If the sentence does not match any of the label, please generate a meaningful new label name.\n \
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
    prompt += (
    "\nReturn ONLY valid JSON. Do not include any explanation, text, or markdown.\n"
    "The output MUST be strictly valid JSON.\n"
    "Format exactly like this:\n"
    '{"merged_labels": ["label1", "label2", "..."]}\n'
    )
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


# ---------------------------------------------------------------------------
# SEAL-Clust v3 prompts
# ---------------------------------------------------------------------------

def prompt_v3_discover_labels(representative_texts: list[str]) -> str:
    """Stage 5 — v3 Label Discovery (one-word, general labels).

    Unlike v2 which allows 2-5 word phrases, v3 enforces strict ONE WORD
    labels that are GENERAL category descriptors (e.g. "Technology", "Sports").
    """
    json_example = {"labels": ["Technology", "Sports", "Finance", "Health"]}
    prompt = (
        "You are an expert text analyst. Below are representative documents sampled "
        "from a large text dataset. Each document represents a cluster of similar texts.\n\n"
        "Your task is to read all these documents and propose a list of meaningful "
        "topic labels that capture the main themes present in the data.\n\n"
        "STRICT RULES:\n"
        "1. Each label MUST be exactly ONE WORD.\n"
        "2. Labels must be GENERAL category descriptors (broad categories, not specific topics).\n"
        "3. Reuse the same label for documents that belong to the same category.\n"
        "4. Do NOT use generic labels like 'Other', 'Miscellaneous', or 'Unknown'.\n"
        "5. Cover all themes — propose too many rather than too few.\n\n"
        "Documents:\n"
    )
    for i, text in enumerate(representative_texts, 1):
        prompt += f"{i}. {text}\n"
    prompt += (
        f"\nReturn the complete list of proposed ONE-WORD labels in JSON format like: {json_example}\n"
        "Return ONLY valid JSON. No explanation, no markdown."
    )
    return prompt


def prompt_v3_consolidate_labels(label_list: list[str], target_k: int) -> str:
    """Stage 7 — v3 Label Consolidation (strict K*).

    Merge candidate one-word labels into exactly K* final labels.
    Uses aggressive language to enforce the exact count.
    """
    json_example = {"merged_labels": ["Label1", "Label2", "Label3"]}
    prompt = (
        f"You are a label-merging machine.\n\n"
        f"INPUT: {len(label_list)} candidate labels.\n"
        f"OUTPUT: exactly {target_k} merged labels.\n\n"
        f"ABSOLUTE CONSTRAINT: Your output list MUST contain EXACTLY "
        f"{target_k} items. Count them carefully before responding. "
        f"Returning {target_k + 1} or {target_k - 1} is WRONG.\n\n"
        f"RULES:\n"
        f"1. Return a JSON list with EXACTLY {target_k} labels.\n"
        f"2. Each label MUST be exactly ONE WORD.\n"
        f"3. Merge aggressively — combine anything semantically related "
        f"into a single broad label.\n"
        f"4. Every original label must fit under one of your merged labels.\n"
        f"5. Do NOT use 'Other', 'Miscellaneous', or 'Unknown'.\n\n"
        f"Candidate labels:\n{label_list}\n\n"
        f"Return EXACTLY {target_k} merged ONE-WORD labels in JSON format "
        f"like: {json_example}\n"
        f"Return ONLY valid JSON. No explanation, no markdown. "
        f"VERIFY: count your output — it must be {target_k}."
    )
    return prompt


def prompt_v3_classify_representative(
    label_list: list[str], document: str,
) -> str:
    """Stage 8 — v3 Representative Classification (single doc).

    Assign exactly one label from the final set to a representative document.
    """
    json_example = {"label": "Technology"}
    prompt = (
        "You are an expert text classifier. Assign exactly one label from the "
        "list below to the given document.\n\n"
        f"Available labels: {label_list}\n\n"
        f"Document: {document}\n\n"
        "Rules:\n"
        "- You MUST choose a label from the list above — no other label is allowed.\n"
        "- Pick the label that best describes the document's main topic.\n"
        f"- Return in JSON format like: {json_example}\n"
        "Return ONLY valid JSON. No explanation, no markdown."
    )
    return prompt


def prompt_v3_classify_representatives_batch(
    label_list: list[str], documents: list[str],
) -> str:
    """Stage 8 — v3 Batched Representative Classification.

    Classify multiple representative documents into K* labels in one call.
    """
    json_example = {"1": "Technology", "2": "Sports", "3": "Finance"}
    prompt = (
        "You are an expert text classifier. Assign exactly one label from the list below "
        "to EACH of the following documents.\n\n"
        f"Available labels: {label_list}\n\n"
        "Documents:\n"
    )
    for i, doc in enumerate(documents, 1):
        prompt += f"{i}. {doc}\n"
    prompt += (
        f"\nRules:\n"
        f"- You MUST choose a label from the list above for each document.\n"
        f"- You MUST classify ALL {len(documents)} documents.\n"
        f"- Return a JSON object mapping each document number to its label like: {json_example}\n"
        f"Return ONLY valid JSON. No explanation, no markdown."
    )
    return prompt


# ---------------------------------------------------------------------------
# SEAL-Clust v4 prompts — improved label quality
# ---------------------------------------------------------------------------
# v4 design rationale:
#   - Discovery: functional-intent focus — identify CATEGORY-LEVEL labels
#     (e.g. "sports", "finance", "shopping") rather than hyper-specific labels
#     ("Sports Results Management", "Financial Planning").  This dramatically
#     reduces label explosion (from 900+ to ~40-80 candidate labels).
#   - Consolidation: concrete merge-vs-keep examples and strict counting.
#     Merge rules reference functional domain equivalence, not surface
#     similarity of label text.
#   - Classification: clean intent-matching — pick the category that
#     captures the CORE action or request in the document.
# ---------------------------------------------------------------------------


def prompt_v4_discover_labels(
    representative_texts: list[str],
    existing_labels: list[str] | None = None,
    dataset_description: str = "",
) -> str:
    """Stage 5 — v4 Label Discovery.

    When *dataset_description* is provided, the LLM gets domain context
    that helps it produce more accurate, appropriately-scoped labels.
    When *existing_labels* is provided the LLM is encouraged to reuse them
    when appropriate, but should still create new labels when needed.
    """
    json_example = {"labels": ["label_a", "label_b", "label_c"]}
    prompt = "You are an expert text analyst.\n\n"

    if dataset_description:
        prompt += (
            f"DATASET CONTEXT: {dataset_description}\n\n"
        )

    prompt += (
        "Your task: examine the documents below and identify the distinct "
        "CATEGORIES they belong to, then return the set of distinct "
        "category labels.\n\n"
    )

    # If we already have labels from previous batches, encourage reuse
    if existing_labels:
        prompt += (
            "LABELS ALREADY IDENTIFIED (from previous batches):\n"
            f"  {existing_labels}\n\n"
            "- Reuse an existing label when a document CLEARLY fits it.\n"
            "- Create a NEW label when a document belongs to a genuinely "
            "different category not represented above.\n\n"
        )

    prompt += (
        "RULES:\n"
        "1. Each label must be ONE lowercase word.\n"
        "2. Labels should describe the HIGH-LEVEL scenario, service, or "
        "feature area — not the specific device, object, or topic.\n"
        "3. Different actions or requests within the SAME service share "
        "ONE label.\n"
        "4. Keep genuinely DIFFERENT services/features as separate labels.\n"
        "5. Do NOT create near-duplicates.\n\n"
        "Documents:\n"
    )
    for i, text in enumerate(representative_texts, 1):
        prompt += f"{i}. {text}\n"
    prompt += (
        f"\nReturn ONLY the list of distinct category labels in JSON "
        f"format like: {json_example}\n"
        "Include both reused labels AND any new ones. "
        "Return ONLY valid JSON. No explanation, no markdown."
    )
    return prompt


def prompt_v4_consolidate_labels(
    label_list: list[str],
    target_k: int,
    dataset_description: str = "",
) -> str:
    """Stage 7 — v4 Label Consolidation.

    When *dataset_description* is provided, gives the LLM domain context
    so it makes smarter merge decisions.
    """
    json_example = {"merged_labels": ["category_a", "category_b", "category_c"]}
    labels_str = "\n".join(f"  - {lbl}" for lbl in label_list)
    prompt = ""

    if dataset_description:
        prompt += f"DATASET CONTEXT: {dataset_description}\n\n"

    prompt += (
        f"You have {len(label_list)} candidate category labels collected "
        f"from a dataset. Many are duplicates, synonyms, or near-duplicates "
        f"that should be merged. Your task: consolidate into EXACTLY "
        f"{target_k} final labels.\n\n"
        f"MERGE RULES (strict):\n"
        f"1. MERGE labels that are exact synonyms or spelling variants "
        f"(e.g. 'lights'/'lighting'/'lamp' → pick one).\n"
        f"2. MERGE labels that clearly refer to the same real-world "
        f"concept (e.g. 'reminder'/'alarm' if they both mean scheduled "
        f"alerts).\n"
        f"3. MERGE hyper-specific labels INTO the broader category they "
        f"belong to ONLY when there is already a broader label in the "
        f"list (e.g. 'coffee'/'recipe' → 'cooking' if 'cooking' exists).\n"
        f"4. DO NOT merge labels that represent DIFFERENT user scenarios "
        f"or services, even if they seem topically related.\n\n"
        f"PRESERVE RULES (critical):\n"
        f"- Labels that describe DIFFERENT user actions, services, or "
        f"feature areas must stay separate even if topically related.\n"
        f"- E.g. 'making food at home' vs. 'ordering food delivery' are "
        f"DIFFERENT scenarios — do NOT merge them.\n"
        f"- E.g. 'sending an email' vs. 'posting on social media' are "
        f"DIFFERENT communication types — do NOT merge them.\n"
        f"- E.g. 'setting an alarm' vs. 'managing a calendar event' are "
        f"DIFFERENT time-management features — do NOT merge them.\n"
        f"- When in doubt, KEEP labels separate rather than merging.\n\n"
        f"LABEL FORMAT:\n"
        f"- Each final label: 1 lowercase word.\n"
        f"- Do NOT create vague catch-all labels like 'stuff', "
        f"'information', 'services', 'other', or 'miscellaneous'.\n"
        f"- The final list must have EXACTLY {target_k} labels.\n\n"
        f"Candidate labels:\n{labels_str}\n\n"
        f"Return EXACTLY {target_k} merged labels in JSON format "
        f"like: {json_example}\n"
        f"Return ONLY valid JSON. No explanation, no markdown."
    )
    return prompt


def prompt_v4_classify_representative(
    label_list: list[str],
    document: str,
    dataset_description: str = "",
) -> str:
    """Stage 8 — v4 Single Representative Classification."""
    json_example = {"label": label_list[0] if label_list else "label_a"}
    prompt = ""

    if dataset_description:
        prompt += f"DATASET CONTEXT: {dataset_description}\n\n"

    prompt += (
        "Classify the document into exactly ONE of the given labels.\n\n"
        f"Labels: {label_list}\n\n"
        f"Document: \"{document}\"\n\n"
        "Instructions:\n"
        "- Determine which category the document belongs to, given the "
        "dataset context above.\n"
        "- You MUST pick from the labels listed above — no other value "
        "is allowed.\n"
        "- If the document could fit multiple labels, pick the most "
        "SPECIFIC one.\n"
        f"- Return in JSON format like: {json_example}\n"
        "Return ONLY valid JSON."
    )
    return prompt


def prompt_v4_classify_representatives_batch(
    label_list: list[str],
    documents: list[str],
    dataset_description: str = "",
) -> str:
    """Stage 8 — v4 Batched Representative Classification."""
    ex_labels = label_list[:3] if len(label_list) >= 3 else label_list
    json_example = {str(i + 1): lbl for i, lbl in enumerate(ex_labels)}
    prompt = ""

    if dataset_description:
        prompt += f"DATASET CONTEXT: {dataset_description}\n\n"

    prompt += (
        "Classify EACH document below into exactly ONE of the given labels.\n\n"
        f"Labels: {label_list}\n\n"
        "Documents:\n"
    )
    for i, doc in enumerate(documents, 1):
        prompt += f"{i}. {doc}\n"
    prompt += (
        f"\nInstructions:\n"
        f"- For each document, determine which category it belongs to, "
        f"given the dataset context.\n"
        f"- Choose the most SPECIFIC matching label.\n"
        f"- You MUST pick from the labels listed above — no other value "
        f"is allowed.\n"
        f"- Be CONSISTENT: similar documents must receive the same label.\n"
        f"- You MUST classify ALL {len(documents)} documents.\n"
        f"- Return a JSON object mapping each document number to its label "
        f"like: {json_example}\n"
        f"Return ONLY valid JSON."
    )
    return prompt
