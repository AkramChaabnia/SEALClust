"""
data.py — Dataset loading helpers shared by all pipeline steps.

Functions
---------
load_dataset(data_path, data, use_large)
    Load a JSONL split from dataset/<data>/{small,large}.jsonl.

get_label_list(data_list)
    Return the ordered-unique list of ground-truth labels present in the data.

get_dataset_description(dataset_name)
    Return a brief domain description for the dataset (no label leakage).
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset domain descriptions — publicly available benchmark metadata.
#
# Each description tells the LLM **what kind of data** it is looking at
# (the domain, the document type, the application context) WITHOUT
# revealing any ground-truth label names.  This is analogous to reading
# the abstract of the dataset paper.
# ---------------------------------------------------------------------------
DATASET_DESCRIPTIONS: dict[str, str] = {
    # MASSIVE — Amazon's virtual-assistant benchmark (scenario level)
    "massive_scenario": (
        "Short commands spoken to a virtual assistant (like Alexa or Siri). "
        "Each command asks the assistant to perform an action or retrieve "
        "information using one of the assistant's built-in features or "
        "connected services (e.g. controlling smart-home devices, setting "
        "reminders, playing media, ordering food, checking the weather, "
        "managing a calendar, sending messages, etc.). "
        "Categories correspond to the assistant SCENARIO — the high-level "
        "service or feature the user is interacting with."
    ),
    # MASSIVE — intent level
    "massive_intent": (
        "Short commands spoken to a virtual assistant. Each command "
        "expresses a specific user intent — the precise action the user "
        "wants the assistant to carry out (e.g. 'set an alarm', 'play a "
        "song', 'get the weather forecast'). Categories correspond to "
        "fine-grained intents within assistant capabilities."
    ),
    # Banking77 — customer-service queries
    "banking77": (
        "Customer-service queries sent to an online bank. Each query is a "
        "short message from a banking customer asking about account issues, "
        "card problems, transfers, fees, or other banking operations. "
        "Categories correspond to the specific banking issue or request type."
    ),
    # CLINC — intent detection (fine-grained)
    "clinc": (
        "Short user utterances directed at a task-oriented dialogue system. "
        "Each utterance expresses a specific intent across many everyday "
        "domains (travel, finance, utilities, etc.). Categories correspond "
        "to the fine-grained user intent."
    ),
    # CLINC — domain level
    "clinc_domain": (
        "Short user utterances directed at a task-oriented dialogue system. "
        "Each utterance belongs to a broad functional domain (e.g. travel, "
        "finance, home, utilities). Categories correspond to the "
        "high-level domain of the request."
    ),
    # MTOP — domain level
    "mtop_domain": (
        "User commands for a task-oriented virtual assistant covering "
        "multiple services. Categories correspond to the high-level "
        "service domain (e.g. messaging, events, navigation, reminders)."
    ),
    # MTOP — intent level
    "mtop_intent": (
        "User commands for a task-oriented virtual assistant. Categories "
        "correspond to the specific action-intent the user wants to perform "
        "(e.g. create a reminder, get directions, send a message)."
    ),
    # ArXiv — fine-grained subject categories
    "arxiv_fine": (
        "Titles and abstracts of academic papers from ArXiv. Categories "
        "correspond to the paper's primary subject area within computer "
        "science and related fields."
    ),
    # FewEvent — event type detection
    "few_event": (
        "Sentences describing real-world events, each annotated with the "
        "type of event mentioned (e.g. an election, a natural disaster, "
        "a sports competition). Categories correspond to the event type."
    ),
    # FewNERD — named entity type recognition
    "few_nerd_nat": (
        "Sentences containing a highlighted named entity. Categories "
        "correspond to the type of the entity (e.g. person, location, "
        "organization, product)."
    ),
    # FewRel — relation classification
    "few_rel_nat": (
        "Sentences describing a relationship between two highlighted "
        "entities. Categories correspond to the type of relationship "
        "(e.g. employer, birthplace, creator)."
    ),
    # GoEmotions — emotion detection
    "go_emotion": (
        "Reddit comments annotated with the emotion expressed by the "
        "writer. Categories correspond to fine-grained emotions "
        "(e.g. joy, anger, surprise, sadness)."
    ),
    # Reddit — subreddit clustering
    "reddit": (
        "Titles of Reddit posts from various subreddits. Categories "
        "correspond to the subreddit or topical community the post "
        "belongs to."
    ),
    # StackExchange — site clustering
    "stackexchange": (
        "Questions posted on different StackExchange sites. Categories "
        "correspond to the StackExchange community (subject area) the "
        "question was posted in."
    ),
}


def get_dataset_description(dataset_name: str) -> str:
    """Return a domain description for *dataset_name*, or a generic fallback."""
    return DATASET_DESCRIPTIONS.get(
        dataset_name,
        "A collection of short text documents. Categories correspond to "
        "the topic, intent, or type of each document.",
    )


def load_dataset(data_path, data, use_large):
    data_file = os.path.join(
        data_path, data, "large.jsonl" if use_large else "small.jsonl"
    )
    logger.info("Loading %s", data_file)
    with open(data_file, "r") as f:
        data_list = [json.loads(line) for line in f]
    logger.info("  %d samples loaded", len(data_list))
    return data_list


def get_label_list(data_list):
    seen = []
    for item in data_list:
        if item["label"] not in seen:
            seen.append(item["label"])
    return seen
