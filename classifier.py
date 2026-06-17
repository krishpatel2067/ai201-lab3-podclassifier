import json
import os
import re
from groq import Groq
from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    VALID_LABELS,
    DATA_PATH,
    TRAIN_FILE,
    LABELS_FILE,
)

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Builds a few-shot classification prompt using the labeled training examples.
    """
    instructions = (
        "You are classifying podcast episodes by their format. Classify the "
        "episode into exactly one of these four labels:\n"
        "- interview: a conversation between a host and one or more guests\n"
        "- solo: a single host speaking from memory, experience, or opinion — "
        "no guests, no assembled external sources\n"
        "- panel: multiple guests with roughly equal speaking time, often "
        "debating or discussing a topic together\n"
        "- narrative: a story assembled from external sources — interviews, "
        "archival audio, reporting — with a clear narrative arc\n\n"
        "Output format:\n"
        "Label: <lowercase label>\n"
        "Reasoning: <a brief reasoning sentence>\n\n"
        "Here are some examples:\n\n"
    )

    example_blocks = []
    for ex in labeled_examples:
        block = (
            f"Title: {ex['title']}\n"
            f"Description: {ex['description']}\n"
            f"Label: {ex['label']}\n"
        )
        example_blocks.append(block)

    prompt = instructions + "\n---\n\n".join(example_blocks)
    prompt += (
        f"\n\nNow classify this episode:\n\n"
        f"Title: (New Episode)\n"
        f"Description: {description}\n"
        f"Label: ?"
    )

    return prompt


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classifies a single podcast episode description using the few-shot LLM
    classifier.
    """
    if not description:
        return {"label": "unknown", "reasoning": "Empty description provided."}

    prompt = build_few_shot_prompt(labeled_examples, description)

    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful classification assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,  # Keep it deterministic
            max_tokens=150,
        )
        response = response.choices[0].message.content.strip()

        # Robust parsing for the two-line format
        lines = [line.strip() for line in response.split("\n") if line.strip()]

        if not lines:
            return {"label": "unknown", "reasoning": "LLM returned an empty response."}

        # Extract the label and reasoning using the specified format
        match = re.match(r"^Label:\s*([a-zA-Z]+)$", lines[0], flags=re.I)
        raw_label = match.group(1).lower() if match else ""

        reasoning = ""

        if len(lines) > 1:
            match = re.match(r"^Reasoning:\s*(.+)$", lines[1], flags=re.I)
            reasoning = match.group(1) if match else ""

        final_label = raw_label if raw_label in VALID_LABELS else "unknown"
        final_reasoning = reasoning if reasoning else "No reasoning provided."

        return {
            "label": final_label,
            "reasoning": final_reasoning,
        }

    except Exception as e:
        return {
            "label": "unknown",
            "reasoning": f"LLM Call failed: {str(e)}",
        }
