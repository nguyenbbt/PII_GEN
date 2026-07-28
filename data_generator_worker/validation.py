from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping, Sequence


_ENTITY_TAG = re.compile(r"<([A-Z][A-Z0-9_]*)>(.*?)</\1>", re.DOTALL)
_ANY_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9_]*>")


def extract_occurrence_contexts(text: str, value: str) -> list[str]:
    """Return the containing sentence/turn for every exact value occurrence."""
    clean = _ANY_TAG.sub("", text)
    contexts: list[str] = []
    search_from = 0
    while value:
        start = clean.find(value, search_from)
        if start < 0:
            break
        left = max((clean.rfind(mark, 0, start) for mark in (".", "!", "?", ";", "\n")), default=-1)
        ends = [clean.find(mark, start + len(value)) for mark in (".", "!", "?", ";", "\n")]
        right_candidates = [position for position in ends if position >= 0]
        right = min(right_candidates) + 1 if right_candidates else len(clean)
        contexts.append(clean[left + 1:right].strip())
        search_from = start + len(value)
    return contexts


def validate_generated_output(
    *,
    tagged_text: str,
    entities: Sequence[Mapping[str, Any]],
    allowed_labels: Sequence[str],
    required_labels: Sequence[str],
    sample_type: str,
    max_entities: int,
) -> list[dict[str, str]]:
    """Validate the LLM contract independently of prompt compliance.

    Entity values are compared exactly. Case variants such as ``"Test"`` and
    ``"test"`` remain distinct because Value Bank entries are opaque values.
    """
    if not isinstance(tagged_text, str) or not tagged_text.strip():
        raise ValueError("tagged_text must be a non-empty string")
    if len(entities) > max_entities:
        raise ValueError(f"entity count exceeds max_entities={max_entities}")

    allowed = set(allowed_labels)
    normalised: list[dict[str, str]] = []
    for raw in entities:
        if not isinstance(raw, Mapping):
            raise ValueError("each entity must be an object")
        label = str(raw.get("label", "")).strip()
        value = str(raw.get("value", "")).strip()
        if not label or not value:
            raise ValueError("each entity requires non-empty label and value")
        if label not in allowed:
            raise ValueError(f"entity label is not allowed: {label}")
        normalised.append({"label": label, "value": value})

    tag_pairs = _ENTITY_TAG.findall(tagged_text)
    text_without_valid_tags = _ENTITY_TAG.sub(lambda match: match.group(2), tagged_text)
    if _ANY_TAG.search(text_without_valid_tags):
        raise ValueError("tagged_text contains malformed, nested, or unmatched entity tags")
    if Counter((item["label"], item["value"]) for item in normalised) != Counter(tag_pairs):
        raise ValueError("entities must correspond exactly to tagged spans")

    if sample_type == "pure_negative":
        if normalised or tag_pairs:
            raise ValueError("pure_negative output cannot contain entities or tags")
    elif sample_type in {"positive", "hard_negative"}:
        present_labels = {item["label"] for item in normalised}
        missing = set(required_labels) - present_labels
        if missing:
            raise ValueError(f"generated output is missing focus labels: {sorted(missing)}")
    else:
        raise ValueError(f"unsupported sample_type: {sample_type}")
    return normalised


def validate_seeded_contract(
    *,
    tagged_text: str,
    entities: Sequence[Mapping[str, Any]],
    positive_entities: Sequence[Mapping[str, Any]],
    decoys: Sequence[Mapping[str, Any]],
    local_context_window: int = 80,
    max_decoy_occurrences: int = 1,
) -> None:
    """Enforce exact positive seeds and untagged, explained decoys."""
    entity_pairs = {(str(item.get("label", "")), str(item.get("value", ""))) for item in entities}
    text_without_tags = _ANY_TAG.sub("", tagged_text)
    for seed in positive_entities:
        label, value = str(seed.get("label", "")), str(seed.get("value", ""))
        expected = f"<{label}>{value}</{label}>"
        if tagged_text.count(expected) < 1:
            raise ValueError(f"positive seed must appear with its exact tag: {label}")
        if (label, value) not in entity_pairs:
            raise ValueError(f"positive seed is missing from entities: {label}")
        if value in _ANY_TAG.sub("", tagged_text.replace(expected, "")):
            raise ValueError(f"positive seed appears outside its required tag: {label}")
    entity_values = {value for _, value in entity_pairs}
    for decoy in decoys:
        value = str(decoy.get("value", ""))
        occurrence_count = tagged_text.count(value)
        if occurrence_count < 1 or occurrence_count > max_decoy_occurrences:
            raise ValueError(
                f"decoy must appear between 1 and {max_decoy_occurrences} times: {value}"
            )
        if value in entity_values or re.search(
            rf"<[A-Za-z][A-Za-z0-9_]*>{re.escape(value)}</[A-Za-z][A-Za-z0-9_]*>", tagged_text
        ):
            raise ValueError(f"decoy must remain untagged and absent from entities: {value}")
        cues = [str(cue).casefold() for cue in decoy.get("required_context_cues", [])]
        contexts = extract_occurrence_contexts(text_without_tags, value)
        if len(contexts) != occurrence_count or not cues or any(
            not any(cue in context.casefold() for cue in cues)
            for context in contexts
        ):
            raise ValueError(f"a decoy occurrence has no required context cue: {value}")
