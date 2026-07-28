from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_ENTITY_TAG = re.compile(r"<([A-Z][A-Z0-9_]*)>(.*?)</\1>", re.DOTALL)
_ANY_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9_]*>")
_TEMPLATE_ARTIFACT = re.compile(
    r"\[(?P<field>(?:"
    r"Tên|Ngày|Chức\s+danh|Địa\s+chỉ|Số|Công\s+ty|Khách\s+hàng|"
    r"Đại\s+diện|Tài\s+xế|Name|Date|Title|Company|Customer|Address|"
    r"Datum|Firma|Kunde|Adresse"
    r")[^\[\]\r\n]{0,70})\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TaggedSpan:
    label: str
    value: str
    start: int
    end: int


def tagged_text_to_clean_and_spans(tagged_text: str) -> tuple[str, list[TaggedSpan]]:
    """Remove valid entity tags while retaining clean-text span boundaries."""
    clean_parts: list[str] = []
    spans: list[TaggedSpan] = []
    source_cursor = 0
    clean_cursor = 0
    for match in _ENTITY_TAG.finditer(tagged_text):
        plain = tagged_text[source_cursor:match.start()]
        clean_parts.append(plain)
        clean_cursor += len(plain)
        value = match.group(2)
        start = clean_cursor
        clean_parts.append(value)
        clean_cursor += len(value)
        spans.append(
            TaggedSpan(
                label=match.group(1),
                value=value,
                start=start,
                end=clean_cursor,
            )
        )
        source_cursor = match.end()
    clean_parts.append(tagged_text[source_cursor:])
    return "".join(clean_parts), spans


def find_template_artifacts(text: str) -> list[tuple[int, int, str]]:
    """Find unresolved human-facing fill-in fields in clean or tagged text."""
    clean = _ANY_TAG.sub("", text)
    return [
        (match.start(), match.end(), match.group(0))
        for match in _TEMPLATE_ARTIFACT.finditer(clean)
    ]


def seed_realization_status(
    tagged_text: str,
    *,
    label: str,
    value: str,
) -> str:
    """Classify whether an exact seed surface is safely represented by NER spans.

    ``partitioned`` allows a composite Value Bank entry to be split along taxonomy
    boundaries, for example an ADDRESS street span followed by a LOCATION span.
    The original seed surface must remain byte-for-byte unchanged and at least one
    partition must retain the seed's original label.
    """
    clean, spans = tagged_text_to_clean_and_spans(tagged_text)
    occurrences: list[tuple[int, int]] = []
    cursor = 0
    while value:
        start = clean.find(value, cursor)
        if start < 0:
            break
        occurrences.append((start, start + len(value)))
        cursor = start + len(value)
    if not occurrences:
        return "missing"

    signatures: list[tuple[tuple[str, int, int], ...]] = []
    occurrence_statuses: list[str] = []
    for start, end in occurrences:
        containing = [
            span
            for span in spans
            if span.start < end and span.end > start
        ]
        exact = any(
            span.label == label and span.start == start and span.end == end
            for span in containing
        )
        if exact and len(containing) == 1:
            occurrence_statuses.append("exact")
            signatures.append(((label, 0, len(value)),))
            continue

        if any(
            span.label == label
            and span.start <= start
            and span.end >= end
            and span.value.strip() == value
            for span in containing
        ):
            occurrence_statuses.append("boundary_whitespace")
            signatures.append(())
            continue

        internal = sorted(
            (
                span
                for span in containing
                if span.start >= start and span.end <= end
            ),
            key=lambda span: (span.start, span.end),
        )
        covered = {
            index
            for span in internal
            for index in range(span.start - start, span.end - start)
        }
        alphanumeric_covered = all(
            not character.isalnum() or index in covered
            for index, character in enumerate(value)
        )
        has_original_label = any(span.label == label for span in internal)
        non_overlapping = all(
            left.end <= right.start
            for left, right in zip(internal, internal[1:])
        )
        if internal and alphanumeric_covered and has_original_label and non_overlapping:
            occurrence_statuses.append("partitioned")
            signatures.append(tuple(
                (span.label, span.start - start, span.end - start)
                for span in internal
            ))
        else:
            occurrence_statuses.append("unannotated")
            signatures.append(())

    if len(set(signatures)) > 1:
        return "inconsistent"
    if all(status == "exact" for status in occurrence_statuses):
        return "exact"
    if all(status == "partitioned" for status in occurrence_statuses):
        return "partitioned"
    if all(status == "boundary_whitespace" for status in occurrence_statuses):
        return "boundary_whitespace"
    return "inconsistent"


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
    """Enforce exact seed surfaces, taxonomy-safe spans, and explained decoys."""
    entity_pairs = {(str(item.get("label", "")), str(item.get("value", ""))) for item in entities}
    text_without_tags = _ANY_TAG.sub("", tagged_text)
    for seed in positive_entities:
        label, value = str(seed.get("label", "")), str(seed.get("value", ""))
        status = seed_realization_status(tagged_text, label=label, value=value)
        if status not in {"exact", "partitioned"}:
            raise ValueError(
                f"positive seed has invalid surface or annotation ({status}): {label}"
            )
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
