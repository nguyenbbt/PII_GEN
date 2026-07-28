from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_PLACEHOLDER = re.compile(r"\[([A-Z][A-Z0-9_]*)_([1-9][0-9]*)\]")
_BARE_PLACEHOLDER = re.compile(r"[A-Z][A-Z0-9_]*_[1-9][0-9]*")
_ENTITY_TAG = re.compile(r"<[A-Z][A-Z0-9_]*>([^<>]+)</[A-Z][A-Z0-9_]*>")


@dataclass(frozen=True)
class PlaceholderBinding:
    label: str
    placeholder: str
    value: str


def build_placeholder_bindings(
    positive_entities: Sequence[Mapping[str, Any]],
) -> list[PlaceholderBinding]:
    """Assign stable, per-class placeholders in seed order."""

    counts: defaultdict[str, int] = defaultdict(int)
    bindings: list[PlaceholderBinding] = []
    for entity in positive_entities:
        label = str(entity.get("label", "")).strip().upper()
        value = str(entity.get("value", ""))
        if not label or not value:
            raise ValueError("positive entity requires non-empty label and value")
        counts[label] += 1
        bindings.append(
            PlaceholderBinding(
                label=label,
                placeholder=f"[{label}_{counts[label]}]",
                value=value,
            )
        )
    return bindings


def placeholder_entities(
    positive_entities: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy positive seed metadata while replacing values with placeholders."""

    bindings = build_placeholder_bindings(positive_entities)
    result: list[dict[str, Any]] = []
    for entity, binding in zip(positive_entities, bindings):
        copied = dict(entity)
        copied["value"] = binding.placeholder
        result.append(copied)
    return result


def replace_entity_placeholders(
    *,
    tagged_text: str,
    entities: Sequence[Mapping[str, Any]],
    positive_entities: Sequence[Mapping[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    """Insert bank values into an LLM placeholder skeleton.

    Replacement is one-pass, so a Value Bank value that happens to contain
    bracketed text cannot trigger a second substitution.
    """

    bindings = build_placeholder_bindings(positive_entities)
    replacements = {binding.placeholder: binding.value for binding in bindings}

    def replace_match(match: re.Match[str]) -> str:
        token = match.group(0)
        return replacements.get(token, token)

    bound_text = _PLACEHOLDER.sub(replace_match, tagged_text)
    bound_entities: list[dict[str, str]] = []
    for raw in entities:
        if not isinstance(raw, Mapping):
            raise ValueError("each generated entity must be an object")
        raw_value = str(raw.get("value", ""))
        bound_entities.append(
            {
                "label": str(raw.get("label", "")),
                "value": _PLACEHOLDER.sub(replace_match, raw_value),
            }
        )

    unresolved = sorted({
        match.group(0)
        for value in [bound_text, *(item["value"] for item in bound_entities)]
        for match in _PLACEHOLDER.finditer(value)
    })
    if unresolved:
        raise ValueError(
            f"generated output contains unknown or unresolved entity placeholders: {unresolved}"
        )
    invalid_bare = sorted({
        value
        for value in [
            *(match.group(1) for match in _ENTITY_TAG.finditer(bound_text)),
            *(item["value"] for item in bound_entities),
        ]
        if _BARE_PLACEHOLDER.fullmatch(value)
    })
    if invalid_bare:
        raise ValueError(
            "generated output contains placeholders without required square "
            f"brackets: {invalid_bare}"
        )
    return bound_text, bound_entities
