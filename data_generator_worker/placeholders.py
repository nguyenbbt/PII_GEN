from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_PLACEHOLDER = re.compile(r"\[([A-Z][A-Z0-9_]*)_([1-9][0-9]*)\]")
_BARE_PLACEHOLDER = re.compile(r"[A-Z][A-Z0-9_]*_[1-9][0-9]*")
_ENTITY_TAG = re.compile(r"<([A-Z][A-Z0-9_]*)>([^<>]+)</\1>")

logger = logging.getLogger(__name__)


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
    language: str = "vi",
) -> tuple[str, list[dict[str, str]]]:
    """Insert bank values into an LLM placeholder skeleton.

    Replacement is one-pass, so a Value Bank value that happens to contain
    bracketed text cannot trigger a second substitution.
    """

    bindings = build_placeholder_bindings(positive_entities)
    by_placeholder = {binding.placeholder: binding for binding in bindings}
    by_bare_placeholder = {
        binding.placeholder[1:-1]: binding
        for binding in bindings
    }
    for raw in entities:
        if not isinstance(raw, Mapping):
            raise ValueError("each generated entity must be an object")

    repaired_bare = 0
    generic_references = 0

    def bind_tag(match: re.Match[str]) -> str:
        nonlocal repaired_bare, generic_references
        label, content = match.group(1), match.group(2)
        binding = by_placeholder.get(content)
        if binding is None and content in by_bare_placeholder:
            binding = by_bare_placeholder[content]
            repaired_bare += 1
        if binding is not None:
            if binding.label != label:
                raise ValueError(
                    f"placeholder {content!r} is wrapped with {label!r}, "
                    f"expected {binding.label!r}"
                )
            return f"<{label}>{binding.value}</{label}>"

        unknown = _PLACEHOLDER.search(content)
        if unknown is not None:
            generic_references += 1
            return _generic_reference(unknown.group(1), language)
        return match.group(0)

    bound_text = _ENTITY_TAG.sub(bind_tag, tagged_text)

    def replace_remaining_placeholder(match: re.Match[str]) -> str:
        nonlocal generic_references
        generic_references += 1
        return _generic_reference(match.group(1), language)

    bound_text = _PLACEHOLDER.sub(
        replace_remaining_placeholder,
        bound_text,
    )

    def replace_remaining_bare(match: re.Match[str]) -> str:
        nonlocal generic_references
        token = match.group(0)
        binding = by_bare_placeholder.get(token)
        if binding is None:
            return token
        generic_references += 1
        return _generic_reference(binding.label, language)

    bound_text = _BARE_PLACEHOLDER.sub(
        replace_remaining_bare,
        bound_text,
    )
    bound_entities = [
        {"label": match.group(1), "value": match.group(2)}
        for match in _ENTITY_TAG.finditer(bound_text)
    ]
    if repaired_bare:
        logger.warning(
            "normalized %s known bare placeholder(s) inside entity tags",
            repaired_bare,
        )
    if generic_references:
        logger.warning(
            "replaced %s untagged or unknown placeholder occurrence(s) "
            "with generic %s references",
            generic_references,
            _language_code(language),
        )
    if len(bound_entities) != len(entities):
        logger.warning(
            "synchronized entity metadata from final tags old_count=%s "
            "new_count=%s",
            len(entities),
            len(bound_entities),
        )
    return bound_text, bound_entities


def _generic_reference(label: str, language: str) -> str:
    code = _language_code(language)
    references = {
        "vi": {
            "PERSON": "người liên quan",
            "ORGANIZATION": "đơn vị liên quan",
            "DATE": "thời điểm liên quan",
            "ACCOUNT_ID": "một tài khoản nội bộ",
            "STAFF_ID": "nhân viên phụ trách",
        },
        "en": {
            "PERSON": "the referenced person",
            "ORGANIZATION": "the relevant organization",
            "DATE": "the relevant date",
            "ACCOUNT_ID": "a specific internal account",
            "STAFF_ID": "the responsible staff member",
        },
        "de": {
            "PERSON": "die betreffende Person",
            "ORGANIZATION": "die betreffende Organisation",
            "DATE": "das betreffende Datum",
            "ACCOUNT_ID": "ein bestimmtes internes Konto",
            "STAFF_ID": "das zuständige Teammitglied",
        },
    }
    defaults = {
        "vi": "một tham chiếu nội bộ",
        "en": "an internal reference",
        "de": "eine interne Referenz",
    }
    return references.get(code, {}).get(
        label,
        defaults.get(code, defaults["en"]),
    )


def _language_code(language: str) -> str:
    normalized = str(language).strip().casefold().replace("_", "-")
    aliases = {
        "vietnamese": "vi",
        "english": "en",
        "german": "de",
    }
    return aliases.get(normalized, normalized.split("-", 1)[0])
