from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Sequence

from pydantic import BaseModel, Field


_TAGGED_ENTITY = re.compile(
    r"<(?P<label>[A-Za-z][A-Za-z0-9_]*)>.*?</(?P=label)>",
    re.DOTALL,
)
_WHITESPACE = re.compile(r"\s+")


class DiversityAuditSample(BaseModel):
    tagged_text: str
    entities: List[Dict[str, str]] = Field(default_factory=list)
    decoy_values: List[str] = Field(default_factory=list)
    context_frame_id: str = ""
    strategy_ids: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    entity_format_variants: Dict[str, str] = Field(default_factory=dict)


class DiversityReport(BaseModel):
    sample_count: int
    entity_unique_ratio: Dict[str, float]
    categorical_value_coverage: Dict[str, int]
    exact_skeleton_duplicate_rate: float
    near_duplicate_rate: float
    context_frame_distribution: Dict[str, int]
    strategy_distribution: Dict[str, int]
    constraint_distribution: Dict[str, int]
    format_variant_distribution: Dict[str, int]


def sentence_skeleton(tagged_text: str, decoy_values: Sequence[str] = ()) -> str:
    skeleton = _TAGGED_ENTITY.sub(
        lambda match: f"<{match.group('label').casefold()}>",
        tagged_text,
    )
    for value in sorted(decoy_values, key=len, reverse=True):
        if value:
            skeleton = skeleton.replace(value, "<decoy>")
    return _WHITESPACE.sub(" ", skeleton.casefold()).strip()


def audit_diversity(
    samples: Sequence[DiversityAuditSample],
    *,
    near_duplicate_threshold: float = 0.88,
    categorical_labels: Sequence[str] = (),
    comparison_window: int = 200,
) -> DiversityReport:
    entity_values: dict[str, list[str]] = defaultdict(list)
    frame_counts: Counter[str] = Counter()
    strategy_counts: Counter[str] = Counter()
    constraint_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    skeletons: list[str] = []

    for sample in samples:
        skeletons.append(sentence_skeleton(sample.tagged_text, sample.decoy_values))
        frame_counts.update([sample.context_frame_id] if sample.context_frame_id else [])
        strategy_counts.update(sample.strategy_ids)
        constraint_counts.update(sample.constraints)
        format_counts.update(f"{label}:{variant}" for label, variant in sample.entity_format_variants.items())
        for entity in sample.entities:
            entity_values[str(entity["label"])].append(str(entity["value"]).strip())

    skeleton_counts = Counter(skeletons)
    duplicate_count = sum(count - 1 for count in skeleton_counts.values())
    near_duplicate_count = sum(
        any(
            candidate != skeleton
            and SequenceMatcher(None, candidate, skeleton).ratio() >= near_duplicate_threshold
            for candidate in skeletons[max(0, index - comparison_window):index]
        )
        for index, skeleton in enumerate(skeletons)
    )
    sample_count = len(samples)
    categorical = set(categorical_labels)

    return DiversityReport(
        sample_count=sample_count,
        entity_unique_ratio={
            label: len(set(values)) / len(values)
            for label, values in entity_values.items()
            if label not in categorical
        },
        categorical_value_coverage={
            label: len(set(values))
            for label, values in entity_values.items()
            if label in categorical
        },
        exact_skeleton_duplicate_rate=duplicate_count / sample_count if sample_count else 0.0,
        near_duplicate_rate=near_duplicate_count / sample_count if sample_count else 0.0,
        context_frame_distribution=dict(frame_counts),
        strategy_distribution=dict(strategy_counts),
        constraint_distribution=dict(constraint_counts),
        format_variant_distribution=dict(format_counts),
    )
