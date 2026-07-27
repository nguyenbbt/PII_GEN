from __future__ import annotations

from difflib import SequenceMatcher
from typing import Sequence

from ..domain.models import DataGenerationResult, GeneratedEntity, NoveltyAssessment, ValidationIssue
from .diversity_metrics import sentence_skeleton


DEFAULT_CATEGORICAL_LABELS = (
    "CARD_ISSUER", "ETHNICITY", "GENDER", "MARITAL", "NATIONALITY", "RELIGION",
)


class NoveltyGuard:
    def __init__(
        self,
        *,
        near_duplicate_threshold: float = 0.9,
        recent_window: int = 100,
        max_feedback: int = 3,
        categorical_labels: Sequence[str] = DEFAULT_CATEGORICAL_LABELS,
    ) -> None:
        self.near_duplicate_threshold = near_duplicate_threshold
        self.recent_window = recent_window
        self.max_feedback = max_feedback
        self.categorical_labels = set(categorical_labels)

    def assess(
        self,
        *,
        tagged_text: str,
        entities: Sequence[GeneratedEntity],
        decoy_values: Sequence[str],
        previous_results: Sequence[DataGenerationResult],
    ) -> NoveltyAssessment:
        candidate_skeleton = sentence_skeleton(tagged_text, decoy_values)
        recent_results = list(previous_results[-self.recent_window:])
        previous_skeletons = [
            result.sentence_skeleton or sentence_skeleton(result.tagged_text)
            for result in recent_results
        ]
        similarities = [
            (SequenceMatcher(None, candidate_skeleton, previous).ratio(), previous)
            for previous in previous_skeletons
        ]
        similarities.sort(reverse=True)
        issues = self._entity_issues(entities, recent_results)
        if similarities:
            highest, _ = similarities[0]
            if highest == 1.0:
                issues.append(ValidationIssue(
                    type="duplicate_sentence_skeleton",
                    scope="TEXT",
                    reason="sentence skeleton duplicates an accepted sample in this run",
                ))
            elif highest >= self.near_duplicate_threshold:
                issues.append(ValidationIssue(
                    type="near_duplicate_sentence",
                    scope="TEXT",
                    reason=f"sentence skeleton similarity {highest:.3f} exceeds the run threshold",
                ))
        return NoveltyAssessment(
            valid=not issues,
            sentence_skeleton=candidate_skeleton,
            max_similarity=similarities[0][0] if similarities else 0.0,
            nearest_skeletons=[skeleton for _, skeleton in similarities[:self.max_feedback]],
            issues=issues,
        )

    def _entity_issues(
        self,
        entities: Sequence[GeneratedEntity],
        previous_results: Sequence[DataGenerationResult],
    ) -> list[ValidationIssue]:
        previous_pairs = {
            (entity.label, entity.value.strip().casefold())
            for result in previous_results
            for entity in result.entities
            if entity.label not in self.categorical_labels
        }
        return [
            ValidationIssue(
                type="duplicate_entity_value",
                scope="SEEDS",
                reason="entity value was already accepted for this label in the current run",
                label=entity.label,
                value=entity.value,
            )
            for entity in entities
            if entity.label not in self.categorical_labels
            and (entity.label, entity.value.strip().casefold()) in previous_pairs
        ]
