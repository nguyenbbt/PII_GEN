from __future__ import annotations

import random
from typing import Sequence

from ..domain.models import (
    FewShotExample,
    GenerationTask,
    GenerationTaxonomyContext,
    LabelGenerationContext,
    TaxonomyLabel,
    TaxonomySnapshot,
)


class TaxonomyContextSelector:
    """Build the exact taxonomy guidance required by one generation task."""

    _FEW_SHOT_COUNT = 3

    def select(
        self,
        taxonomy: TaxonomySnapshot,
        task: GenerationTask,
    ) -> GenerationTaxonomyContext:
        labels_by_code = {label.code: label for label in taxonomy.labels}
        focus_code = task.focus_label or task.focus_labels[0]
        robin_codes = (
            task.robin_labels
            if task.focus_label
            else [
                label
                for label in task.focus_labels
                if label != focus_code
            ]
        )
        requested_codes = [focus_code, *robin_codes]
        missing = set(requested_codes) - set(labels_by_code)
        if missing:
            raise ValueError(
                f"task labels absent from taxonomy: {sorted(missing)}"
            )

        focus = labels_by_code[focus_code]
        examples = self._examples_for_sample_type(
            focus,
            str(task.sample_type),
            task.random_seed,
        )
        return GenerationTaxonomyContext(
            taxonomy_version_id=taxonomy.version_id,
            sample_type=task.sample_type,
            focus_label=self._guidance(focus, examples),
            robin_labels=[
                self._guidance(labels_by_code[code])
                for code in robin_codes
            ],
            available_labels=[
                self._guidance(labels_by_code[code])
                for code in (task.annotation_labels or requested_codes)
                if code not in requested_codes
            ],
        )

    @classmethod
    def _examples_for_sample_type(
        cls,
        label: TaxonomyLabel,
        sample_type: str,
        random_seed: int,
    ) -> list[FewShotExample]:
        groups = {
            "positive": label.examples.positive,
            "pure_negative": label.examples.pure_negative,
            "hard_negative": label.examples.hard_negative,
        }
        examples = groups[sample_type]
        if len(examples) <= cls._FEW_SHOT_COUNT:
            return list(examples)

        rng = random.Random(f"{random_seed}:{label.code}:{sample_type}")
        indexes = sorted(
            rng.sample(range(len(examples)), cls._FEW_SHOT_COUNT)
        )
        return [examples[index] for index in indexes]

    @staticmethod
    def _guidance(
        label: TaxonomyLabel,
        examples: Sequence[FewShotExample] = (),
    ) -> LabelGenerationContext:
        return LabelGenerationContext(
            label=label.code,
            definition=label.definition,
            rule="\n".join(label.rules),
            examples=list(examples),
        )
