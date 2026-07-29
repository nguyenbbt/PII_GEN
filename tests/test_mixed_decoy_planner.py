import random
import unittest
from pathlib import Path

from pii_factory.application.mixed_decoy_planner import (
    MixedContrastiveDecoyPlanner,
)
from pii_factory.domain.models import (
    GenerationTask,
    HardNegativeConfig,
    PositiveEntitySeed,
)
from pii_factory.infrastructure.json_taxonomy import JsonTaxonomyParser


class MixedContrastiveDecoyPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.taxonomy = JsonTaxonomyParser().parse_file(
            Path("pii_taxonomy_rules.json")
        ).labels
        cls.labels = {label.code: label for label in cls.taxonomy}

    @staticmethod
    def _task(ordinal: int = 1) -> GenerationTask:
        return GenerationTask(
            task_id=f"mixed-{ordinal}",
            run_id="run-mixed",
            sequence_no=ordinal,
            hard_negative_ordinal=ordinal,
            language="vi",
            focus_labels=["PERSON", "EMAIL"],
            focus_label="PERSON",
            robin_labels=["EMAIL"],
            difficulty="hard",
            sample_type="hard_negative",
            sample_structure={"type": "contract"},
            max_entities=6,
            max_attempts=3,
            random_seed=42,
        )

    @staticmethod
    def _positive_entities() -> list[PositiveEntitySeed]:
        return [
            PositiveEntitySeed(
                label="PERSON",
                value="Nguyễn Minh Khôi",
                semantic_role="requester_name",
            ),
            PositiveEntitySeed(
                label="EMAIL",
                value="khoi.nguyen@example.com",
                semantic_role="contact_email",
            ),
        ]

    def test_plan_is_reproducible_and_links_decoy_to_positive_anchor(
        self,
    ) -> None:
        planner = MixedContrastiveDecoyPlanner(
            HardNegativeConfig(mode="mixed_contrastive")
        )

        first = planner.plan(
            task=self._task(),
            taxonomy=self.taxonomy,
            positive_entities=self._positive_entities(),
            rng=random.Random(73),
            count=1,
        )
        repeated = planner.plan(
            task=self._task(),
            taxonomy=self.taxonomy,
            positive_entities=self._positive_entities(),
            rng=random.Random(73),
            count=1,
        )

        self.assertEqual(first, repeated)
        self.assertEqual(len(first.decoys), 1)
        decoy = first.decoys[0]
        self.assertIsNotNone(decoy.realization_plan)
        assert decoy.realization_plan is not None
        self.assertIn(
            decoy.realization_plan.anchor_label,
            {"PERSON", "EMAIL"},
        )
        self.assertIn(
            decoy.realization_plan.source_example_ids[0],
            {
                example.id
                for example in self.labels[decoy.target_label]
                .examples.hard_negative
            },
        )
        self.assertIn(
            decoy.realization_plan.blueprint_id,
            {
                blueprint.id
                for blueprint in self.labels[decoy.target_label]
                .decoy_blueprints
            },
        )
        self.assertIn(
            first.context_frame.domain,
            decoy.realization_plan.compatible_domains,
        )

    def test_technical_blueprints_respect_prefix_hard_cap(self) -> None:
        config = HardNegativeConfig(
            mode="mixed_contrastive",
            technical_decoy_max_ratio=0.15,
        )
        planner = MixedContrastiveDecoyPlanner(config)
        technical = 0

        for ordinal in range(1, 41):
            result = planner.plan(
                task=self._task(ordinal),
                taxonomy=self.taxonomy,
                positive_entities=self._positive_entities(),
                rng=random.Random(ordinal * 101),
                count=1,
            )
            technical += int(result.decoys[0].family == "technical_schema")
            self.assertLessEqual(
                technical,
                int(ordinal * config.technical_decoy_max_ratio),
            )
        self.assertEqual(technical, 4)

    def test_selected_blueprint_is_compatible_with_structure_and_context(
        self,
    ) -> None:
        planner = MixedContrastiveDecoyPlanner(
            HardNegativeConfig(mode="mixed_contrastive")
        )

        for ordinal in range(1, 21):
            result = planner.plan(
                task=self._task(ordinal),
                taxonomy=self.taxonomy,
                positive_entities=self._positive_entities(),
                rng=random.Random(ordinal),
                count=1,
            )
            decoy = result.decoys[0]
            assert decoy.realization_plan is not None
            blueprint = next(
                item
                for item in self.labels[decoy.target_label].decoy_blueprints
                if item.id == decoy.realization_plan.blueprint_id
            )
            self.assertIn("contract", blueprint.compatible_structures)
            if blueprint.compatible_domains:
                self.assertIn(
                    result.context_frame.domain,
                    blueprint.compatible_domains,
                )

    def test_seed_retry_can_exclude_the_current_blueprint(self) -> None:
        planner = MixedContrastiveDecoyPlanner(
            HardNegativeConfig(mode="mixed_contrastive")
        )
        first = planner.plan(
            task=self._task(),
            taxonomy=self.taxonomy,
            positive_entities=self._positive_entities(),
            rng=random.Random(73),
            count=1,
        )
        current_id = first.decoys[0].strategy_id

        replacement = planner.plan(
            task=self._task(),
            taxonomy=self.taxonomy,
            positive_entities=self._positive_entities(),
            rng=random.Random(73),
            count=1,
            excluded_blueprint_ids={current_id},
        )

        self.assertNotEqual(
            replacement.decoys[0].strategy_id,
            current_id,
        )

    def test_multiple_decoys_never_reuse_the_same_surface(self) -> None:
        result = MixedContrastiveDecoyPlanner(
            HardNegativeConfig(
                mode="mixed_contrastive",
                min_decoys=2,
                max_decoys=2,
            )
        ).plan(
            task=self._task(),
            taxonomy=self.taxonomy,
            positive_entities=self._positive_entities(),
            rng=random.Random(91),
            count=2,
        )

        surfaces = [
            decoy.value.casefold()
            for decoy in result.decoys
        ]
        self.assertEqual(len(surfaces), 2)
        self.assertEqual(len(set(surfaces)), 2)


if __name__ == "__main__":
    unittest.main()
