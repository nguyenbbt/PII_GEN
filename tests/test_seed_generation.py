import random
import unittest
from pathlib import Path

from pii_factory.application.seed_generation import (
    HARD_NEGATIVE_SUPPORT,
    ContextFrameSelector,
    ContextSelectionError,
    HardNegativeSeedFactory,
    PositiveSeedFactory,
    PureNegativeContentFactory,
)
from pii_factory.application.value_bank import ValueBankEntityProvider
from pii_factory.application.validators import SeedPackValidator
from pii_factory.bootstrap import build_pipeline
from pii_factory.domain.models import (
    GenerationTask,
    HardNegativeConfig,
    SampleType,
    TaxonomyLabel,
    ValidationConfig,
)


def task(sample_type: str, labels: list[str]) -> GenerationTask:
    return GenerationTask(
        task_id=f"task-{sample_type}", run_id="run-1", sequence_no=1, language="vi",
        focus_labels=labels, difficulty="hard", sample_type=sample_type,
        max_entities=6, max_attempts=3, random_seed=42,
    )


class SeedGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = ValueBankEntityProvider("PII_Value_Bank")
        self.selector = ContextFrameSelector()
        self.hard_config = HardNegativeConfig()
        self.validator = SeedPackValidator(self.hard_config, ValidationConfig())

    def test_positive_has_unique_seed_for_every_focus_label(self) -> None:
        labels = [TaxonomyLabel(code=code, definition=code) for code in ["ADDRESS", "DATE", "EMAIL"]]
        pack = PositiveSeedFactory(self.provider, self.selector).build(
            task("positive", [label.code for label in labels]), labels, random.Random(42)
        )

        self.assertEqual({seed.label for seed in pack.positive_entities}, {"ADDRESS", "DATE", "EMAIL"})
        self.assertEqual(len({seed.value.casefold() for seed in pack.positive_entities}), 3)
        self.assertTrue(self.validator.validate(pack, [label.code for label in labels], labels).valid)
        address = next(seed.value for seed in pack.positive_entities if seed.label == "ADDRESS")
        self.assertNotRegex(address, r"Jane|John|Smith|County|Street|Avenue")

    def test_pure_negative_factory_never_builds_pii_or_decoys(self) -> None:
        taxonomy = [TaxonomyLabel(code="EMAIL", definition="email")]
        pack = PureNegativeContentFactory(self.selector).build(
            task("pure_negative", ["EMAIL"]), taxonomy, random.Random(42)
        )

        self.assertEqual(pack.positive_entities, [])
        self.assertEqual(pack.decoys, [])
        self.assertIsNotNone(pack.content_seeds)
        self.assertTrue(self.validator.validate(pack, ["EMAIL"], taxonomy).valid)

    def test_hard_negative_uses_one_supported_label_aware_decoy(self) -> None:
        taxonomy = [TaxonomyLabel(code="DATE", definition="DATE")]
        pack = HardNegativeSeedFactory(self.provider, self.selector, self.hard_config).build(
            task("hard_negative", ["DATE"]), taxonomy, random.Random(42)
        )

        self.assertEqual(len(pack.decoys), 1)
        self.assertEqual(pack.decoys[0].target_label, "DATE")
        self.assertEqual(pack.positive_entities, [])
        self.assertEqual(pack.hard_negative_mode, "decoy_only")
        self.assertTrue(HARD_NEGATIVE_SUPPORT["DATE"])
        self.assertTrue(HARD_NEGATIVE_SUPPORT["ADDRESS"])
        self.assertNotRegex(pack.decoys[0].value, r"address-\d+-beta")
        self.assertTrue(self.validator.validate(pack, [label.code for label in taxonomy], taxonomy).valid)

    def test_hard_negative_registry_uses_exactly_all_taxonomy_labels(self) -> None:
        pipeline, _, _ = build_pipeline(offline=True)
        taxonomy = pipeline.taxonomy_service.import_json(
            Path("pii_taxonomy_rules.json")
        )

        self.assertEqual(set(HARD_NEGATIVE_SUPPORT), {label.code for label in taxonomy.labels})
        self.assertTrue(all(HARD_NEGATIVE_SUPPORT.values()))

    def test_every_taxonomy_strategy_builds_a_collision_free_seed_pack(self) -> None:
        pipeline, _, _ = build_pipeline(offline=True)
        taxonomy = pipeline.taxonomy_service.import_json(
            Path("pii_taxonomy_rules.json")
        )
        taxonomy_codes = {label.code for label in taxonomy.labels}
        factory = HardNegativeSeedFactory(self.provider, self.selector, self.hard_config)

        for index, label in enumerate(taxonomy.labels, start=1):
            with self.subTest(label=label.code):
                pack = factory.build(
                    task("hard_negative", [label.code]), taxonomy.labels, random.Random(index)
                )
                result = self.validator.validate(pack, [label.code], taxonomy.labels)
                self.assertTrue(result.valid, [issue.dict() for issue in result.issues])
                self.assertEqual(pack.decoys[0].target_label, label.code)
                self.assertTrue(set(pack.decoys[0].negative_labels).issubset(taxonomy_codes))

    def test_mixed_contrastive_mode_remains_explicitly_available(self) -> None:
        config = HardNegativeConfig(mode="mixed_contrastive")
        taxonomy = [TaxonomyLabel(code="DATE", definition="DATE")]
        pack = HardNegativeSeedFactory(self.provider, self.selector, config).build(
            task("hard_negative", ["DATE"]), taxonomy, random.Random(42)
        )

        self.assertEqual(pack.hard_negative_mode, "mixed_contrastive")
        self.assertEqual([seed.label for seed in pack.positive_entities], ["DATE"])
        self.assertEqual(pack.decoys[0].target_label, "DATE")
        self.assertTrue(SeedPackValidator(config, ValidationConfig()).validate(
            pack, ["DATE"], taxonomy
        ).valid)

    def test_vietnamese_address_from_bank_is_reproducible(self) -> None:
        first = self.provider.generate("ADDRESS", "vi", random.Random(101))
        repeated = self.provider.generate("ADDRESS", "vi", random.Random(101))
        different = self.provider.generate("ADDRESS", "vi", random.Random(202))

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)
        self.assertIn(first, self.provider.values_for("vi", "ADDRESS"))

    def test_unknown_label_routes_to_context_scope(self) -> None:
        with self.assertRaises(ContextSelectionError) as captured:
            self.selector.select(["UNKNOWN_LABEL"], random.Random(1))
        self.assertEqual(captured.exception.scope, "CONTEXT")


if __name__ == "__main__":
    unittest.main()
