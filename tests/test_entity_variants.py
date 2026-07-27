import random
import unittest

from pii_factory.application.entity_variants import FakerEntityProvider
from pii_factory.application.seed_generation import ContextFrameSelector, PositiveSeedFactory
from pii_factory.domain.models import GenerationTask, TaxonomyLabel


FOCUS_LABELS = (
    "PERSON", "PHONE", "EMAIL", "ADDRESS", "DATE", "TIME", "MONEY", "URL",
    "IP", "CARD_NUMBER", "PLATE", "PASSPORT", "MEDICAL_INFO", "LOCATION",
)


class EntityVariantTests(unittest.TestCase):
    def test_focus_labels_have_multiple_reproducible_format_variants(self) -> None:
        provider = FakerEntityProvider("vi_VN")

        for label in FOCUS_LABELS:
            with self.subTest(label=label):
                first = [provider.generate_with_variant(label, random.Random(seed)) for seed in range(80)]
                repeated = [provider.generate_with_variant(label, random.Random(seed)) for seed in range(80)]
                self.assertEqual(first, repeated)
                self.assertGreaterEqual(len({item.format_variant for item in first}), 2)

    def test_open_ended_focus_labels_have_high_value_uniqueness(self) -> None:
        provider = FakerEntityProvider("vi_VN")
        open_ended = ("PHONE", "EMAIL", "ADDRESS", "DATE", "TIME", "MONEY", "URL", "IP",
                      "CARD_NUMBER", "PLATE", "PASSPORT")

        for label in open_ended:
            with self.subTest(label=label):
                values = {
                    provider.generate_with_variant(label, random.Random(seed)).value.casefold()
                    for seed in range(100)
                }
                self.assertGreaterEqual(len(values), 90)

    def test_positive_seed_records_the_selected_format_variant(self) -> None:
        task = GenerationTask(
            task_id="variant-task", run_id="run", sequence_no=1, language="vi",
            focus_labels=["TIME"], difficulty="medium", sample_type="positive",
            max_entities=2, max_attempts=3, random_seed=42,
        )
        taxonomy = [TaxonomyLabel(code="TIME", definition="Thời gian cụ thể")]

        pack = PositiveSeedFactory(FakerEntityProvider("vi_VN"), ContextFrameSelector()).build(
            task, taxonomy, random.Random(42)
        )

        seed = pack.positive_entities[0]
        self.assertEqual(task.diversity_profile.entity_format_variants["TIME"], seed.format_variant)
        self.assertNotEqual(seed.format_variant, "default")


if __name__ == "__main__":
    unittest.main()
