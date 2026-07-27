import random
import unittest

from pii_factory.application.seed_generation import (
    HARD_NEGATIVE_STRATEGIES,
    ContextFrameSelector,
    FakerEntityProvider,
    HardNegativeSeedFactory,
)
from pii_factory.application.validators import SeedPackValidator
from pii_factory.domain.models import GenerationTask, HardNegativeConfig, TaxonomyLabel, ValidationConfig


FOCUS_LABELS = (
    "PERSON", "PHONE", "EMAIL", "ADDRESS", "DATE", "TIME", "MONEY", "URL",
    "IP", "CARD_NUMBER", "PLATE", "PASSPORT", "MEDICAL_INFO", "LOCATION",
)


class HardNegativeDiversityTests(unittest.TestCase):
    def test_focus_labels_have_multiple_strategy_families(self) -> None:
        for label in HARD_NEGATIVE_STRATEGIES:
            with self.subTest(label=label):
                strategies = HARD_NEGATIVE_STRATEGIES[label]
                self.assertGreaterEqual(len(strategies), 3)
                self.assertGreaterEqual(len({strategy.family for strategy in strategies}), 2)

    def test_seeded_selection_covers_all_registered_focus_strategies(self) -> None:
        config = HardNegativeConfig()
        factory = HardNegativeSeedFactory(FakerEntityProvider(), ContextFrameSelector(), config)
        validator = SeedPackValidator(config, ValidationConfig())

        for label in FOCUS_LABELS:
            taxonomy = [TaxonomyLabel(code=label, definition=label)]
            seen: set[str] = set()
            for seed in range(80):
                task = GenerationTask(
                    task_id=f"{label}-{seed}", run_id="run", sequence_no=seed + 1, language="vi",
                    focus_labels=[label], difficulty="hard", sample_type="hard_negative",
                    max_entities=2, max_attempts=3, random_seed=seed,
                )
                pack = factory.build(task, taxonomy, random.Random(seed))
                self.assertTrue(validator.validate(pack, [label], taxonomy).valid)
                seen.add(pack.decoys[0].strategy_id)
            self.assertEqual(seen, {strategy.strategy_id for strategy in HARD_NEGATIVE_STRATEGIES[label]})


if __name__ == "__main__":
    unittest.main()
