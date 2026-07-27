import random
import unittest

from pii_factory.application.seed_generation import ContextFrameSelector, PureNegativeContentFactory
from pii_factory.domain.models import GenerationTask, TaxonomyLabel


class PureNegativeContentDiversityTests(unittest.TestCase):
    def test_content_seed_vocabulary_varies_across_tasks(self) -> None:
        factory = PureNegativeContentFactory(ContextFrameSelector())
        taxonomy = [TaxonomyLabel(code="DATE", definition="Ngày cụ thể")]
        variants = set()

        for seed in range(20):
            task = GenerationTask(
                task_id=f"task-{seed}", run_id="run", sequence_no=seed + 1, language="vi",
                focus_labels=["DATE"], difficulty="medium", sample_type="pure_negative",
                max_entities=2, max_attempts=3, random_seed=seed,
            )
            pack = factory.build(task, taxonomy, random.Random(seed))
            content = pack.content_seeds
            variants.add((tuple(content.generic_roles), tuple(content.actions), tuple(content.objects)))

        self.assertGreaterEqual(len(variants), 10)


if __name__ == "__main__":
    unittest.main()
