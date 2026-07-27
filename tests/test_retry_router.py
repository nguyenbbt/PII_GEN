import random
import unittest

from pii_factory.application.retry import RegenerationRouter
from pii_factory.application.seed_generation import ContextFrameSelector, build_sample_type_router
from pii_factory.domain.models import GenerationTask, HardNegativeConfig, TaxonomyLabel, ValueBankConfig


class RetryRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seed_router = build_sample_type_router(
            ValueBankConfig(),
            HardNegativeConfig(),
        )
        self.router = RegenerationRouter(self.seed_router, ContextFrameSelector())
        self.task = GenerationTask(
            task_id="retry-task", run_id="run", sequence_no=1, language="vi",
            focus_labels=["DATE"], difficulty="medium", sample_type="positive",
            max_entities=2, max_attempts=3, random_seed=42,
        )
        self.taxonomy = [TaxonomyLabel(code="DATE", definition="Ngày cụ thể")]
        self.current = self.seed_router.build_seed_pack(self.task, self.taxonomy, random.Random(42))

    def test_text_retry_keeps_seed_and_context(self) -> None:
        routed = self.router.route(
            "TEXT", task=self.task, taxonomy=self.taxonomy, rng=random.Random(43), current=self.current
        )
        self.assertIs(routed, self.current)

    def test_seed_retry_creates_new_seed_pack(self) -> None:
        routed = self.router.route(
            "SEEDS", task=self.task, taxonomy=self.taxonomy, rng=random.Random(43), current=self.current
        )
        self.assertNotEqual(routed.seed_pack_id, self.current.seed_pack_id)

    def test_context_retry_keeps_seed_pack_but_changes_frame(self) -> None:
        routed = self.router.route(
            "CONTEXT", task=self.task, taxonomy=self.taxonomy, rng=random.Random(43), current=self.current
        )
        self.assertEqual(routed.seed_pack_id, self.current.seed_pack_id)
        self.assertEqual(routed.positive_entities, self.current.positive_entities)
        self.assertNotEqual(routed.context_frame.frame_id, self.current.context_frame.frame_id)


if __name__ == "__main__":
    unittest.main()
