import json
import unittest

from pii_factory.bootstrap import build_pipeline
from pii_factory.application.services import OutputValidationError
from pii_factory.domain.models import (
    CreateRunRequest, DeterministicValidationResult, RunConfig, TaxonomyLabel,
    TaxonomySnapshot, ValidationIssue,
)


def fixed_config(sample_type: str) -> RunConfig:
    return RunConfig(
        num_samples=1,
        batch_size=1,
        focus_labels=["DATE"],
        sample_type_distribution={
            "positive": float(sample_type == "positive"),
            "pure_negative": float(sample_type == "pure_negative"),
            "hard_negative": float(sample_type == "hard_negative"),
        },
        max_entities={"easy": 2, "medium": 2, "hard": 2},
    )


class PipelineSeedStrategyTests(unittest.TestCase):
    def _pipeline_for(self, sample_type: str):
        pipeline, repository, events = build_pipeline(offline=True)
        taxonomy = TaxonomySnapshot(labels=[TaxonomyLabel(code="DATE", definition="Ngày cụ thể")])
        run = pipeline.create_run(CreateRunRequest(taxonomy=taxonomy, config=fixed_config(sample_type)))
        return pipeline, repository, events, run

    def test_positive_pipeline_uses_faker_seed(self) -> None:
        pipeline, _, _, run = self._pipeline_for("positive")
        result = pipeline.generate_pending(run.run_id, 1)[0]
        self.assertEqual(result.entities[0].value, result.tagged_text.split("<DATE>")[1].split("</DATE>")[0])
        self.assertTrue(result.seed_validation.valid)

    def test_pure_negative_pipeline_has_no_entities(self) -> None:
        pipeline, _, events, run = self._pipeline_for("pure_negative")
        result = pipeline.generate_pending(run.run_id, 1)[0]
        seed_event = next(event for event in events.list_events() if event.event_type == "seed.validated")
        self.assertEqual(result.entities, [])
        self.assertEqual(seed_event.payload["seed_pack"]["positive_entities"], [])
        self.assertEqual(seed_event.payload["seed_pack"]["decoys"], [])

    def test_hard_negative_pipeline_has_one_untagged_decoy(self) -> None:
        pipeline, _, events, run = self._pipeline_for("hard_negative")
        result = pipeline.generate_pending(run.run_id, 1)[0]
        seed_event = next(event for event in events.list_events() if event.event_type == "seed.validated")
        decoy = seed_event.payload["seed_pack"]["decoys"][0]
        self.assertIn(decoy["value"], result.tagged_text)
        self.assertEqual(result.entities, [])
        self.assertEqual(seed_event.payload["seed_pack"]["positive_entities"], [])
        self.assertEqual(seed_event.payload["seed_pack"]["hard_negative_mode"], "decoy_only")
        self.assertEqual(len(seed_event.payload["seed_pack"]["decoys"]), 1)

    def test_text_retry_keeps_seed_pack_and_records_text_scope(self) -> None:
        class RetryClient:
            def __init__(self):
                self.calls = 0

            def generate(self, messages):
                self.calls += 1
                content = messages[-1]["content"].split("```json\n", 1)[1].split("\n```", 1)[0]
                payload = json.loads(content)
                if self.calls == 1:
                    return "Hồ sơ chưa có ngày.", [], 10, 5, 15
                seed = payload["positive_entities"][0]
                text = f"Hồ sơ hẹn vào <{seed['label']}>{seed['value']}</{seed['label']}>."
                return text, [{"label": seed["label"], "value": seed["value"]}], 10, 5, 15

        pipeline, _, events, run = self._pipeline_for("positive")
        client = RetryClient()
        pipeline.generator.client = client
        result = pipeline.generate_pending(run.run_id, 1)[0]
        rejected = next(event for event in events.list_events() if event.event_type == "data.generation.rejected")
        self.assertEqual(client.calls, 2)
        self.assertEqual(result.attempt_no, 2)
        self.assertEqual(rejected.payload["seed_pack_id"], result.seed_pack_id)
        self.assertEqual(rejected.payload["regeneration_scope"], "TEXT")

    def test_seed_scoped_retry_builds_a_new_seed_pack(self) -> None:
        pipeline, _, _, run = self._pipeline_for("positive")
        original_generate = pipeline.generator.generate
        seed_pack_ids: list[str] = []

        def fail_first_seed(request, validator):
            seed_pack_ids.append(request.seed_pack.seed_pack_id)
            if len(seed_pack_ids) == 1:
                raise OutputValidationError(DeterministicValidationResult(
                    valid=False,
                    issues=[ValidationIssue(
                        type="duplicate_entity_value", scope="SEEDS",
                        reason="entity value already exists",
                    )],
                ))
            return original_generate(request, validator)

        pipeline.generator.generate = fail_first_seed

        result = pipeline.generate_pending(run.run_id, 1)[0]

        self.assertEqual(result.attempt_no, 2)
        self.assertNotEqual(seed_pack_ids[0], seed_pack_ids[1])


if __name__ == "__main__":
    unittest.main()
