import unittest
from pathlib import Path

from pii_factory.bootstrap import build_pipeline
from pii_factory.domain.models import CreateRunRequest, RunConfig, SampleType, TaxonomyLabel, TaxonomySnapshot
from pii_factory.infrastructure.json_taxonomy import JsonTaxonomyParser


class EntryToDataGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline, self.repository, self.events = build_pipeline(offline=True)
        self.request = CreateRunRequest(
            run_name="offline-vi-demo",
            config=RunConfig(
                target_samples=2, language="vi", focus_labels=["TIME"],
                sample_type=SampleType.HARD_NEGATIVE, max_entities=2,
            ),
            taxonomy=TaxonomySnapshot(labels=[TaxonomyLabel(code="TIME", definition="Mốc thời gian cụ thể")]),
        )

    def test_creates_tasks_generates_llm_entities_and_cost(self) -> None:
        run = self.pipeline.create_run(self.request)

        results = self.pipeline.generate_pending(run.run_id, limit=2)

        self.assertEqual(len(self.repository.list_tasks(run.run_id)), 2)
        self.assertEqual(len(results), 2)
        self.assertNotIn("<TIME>", results[0].tagged_text)
        first_seed_event = next(event for event in self.events.list_events() if event.event_type == "seed.validated")
        self.assertIn(first_seed_event.payload["seed_pack"]["decoys"][0]["value"], results[0].tagged_text)
        self.assertEqual(results[0].entities, [])
        self.assertTrue(results[0].seed_validation.valid)
        self.assertTrue(results[0].output_validation.valid)
        self.assertGreater(results[0].token_usage.money_cost, 0)
        self.assertEqual([event.event_type for event in self.events.list_events()].count("data.generated"), 2)
        context = results[0].taxonomy_context_used
        self.assertEqual(context.focus_label.label, "TIME")
        self.assertEqual(context.focus_label.definition, "Mốc thời gian cụ thể")
        generated_event = next(event for event in self.events.list_events() if event.event_type == "data.generated")
        self.assertEqual(
            generated_event.payload["taxonomy_context_used"],
            context.dict(),
        )
    def test_reprocessing_a_generated_task_is_idempotent(self) -> None:
        run = self.pipeline.create_run(self.request)
        first = self.pipeline.generate_pending(run.run_id, limit=1)
        second = self.pipeline.generate_pending(run.run_id, limit=1)

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertNotEqual(first[0].task_id, second[0].task_id)

    def test_uses_registered_taxonomy_version_for_a_run(self) -> None:
        taxonomy = self.pipeline.taxonomy_service.register(self.request.taxonomy)
        request = CreateRunRequest(
            run_name="versioned-taxonomy-run",
            taxonomy_version_id=taxonomy.version_id,
            config=RunConfig(target_samples=1, focus_labels=["TIME"]),
        )

        run = self.pipeline.create_run(request)

        self.assertEqual(run.taxonomy_version_id, taxonomy.version_id)
        self.assertEqual(len(self.repository.list_tasks(run.run_id)), 1)

    def test_generator_uses_structured_json_taxonomy_guidance(self) -> None:
        taxonomy = self.pipeline.taxonomy_service.import_json(
            Path("pii_taxonomy_rules.json")
        )
        config = RunConfig(
            num_samples=1,
            focus_label="API_KEY",
            robin_labels=["EMAIL"],
            robin_selection={
                "min_per_sample": 1,
                "max_per_sample": 1,
            },
            difficulty_distribution={
                "easy": 0.0,
                "medium": 1.0,
                "hard": 0.0,
            },
            sample_type_distribution={
                "positive": 1.0,
                "pure_negative": 0.0,
                "hard_negative": 0.0,
            },
            max_entities={"easy": 2, "medium": 2, "hard": 2},
        )
        run = self.pipeline.create_run(CreateRunRequest(
            taxonomy_version_id=taxonomy.version_id,
            config=config,
        ))

        result = self.pipeline.generate_pending(run.run_id, limit=1)[0]

        context = result.taxonomy_context_used
        self.assertEqual(context.focus_label.label, "API_KEY")
        self.assertEqual(len(context.focus_label.examples), 3)
        self.assertEqual(
            [label.label for label in context.robin_labels],
            ["EMAIL"],
        )
        self.assertEqual(context.robin_labels[0].examples, [])

    def test_mixed_decoy_target_receives_only_selected_hard_negative_examples(
        self,
    ) -> None:
        taxonomy = self.pipeline.taxonomy_service.import_json(
            Path("pii_taxonomy_rules.json")
        )
        run = self.pipeline.create_run(CreateRunRequest(
            taxonomy_version_id=taxonomy.version_id,
            config=RunConfig(
                num_samples=1,
                focus_label="PERSON",
                robin_labels=["EMAIL"],
                robin_selection={
                    "min_per_sample": 1,
                    "max_per_sample": 1,
                },
                difficulty_distribution={
                    "easy": 0.0,
                    "medium": 0.0,
                    "hard": 1.0,
                },
                sample_type_distribution={
                    "positive": 0.0,
                    "pure_negative": 0.0,
                    "hard_negative": 1.0,
                },
                hard_negative={
                    "mode": "mixed_contrastive",
                    "min_decoys": 1,
                    "max_decoys": 1,
                    "max_focus_labels": 2,
                },
                max_entities={
                    "easy": 2,
                    "medium": 2,
                    "hard": 2,
                },
                complexity_limits={
                    "positive": 2,
                    "pure_negative": 1,
                    "hard_negative": 3,
                },
            ),
        ))

        result = self.pipeline.generate_pending(
            run.run_id,
            limit=1,
        )[0]
        context = result.taxonomy_context_used

        self.assertEqual(len(context.decoy_labels), 1)
        self.assertIn(
            context.decoy_labels[0].label,
            {"PERSON", "EMAIL"},
        )
        self.assertEqual(
            len(context.decoy_labels[0].examples),
            3,
        )
        self.assertEqual(
            context.robin_labels[0].examples,
            [],
        )

    def test_parses_the_real_taxonomy_json(self) -> None:
        taxonomy = JsonTaxonomyParser().parse_file(
            Path("pii_taxonomy_rules.json")
        )

        self.assertEqual(len(taxonomy.labels), 44)
        self.assertIn("TIME", [label.code for label in taxonomy.labels])


if __name__ == "__main__":
    unittest.main()
