import unittest

from pii_factory.application.diversity_metrics import DiversityAuditSample, audit_diversity
from pii_factory.bootstrap import build_pipeline
from pii_factory.domain.models import CreateRunRequest, RunConfig, TaxonomyLabel, TaxonomySnapshot


class DiversityEndToEndTests(unittest.TestCase):
    def test_offline_batch_has_balanced_contexts_and_low_exact_template_duplication(self) -> None:
        pipeline, _, events = build_pipeline(offline=True)
        taxonomy = TaxonomySnapshot(labels=[TaxonomyLabel(code="DATE", definition="Ngày cụ thể")])
        config = RunConfig(
            num_samples=40, batch_size=40,
            focus_labels=["DATE"],
            sample_type_distribution={"positive": 1.0, "pure_negative": 0.0, "hard_negative": 0.0},
            max_entities={"easy": 2, "medium": 3, "hard": 4},
            validation={"novelty_mode": "audit"},
            random_seed=174,
        )
        run = pipeline.create_run(CreateRunRequest(taxonomy=taxonomy, config=config))

        results = pipeline.generate_pending(run.run_id, 40)
        seed_events = {
            event.payload["task_id"]: event.payload["seed_pack"]
            for event in events.list_events()
            if event.event_type == "seed.validated"
        }
        samples = [
            DiversityAuditSample(
                tagged_text=result.tagged_text,
                entities=[entity.dict() for entity in result.entities],
                context_frame_id=result.context_frame_id,
                entity_format_variants=result.diversity_profile.entity_format_variants,
                strategy_ids=[item["strategy_id"] for item in seed_events[result.task_id]["decoys"]],
                decoy_values=[item["value"] for item in seed_events[result.task_id]["decoys"]],
                constraints=result.generation_query.constraints,
            )
            for result in results
        ]

        report = audit_diversity(samples)

        self.assertEqual(report.sample_count, 40)
        self.assertLessEqual(report.exact_skeleton_duplicate_rate, 0.05)
        self.assertGreaterEqual(len(report.context_frame_distribution), 6)
        self.assertLessEqual(max(report.context_frame_distribution.values()), 8)


if __name__ == "__main__":
    unittest.main()
