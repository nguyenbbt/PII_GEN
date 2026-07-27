import unittest
from collections import Counter
from pathlib import Path

from pii_factory.application.diversity import DiversityPlanner
from pii_factory.bootstrap import build_pipeline
from pii_factory.domain.models import CreateRunRequest, RunConfig, TaxonomyLabel, TaxonomySnapshot


class DiversityPlannerTests(unittest.TestCase):
    def test_same_seed_is_reproducible_and_different_seed_changes_plan(self) -> None:
        def profiles(seed: int):
            planner = DiversityPlanner(seed)
            return [planner.plan(["DATE"]).dict() for _ in range(12)]

        self.assertEqual(profiles(42), profiles(42))
        self.assertNotEqual(profiles(42), profiles(43))

    def test_quota_axes_cover_every_option_before_repeating(self) -> None:
        planner = DiversityPlanner(42)
        profiles = [planner.plan(["DATE"]) for _ in range(12)]

        self.assertGreaterEqual(len({profile.speaker_role for profile in profiles}), 4)
        self.assertGreaterEqual(len({profile.document_structure for profile in profiles}), 4)
        self.assertGreaterEqual(len({profile.language_register for profile in profiles}), 4)
        context_counts = {
            frame_id: sum(profile.context_frame_id == frame_id for profile in profiles)
            for frame_id in {profile.context_frame_id for profile in profiles}
        }
        self.assertLessEqual(max(context_counts.values()), min(context_counts.values()) + 1)

    def test_weighted_length_quota_is_exact_reproducible_and_resolved(self) -> None:
        config = RunConfig(
            num_samples=10,
            focus_labels=["PERSON"],
            sample_structure={"type": "contract"},
            sample_length_distribution={
                "short": 0.2,
                "medium": 0.5,
                "long": 0.3,
            },
            sample_type_distribution={
                "positive": 1.0,
                "pure_negative": 0.0,
                "hard_negative": 0.0,
            },
            random_seed=174,
        )

        planned = []
        for _ in range(2):
            pipeline, repository, _ = build_pipeline(offline=True)
            taxonomy = pipeline.taxonomy_service.import_json(
                Path("pii_taxonomy_rules.json")
            )
            run = pipeline.create_run(
                CreateRunRequest(
                    taxonomy_version_id=taxonomy.version_id,
                    config=config,
                )
            )
            tasks = repository.list_tasks(run.run_id)
            planned.append([
                (
                    task.diversity_profile.length_bucket,
                    task.length_target.dict(),
                )
                for task in tasks
            ])

        self.assertEqual(planned[0], planned[1])
        self.assertEqual(
            Counter(bucket for bucket, _ in planned[0]),
            {"short": 2, "medium": 5, "long": 3},
        )
        targets = {bucket: target for bucket, target in planned[0]}
        self.assertEqual(
            (targets["short"]["min_words"], targets["short"]["max_words"]),
            (80, 120),
        )
        self.assertEqual(
            (targets["medium"]["min_words"], targets["medium"]["max_words"]),
            (150, 230),
        )
        self.assertEqual(
            (targets["long"]["min_words"], targets["long"]["max_words"]),
            (260, 400),
        )

    def test_pipeline_carries_profile_to_seed_prompt_and_result(self) -> None:
        pipeline, repository, events = build_pipeline(offline=True)
        taxonomy = TaxonomySnapshot(labels=[TaxonomyLabel(code="DATE", definition="Ngày cụ thể")])
        config = RunConfig(
            num_samples=1,
            batch_size=1,
            focus_labels=["DATE"],
            sample_type_distribution={"positive": 1.0, "pure_negative": 0.0, "hard_negative": 0.0},
        )

        run = pipeline.create_run(CreateRunRequest(taxonomy=taxonomy, config=config))
        task = repository.list_tasks(run.run_id)[0]
        result = pipeline.generate_pending(run.run_id, 1)[0]
        seed_event = next(event for event in events.list_events() if event.event_type == "seed.validated")

        self.assertIsNotNone(task.diversity_profile)
        self.assertEqual(
            seed_event.payload["seed_pack"]["context_frame"]["frame_id"],
            task.diversity_profile.context_frame_id,
        )
        self.assertEqual(result.diversity_profile, task.diversity_profile)


if __name__ == "__main__":
    unittest.main()
