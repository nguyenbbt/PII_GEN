import unittest

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
