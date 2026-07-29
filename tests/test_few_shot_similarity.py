import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pii_factory.bootstrap import build_pipeline
from pii_factory.application.few_shot_similarity import (
    FewShotImitationGuard,
)
from pii_factory.domain.models import (
    CreateRunRequest,
    FewShotExample,
    RunConfig,
)


class FewShotImitationGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = FewShotImitationGuard()
        self.example = FewShotExample(
            id="api_key_positive_1",
            expected_tagged_text=(
                "Yêu cầu được xác thực bằng API key "
                "<API_KEY>tok_prod_7788ZZ</API_KEY>."
            ),
            rationale="API key xác thực yêu cầu.",
        )

    def test_rejects_an_exact_few_shot_copy(self) -> None:
        issue = self.guard.find_imitation(
            self.example.expected_tagged_text,
            [self.example],
        )

        self.assertIsNotNone(issue)
        self.assertEqual(issue.type, "few_shot_imitation")
        self.assertIn("api_key_positive_1", issue.reason)

    def test_rejects_the_same_template_with_a_different_entity(self) -> None:
        issue = self.guard.find_imitation(
            (
                "Yêu cầu được xác thực bằng API key "
                "<API_KEY>service_test_9922</API_KEY>."
            ),
            [self.example],
        )

        self.assertIsNotNone(issue)

    def test_allows_a_different_context_and_sentence_structure(self) -> None:
        issue = self.guard.find_imitation(
            (
                "Trong biên bản bàn giao, nhóm vận hành ghi nhận kết nối thanh "
                "toán bị từ chối sau khi kho bí mật cấp "
                "<API_KEY>service_test_9922</API_KEY> cho phiên triển khai mới."
            ),
            [self.example],
        )

        self.assertIsNone(issue)

    def test_decoy_examples_are_checked_and_duplicate_ids_are_safe(
        self,
    ) -> None:
        decoy_example = FewShotExample(
            id="card_issuer_hard_negative_1",
            expected_tagged_text=(
                "Bộ phận hồ sơ yêu cầu bổ sung visa đi công tác."
            ),
            rationale="Visa is a travel document in this event.",
        )

        issue = self.guard.find_imitation(
            decoy_example.expected_tagged_text,
            [decoy_example, decoy_example],
        )

        self.assertIsNotNone(issue)
        self.assertIn("card_issuer_hard_negative_1", issue.reason)


class FewShotImitationPipelineTests(unittest.TestCase):
    def test_guard_regenerates_even_when_quality_checks_are_disabled(self) -> None:
        class FewShotCopyingClient:
            @staticmethod
            def generate(messages):
                content = messages[1]["content"]
                envelope = json.loads(
                    content.split("```json\n", 1)[1].split("\n```", 1)[0]
                )
                seed = envelope["positive_entities"][0]
                tagged_text = (
                    "Yêu cầu được xác thực bằng API key "
                    f"<API_KEY>{seed['value']}</API_KEY>."
                )
                return (
                    tagged_text,
                    [{"label": "API_KEY", "value": seed["value"]}],
                    50,
                    20,
                    70,
                )

        with TemporaryDirectory() as directory:
            pipeline, repository, events = build_pipeline(
                offline=True,
                output_directory=Path(directory),
            )
            pipeline.generator.client = FewShotCopyingClient()
            taxonomy = pipeline.taxonomy_service.import_json(
                Path("pii_taxonomy_rules.json")
            )
            run = pipeline.create_run(CreateRunRequest(
                taxonomy_version_id=taxonomy.version_id,
                config=RunConfig(
                    num_samples=1,
                    focus_label="API_KEY",
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
                    max_regenerate_attempts=1,
                    max_task_replacements=0,
                    validation={"quality_checks_enabled": False},
                ),
            ))

            results = pipeline.generate_pending(run.run_id, limit=1)

            self.assertEqual(results, [])
            self.assertEqual(repository.get_run(run.run_id).status, "FAILED")
            rejected = [
                event
                for event in events.list_events()
                if event.event_type == "data.generation.rejected"
            ]
            self.assertEqual(len(rejected), 2)
            self.assertTrue(all(
                any(
                    issue["type"] == "few_shot_imitation"
                    for issue in event.payload["output_validation"]["issues"]
                )
                for event in rejected
            ))


if __name__ == "__main__":
    unittest.main()
