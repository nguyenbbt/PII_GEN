import json
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pii_factory.bootstrap import build_pipeline
from pii_factory.application.verification import VerifierInfrastructureError
from pii_factory.domain.models import (
    CreateRunRequest,
    GeneratedEntity,
    RepairResult,
    RunConfig,
    TaxonomyLabel,
    TaxonomySnapshot,
    TokenUsage,
    VerificationIssue,
    VerifierDecision,
)


class QualityFirstPipelineTests(unittest.TestCase):
    def test_disabled_quality_checks_skip_verifier_and_keep_technical_gate(self) -> None:
        class VerifierMustNotRun:
            @staticmethod
            def judge(messages):
                raise AssertionError("quality verifier must be disabled")

            @staticmethod
            def repair(messages):
                raise AssertionError("quality repair must be disabled")

        with TemporaryDirectory() as directory:
            pipeline, _, events = build_pipeline(
                offline=True,
                output_directory=Path(directory),
            )
            pipeline.verifier.client = VerifierMustNotRun()
            run = pipeline.create_run(self._positive_person_request(
                validation={"quality_checks_enabled": False},
            ))

            result = pipeline.generate_pending(run.run_id, 1)[0]

            self.assertIsNone(result.verification_trace)
            self.assertTrue(result.output_validation.valid)
            self.assertIsNotNone(result.formatted_sample)
            self.assertEqual(
                result.pipeline_token_usage.verifier_judge.total_tokens,
                0,
            )
            event_types = [event.event_type for event in events.list_events()]
            self.assertNotIn("data.verification.judged", event_types)
            self.assertNotIn("data.verification.rejected", event_types)

    def test_offline_pipeline_accepts_formats_and_publishes_final_dataset(self) -> None:
        with TemporaryDirectory() as directory:
            pipeline, repository, events = build_pipeline(
                offline=True,
                output_directory=Path(directory),
            )
            run = pipeline.create_run(
                CreateRunRequest(
                    run_name="../quality vi",
                    config=RunConfig(
                        num_samples=1,
                        batch_size=1,
                        focus_labels=["PERSON"],
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
                        validation={"quality_checks_enabled": True},
                    ),
                    taxonomy=TaxonomySnapshot(
                        labels=[
                            TaxonomyLabel(
                                code="PERSON",
                                definition="Tên của một người.",
                            )
                        ]
                    ),
                )
            )

            results = pipeline.generate_pending(run.run_id, limit=1)

            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertEqual(result.verification_trace.outcome, "PASS")
            self.assertIsNotNone(result.formatted_sample)
            self.assertEqual(
                result.pipeline_token_usage.total.total_tokens,
                result.token_usage.total_tokens,
            )
            for entity in result.formatted_sample.entities:
                self.assertEqual(
                    result.formatted_sample.text[entity.start:entity.end],
                    entity.text,
                )

            completed_run = repository.get_run(run.run_id)
            self.assertEqual(completed_run.status, "COMPLETED")
            output_path = Path(completed_run.output_path)
            self.assertTrue(output_path.exists())
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                [result.formatted_sample.dict()],
            )
            self.assertNotIn("..", output_path.name)
            event_types = [event.event_type for event in events.list_events()]
            self.assertLess(
                event_types.index("data.generated"),
                event_types.index("data.deterministic.validated"),
            )
            self.assertLess(
                event_types.index("data.verification.judged"),
                event_types.index("sample.formatted"),
            )
            self.assertLess(
                event_types.index("sample.formatted"),
                event_types.index("sample.accepted"),
            )

    def test_fixable_candidate_is_repaired_and_costed_before_formatting(self) -> None:
        class FixingVerifier:
            def __init__(self) -> None:
                self.judge_calls = 0

            def judge(self, messages):
                self.judge_calls += 1
                if self.judge_calls == 1:
                    return VerifierDecision(
                        status="FIXABLE",
                        score=82,
                        issues=[VerificationIssue(
                            type="LOCAL_WORDING",
                            severity="low",
                            field="tagged_text",
                            reason="Cần sửa wording cục bộ.",
                            suggested_fix="Giữ seed và sửa wording.",
                        )],
                        token_usage=self._usage(),
                        latency_ms=1,
                        model="fake-verifier",
                        prompt_version="judge.test",
                    )
                return VerifierDecision(
                    status="PASS",
                    score=99,
                    issues=[],
                    token_usage=self._usage(),
                    latency_ms=1,
                    model="fake-verifier",
                    prompt_version="judge.test",
                )

            def repair(self, messages):
                envelope = json.loads(messages[-1]["content"])
                current = envelope["candidate"]
                return RepairResult(
                    tagged_text=current["tagged_text"],
                    entities=[
                        GeneratedEntity(**entity)
                        for entity in current["entities"]
                    ],
                    token_usage=self._usage(),
                    latency_ms=1,
                    model="fake-verifier",
                    prompt_version="repair.test",
                )

            @staticmethod
            def _usage():
                return TokenUsage(
                    input_tokens=20,
                    output_tokens=10,
                    total_tokens=30,
                    money_cost=Decimal("0.002"),
                )

        with TemporaryDirectory() as directory:
            pipeline, _, _ = build_pipeline(
                offline=True,
                output_directory=Path(directory),
            )
            pipeline.verifier.client = FixingVerifier()
            run = pipeline.create_run(self._positive_person_request())

            result = pipeline.generate_pending(run.run_id, 1)[0]

            self.assertEqual(result.verification_trace.outcome, "FIXED")
            self.assertEqual(
                result.pipeline_token_usage.verifier_judge.total_tokens,
                30,
            )
            self.assertEqual(
                result.pipeline_token_usage.verifier_repair.total_tokens,
                30,
            )
            self.assertEqual(
                result.pipeline_token_usage.verifier_rejudge.total_tokens,
                30,
            )
            self.assertEqual(
                result.pipeline_token_usage.total.total_tokens,
                result.token_usage.total_tokens + 90,
            )

    def test_fixable_deterministic_metadata_issue_reaches_judge_and_repair(self) -> None:
        class MissingMetadataGenerator:
            @staticmethod
            def generate(messages):
                payload = json.loads(
                    messages[-1]["content"].split("```json\n", 1)[1].split(
                        "\n```", 1
                    )[0]
                )
                seed = payload["positive_entities"][0]
                tagged_text = (
                    f"Chị <{seed['label']}>{seed['value']}</{seed['label']}> "
                    "đã gửi hồ sơ."
                )
                return tagged_text, [], 10, 5, 15

        class MetadataRepairVerifier:
            def __init__(self) -> None:
                self.judge_envelopes = []

            def judge(self, messages):
                envelope = json.loads(messages[-1]["content"])
                self.judge_envelopes.append(envelope)
                if len(self.judge_envelopes) == 1:
                    return VerifierDecision(
                        status="FIXABLE",
                        score=80,
                        issues=[VerificationIssue(
                            type="missing_entity_metadata",
                            severity="low",
                            field="TEXT",
                            reason="Entity metadata is missing.",
                            suggested_fix="Restore metadata from the tagged span.",
                        )],
                        token_usage=TokenUsage.zero(),
                        latency_ms=1,
                        model="fake-verifier",
                        prompt_version="judge.test",
                    )
                return VerifierDecision(
                    status="PASS",
                    score=99,
                    issues=[],
                    token_usage=TokenUsage.zero(),
                    latency_ms=1,
                    model="fake-verifier",
                    prompt_version="judge.test",
                )

            @staticmethod
            def repair(messages):
                envelope = json.loads(messages[-1]["content"])
                seed = envelope["seed_pack"]["positive_entities"][0]
                return RepairResult(
                    tagged_text=envelope["candidate"]["tagged_text"],
                    entities=[
                        GeneratedEntity(label=seed["label"], value=seed["value"])
                    ],
                    token_usage=TokenUsage.zero(),
                    latency_ms=1,
                    model="fake-verifier",
                    prompt_version="repair.test",
                )

        with TemporaryDirectory() as directory:
            pipeline, _, _ = build_pipeline(
                offline=True,
                output_directory=Path(directory),
            )
            pipeline.generator.client = MissingMetadataGenerator()
            verifier = MetadataRepairVerifier()
            pipeline.verifier.client = verifier
            run = pipeline.create_run(self._positive_person_request())

            result = pipeline.generate_pending(run.run_id, 1)[0]

            issue_types = {
                issue["type"]
                for issue in verifier.judge_envelopes[0]["deterministic_issues"]
            }
            self.assertIn("missing_entity_metadata", issue_types)
            self.assertEqual(result.verification_trace.outcome, "FIXED")
            self.assertTrue(result.output_validation.valid)
            self.assertEqual(len(result.formatted_sample.entities), 1)

    def test_rejected_task_is_replaced_in_same_slot_and_cost_is_retained(self) -> None:
        class FailThenSucceedClient:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, messages):
                self.calls += 1
                payload = json.loads(
                    messages[-1]["content"].split("```json\n", 1)[1].split(
                        "\n```", 1
                    )[0]
                )
                if self.calls == 1:
                    return "Không có entity.", [], 10, 5, 15
                seed = payload["positive_entities"][0]
                return (
                    f"Chị <{seed['label']}>{seed['value']}</{seed['label']}> đã gửi hồ sơ.",
                    [{"label": seed["label"], "value": seed["value"]}],
                    10,
                    5,
                    15,
                )

        with TemporaryDirectory() as directory:
            pipeline, repository, _ = build_pipeline(
                offline=True,
                output_directory=Path(directory),
            )
            run = pipeline.create_run(
                self._positive_person_request(
                    max_regenerate_attempts=0,
                    max_task_replacements=1,
                )
            )
            original = repository.list_tasks(run.run_id)[0]
            pipeline.generator.client = FailThenSucceedClient()

            result = pipeline.generate_pending(run.run_id, 1)[0]
            tasks = repository.list_tasks(run.run_id)
            replacement = tasks[-1]

            self.assertEqual(len(tasks), 2)
            self.assertEqual(original.status, "REJECTED")
            self.assertEqual(replacement.status, "ACCEPTED")
            self.assertEqual(replacement.slot_no, original.slot_no)
            self.assertEqual(replacement.focus_labels, original.focus_labels)
            self.assertEqual(replacement.difficulty, original.difficulty)
            self.assertEqual(replacement.sample_type, original.sample_type)
            self.assertNotEqual(replacement.random_seed, original.random_seed)
            self.assertNotEqual(
                replacement.diversity_profile.context_frame_id,
                original.diversity_profile.context_frame_id,
            )
            self.assertEqual(result.pipeline_token_usage.generator.total_tokens, 30)
            self.assertEqual(repository.get_run(run.run_id).status, "COMPLETED")

    def test_exhausted_replacement_budget_fails_without_final_json(self) -> None:
        class AlwaysInvalidClient:
            @staticmethod
            def generate(messages):
                return "Không có entity.", [], 10, 5, 15

        with TemporaryDirectory() as directory:
            pipeline, repository, _ = build_pipeline(
                offline=True,
                output_directory=Path(directory),
            )
            run = pipeline.create_run(
                self._positive_person_request(
                    max_regenerate_attempts=0,
                    max_task_replacements=1,
                )
            )
            pipeline.generator.client = AlwaysInvalidClient()

            results = pipeline.generate_pending(run.run_id, 1)

            self.assertEqual(results, [])
            self.assertEqual(repository.get_run(run.run_id).status, "FAILED")
            self.assertEqual(list(Path(directory).glob("*.json")), [])

    def test_failed_run_keeps_only_partial_artifact_for_accepted_slots(self) -> None:
        class AcceptFirstThenFail:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, messages):
                self.calls += 1
                payload = json.loads(
                    messages[-1]["content"].split("```json\n", 1)[1].split(
                        "\n```", 1
                    )[0]
                )
                if self.calls > 1:
                    return "Không có entity.", [], 10, 5, 15
                seed = payload["positive_entities"][0]
                return (
                    f"Chị <{seed['label']}>{seed['value']}</{seed['label']}> "
                    "đã gửi hồ sơ.",
                    [{"label": seed["label"], "value": seed["value"]}],
                    10,
                    5,
                    15,
                )

        with TemporaryDirectory() as directory:
            pipeline, repository, _ = build_pipeline(
                offline=True,
                output_directory=Path(directory),
            )
            request = self._positive_person_request(
                max_regenerate_attempts=0,
                max_task_replacements=0,
            )
            request.config.num_samples = 2
            request.config.batch_size = 2
            run = pipeline.create_run(request)
            pipeline.generator.client = AcceptFirstThenFail()

            results = pipeline.generate_pending(run.run_id, 2)

            files = list(Path(directory).glob("*.json"))
            self.assertEqual(len(results), 1)
            self.assertEqual(repository.get_run(run.run_id).status, "FAILED")
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].name.endswith(".partial.json"))
            self.assertEqual(
                len(json.loads(files[0].read_text(encoding="utf-8"))),
                1,
            )

    def test_verifier_infrastructure_failure_reuses_candidate_without_consuming_attempt(self) -> None:
        class CountingGeneratorClient:
            def __init__(self, delegate) -> None:
                self.delegate = delegate
                self.calls = 0

            def generate(self, messages):
                self.calls += 1
                return self.delegate.generate(messages)

        class FailOnceVerifier:
            def __init__(self) -> None:
                self.calls = 0

            def judge(self, messages):
                self.calls += 1
                if self.calls == 1:
                    raise VerifierInfrastructureError(
                        "invalid verifier payload",
                        stage="judge",
                        token_usage=TokenUsage(
                            input_tokens=20,
                            output_tokens=10,
                            total_tokens=30,
                            money_cost=Decimal("0.002"),
                        ),
                    )
                return VerifierDecision(
                    status="PASS",
                    score=99,
                    issues=[],
                    token_usage=TokenUsage(
                        input_tokens=20,
                        output_tokens=10,
                        total_tokens=30,
                        money_cost=Decimal("0.002"),
                    ),
                    latency_ms=1,
                    model="fake-verifier",
                    prompt_version="judge.test",
                )

            def repair(self, messages):
                raise AssertionError("repair was not expected")

        with TemporaryDirectory() as directory:
            pipeline, _, _ = build_pipeline(
                offline=True,
                output_directory=Path(directory),
            )
            generator_client = CountingGeneratorClient(pipeline.generator.client)
            pipeline.generator.client = generator_client
            pipeline.verifier.client = FailOnceVerifier()
            run = pipeline.create_run(self._positive_person_request())

            with self.assertRaises(VerifierInfrastructureError):
                pipeline.generate_pending(run.run_id, 1)
            result = pipeline.generate_pending(run.run_id, 1)[0]

            self.assertEqual(generator_client.calls, 1)
            self.assertEqual(result.attempt_no, 1)
            self.assertEqual(
                result.pipeline_token_usage.verifier_judge.total_tokens,
                60,
            )

    @staticmethod
    def _positive_person_request(**config_overrides):
        config_overrides.setdefault(
            "validation",
            {"quality_checks_enabled": True},
        )
        config = RunConfig(
            num_samples=1,
            batch_size=1,
            focus_labels=["PERSON"],
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
            **config_overrides,
        )
        return CreateRunRequest(
            run_name="quality-vi",
            config=config,
            taxonomy=TaxonomySnapshot(
                labels=[
                    TaxonomyLabel(
                        code="PERSON",
                        definition="Tên của một người.",
                    )
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
