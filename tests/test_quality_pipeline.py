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
from pii_factory.ports import CompletionClientError


class QualityFirstPipelineTests(unittest.TestCase):
    def test_placeholder_binding_failure_keeps_generator_cost(self) -> None:
        class InvalidPlaceholderGenerator:
            @staticmethod
            def generate(messages):
                return (
                    "<PERSON>[UNKNOWN_1]</PERSON>",
                    [{"label": "PERSON", "value": "[UNKNOWN_1]"}],
                    10,
                    5,
                    15,
                )

        with TemporaryDirectory() as directory:
            pipeline, repository, _ = build_pipeline(
                offline=True,
                output_directory=Path(directory),
            )
            pipeline.generator.client = InvalidPlaceholderGenerator()
            run = pipeline.create_run(self._positive_person_request(
                max_regenerate_attempts=0,
                max_task_replacements=0,
            ))

            self.assertEqual(pipeline.generate_pending(run.run_id, 1), [])

            usage = pipeline._pipeline_usage(run.run_id, 1).generator
            self.assertEqual(repository.get_run(run.run_id).status, "FAILED")
            self.assertEqual(usage.input_tokens, 10)
            self.assertEqual(usage.output_tokens, 5)
            self.assertEqual(usage.total_tokens, 15)

    def test_generator_metadata_is_synchronized_without_losing_cost(self) -> None:
        class InvalidEntityGenerator:
            @staticmethod
            def generate(messages):
                return (
                    "<PERSON>[PERSON_1]</PERSON>",
                    [{"label": "", "value": "[PERSON_1]"}],
                    10,
                    5,
                    15,
                )

        with TemporaryDirectory() as directory:
            pipeline, repository, _ = build_pipeline(
                offline=True,
                output_directory=Path(directory),
            )
            pipeline.generator.client = InvalidEntityGenerator()
            run = pipeline.create_run(self._positive_person_request(
                max_regenerate_attempts=0,
                max_task_replacements=0,
            ))

            results = pipeline.generate_pending(run.run_id, 1)

            usage = pipeline._pipeline_usage(run.run_id, 1).generator
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].entities[0].label, "PERSON")
            self.assertEqual(repository.get_run(run.run_id).status, "COMPLETED")
            self.assertEqual(usage.total_tokens, 15)

    def test_exhausted_paid_generator_responses_keep_cost(self) -> None:
        class ExhaustedPaidGenerator:
            @staticmethod
            def generate(messages):
                raise CompletionClientError(
                    "generator returned malformed content",
                    raw_usage=(20, 40, 60),
                )

        with TemporaryDirectory() as directory:
            pipeline, repository, _ = build_pipeline(
                offline=True,
                output_directory=Path(directory),
            )
            pipeline.generator.client = ExhaustedPaidGenerator()
            run = pipeline.create_run(self._positive_person_request(
                max_regenerate_attempts=0,
                max_task_replacements=0,
            ))

            self.assertEqual(pipeline.generate_pending(run.run_id, 1), [])

            usage = pipeline._pipeline_usage(run.run_id, 1).generator
            self.assertEqual(repository.get_run(run.run_id).status, "FAILED")
            self.assertEqual(usage.input_tokens, 20)
            self.assertEqual(usage.output_tokens, 40)
            self.assertEqual(usage.total_tokens, 60)

    def test_regenerate_deterministic_failure_skips_paid_verifier(self) -> None:
        class ShortGenerator:
            @staticmethod
            def generate(messages):
                payload = json.loads(
                    messages[-1]["content"].split("```json\n", 1)[1].split(
                        "\n```", 1
                    )[0]
                )
                seed = payload["positive_entities"][0]
                return (
                    f"<{seed['label']}>{seed['value']}</{seed['label']}>.",
                    [{"label": seed["label"], "value": seed["value"]}],
                    10,
                    5,
                    15,
                )

        class VerifierMustNotRun:
            @staticmethod
            def judge(messages):
                raise AssertionError(
                    "deterministic REGENERATE must not call the paid verifier"
                )

            @staticmethod
            def repair(messages):
                raise AssertionError("repair must not run")

        with TemporaryDirectory() as directory:
            pipeline, repository, events = build_pipeline(
                offline=True,
                output_directory=Path(directory),
            )
            pipeline.enforce_quality_targets = True
            pipeline.generator.client = ShortGenerator()
            pipeline.verifier.client = VerifierMustNotRun()
            run = pipeline.create_run(self._positive_person_request(
                max_regenerate_attempts=0,
                max_task_replacements=0,
                verifier={"enabled": True},
            ))

            results = pipeline.generate_pending(run.run_id, 1)

            self.assertEqual(results, [])
            self.assertEqual(repository.get_run(run.run_id).status, "FAILED")
            event_types = [event.event_type for event in events.list_events()]
            self.assertIn("data.generation.rejected", event_types)
            self.assertNotIn("data.verification.judged", event_types)

    def test_explicit_verifier_switch_runs_judge_when_legacy_quality_flag_is_false(self) -> None:
        with TemporaryDirectory() as directory:
            pipeline, _, _ = build_pipeline(
                offline=True,
                output_directory=Path(directory),
            )
            run = pipeline.create_run(self._positive_person_request(
                validation={"quality_checks_enabled": False},
                verifier={"enabled": True},
            ))

            result = pipeline.generate_pending(run.run_id, 1)[0]

            self.assertIsNotNone(result.verification_trace)
            self.assertEqual(result.verification_trace.outcome, "PASS")

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
            self.assertEqual(
                result.formatted_sample.token_usage.input_tokens,
                result.pipeline_token_usage.total.input_tokens,
            )
            self.assertEqual(
                result.formatted_sample.token_usage.output_tokens,
                result.pipeline_token_usage.total.output_tokens,
            )
            self.assertEqual(
                result.formatted_sample.token_usage.generator.input_tokens,
                result.pipeline_token_usage.generator.input_tokens,
            )
            self.assertEqual(
                result.formatted_sample.token_usage.verifier.input_tokens,
                result.pipeline_token_usage.verifier_total().input_tokens,
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

    def test_missing_generator_metadata_is_synchronized_before_judge(self) -> None:
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

            first_envelope = verifier.judge_envelopes[0]
            self.assertEqual(first_envelope["deterministic_issues"], [])
            self.assertEqual(
                first_envelope["candidate"]["entities"],
                [{"label": "PERSON", "value": "Nguyen An"}],
            )
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

    def test_best_effort_mode_refuses_missing_required_entity(self) -> None:
        class AlwaysSemanticallyInvalidClient:
            @staticmethod
            def generate(messages):
                return "No required entity was generated.", [], 10, 5, 15

        with TemporaryDirectory() as directory:
            pipeline, repository, events = build_pipeline(
                offline=True,
                output_directory=Path(directory),
            )
            run = pipeline.create_run(
                self._positive_person_request(
                    max_regenerate_attempts=0,
                    max_task_replacements=1,
                    validation={
                        "quality_checks_enabled": True,
                        "accept_last_candidate_on_exhaustion": True,
                    },
                )
            )
            pipeline.generator.client = AlwaysSemanticallyInvalidClient()

            results = pipeline.generate_pending(run.run_id, 1)

            self.assertEqual(results, [])
            self.assertEqual(repository.get_run(run.run_id).status, "FAILED")
            event_types = [event.event_type for event in events.list_events()]
            self.assertNotIn("sample.fallback_accepted", event_types)

    def test_best_effort_mode_can_audit_verifier_regenerate_and_finalize(self) -> None:
        class AlwaysRegenerateVerifier:
            @staticmethod
            def judge(messages):
                return VerifierDecision(
                    status="REGENERATE",
                    score=40,
                    issues=[VerificationIssue(
                        type="CONTENT_QUALITY",
                        severity="high",
                        field="candidate",
                        reason="The verifier requests another generation.",
                        suggested_fix="Generate a more natural candidate.",
                    )],
                    token_usage=TokenUsage.zero(),
                    latency_ms=1,
                    model="fake-verifier",
                    prompt_version="judge.test",
                )

            @staticmethod
            def repair(messages):
                raise AssertionError("repair must not run for REGENERATE")

        with TemporaryDirectory() as directory:
            pipeline, repository, events = build_pipeline(
                offline=True,
                output_directory=Path(directory),
            )
            pipeline.verifier.client = AlwaysRegenerateVerifier()
            run = pipeline.create_run(self._positive_person_request(
                max_regenerate_attempts=0,
                max_task_replacements=0,
                validation={
                    "quality_checks_enabled": True,
                    "accept_last_candidate_on_exhaustion": True,
                },
                verifier={"enabled": True},
            ))

            result = pipeline.generate_pending(run.run_id, 1)[0]

            self.assertEqual(result.verification_trace.outcome, "REGENERATE")
            self.assertEqual(repository.get_run(run.run_id).status, "COMPLETED")
            self.assertIn(
                "sample.fallback_accepted",
                [event.event_type for event in events.list_events()],
            )

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

            files = list(Path(directory).rglob("*.json"))
            self.assertEqual(len(results), 1)
            self.assertEqual(repository.get_run(run.run_id).status, "FAILED")
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].name.endswith(".partial.json"))
            self.assertEqual(
                len(json.loads(files[0].read_text(encoding="utf-8"))),
                1,
            )
            accepted_usage = results[0].pipeline_token_usage
            run_usage = pipeline.run_usage(run.run_id)
            self.assertEqual(accepted_usage.generator.total_tokens, 15)
            self.assertEqual(run_usage.generator.total_tokens, 30)
            self.assertEqual(run_usage.total.total_tokens, 30)

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

    def test_rejudge_failure_keeps_prior_judge_repair_and_rejudge_costs(self) -> None:
        class FailFirstRejudgeVerifier:
            def __init__(self) -> None:
                self.judge_calls = 0

            def judge(self, messages):
                self.judge_calls += 1
                if self.judge_calls == 1:
                    return VerifierDecision(
                        status="FIXABLE",
                        score=80,
                        issues=[VerificationIssue(
                            type="LOCAL_WORDING",
                            severity="low",
                            field="tagged_text",
                            reason="Local wording needs repair.",
                            suggested_fix="Apply a local wording repair.",
                        )],
                        token_usage=self._usage(),
                        latency_ms=1,
                        model="fake-verifier",
                        prompt_version="judge.test",
                    )
                if self.judge_calls == 2:
                    raise VerifierInfrastructureError(
                        "temporary rejudge failure",
                        stage="judge",
                        token_usage=self._usage(),
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
            pipeline.verifier.client = FailFirstRejudgeVerifier()
            run = pipeline.create_run(self._positive_person_request())

            with self.assertRaises(VerifierInfrastructureError) as raised:
                pipeline.generate_pending(run.run_id, 1)
            self.assertEqual(raised.exception.stage, "rejudge")

            result = pipeline.generate_pending(run.run_id, 1)[0]

            self.assertEqual(
                result.pipeline_token_usage.verifier_judge.total_tokens,
                60,
            )
            self.assertEqual(
                result.pipeline_token_usage.verifier_repair.total_tokens,
                30,
            )
            self.assertEqual(
                result.pipeline_token_usage.verifier_rejudge.total_tokens,
                30,
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
