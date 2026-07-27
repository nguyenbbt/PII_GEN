from decimal import Decimal
import json
import unittest

from pii_factory.application.verification import (
    DeterministicIssueRouter,
    VerificationRoutingError,
    VerifierService,
)
from pii_factory.domain.models import (
    ContextFrame,
    DeterministicValidationResult,
    DiversityProfile,
    GeneratedEntity,
    GenerationCandidate,
    GenerationQuery,
    GenerationTask,
    GenerationTaxonomyContext,
    LabelGenerationContext,
    LengthTarget,
    PositiveEntitySeed,
    RepairResult,
    SeedPack,
    TokenUsage,
    ValidationIssue,
    VerificationIssue,
    VerifierDecision,
)


def token_usage() -> TokenUsage:
    return TokenUsage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        money_cost=Decimal("0.001"),
    )


def issue(severity: str = "low") -> VerificationIssue:
    return VerificationIssue(
        type="BOUNDARY",
        severity=severity,
        field="tagged_text",
        reason="Dấu câu nằm trong entity boundary.",
        suggested_fix="Đưa dấu câu ra ngoài tag.",
    )


def decision(status: str) -> VerifierDecision:
    issues = [] if status == "PASS" else [
        issue("critical" if status == "REJECTED" else ("low" if status == "FIXABLE" else "high"))
    ]
    return VerifierDecision(
        status=status,
        score=98 if status == "PASS" else 60,
        issues=issues,
        token_usage=token_usage(),
        latency_ms=2,
        model="offline-verifier",
        prompt_version="judge.test",
    )


def candidate() -> GenerationCandidate:
    return GenerationCandidate(
        task_id="task-1",
        attempt_no=1,
        seed_pack_id="seed-1",
        context_frame_id="support",
        generation_query=GenerationQuery(
            language="vi",
            focus_labels=["PERSON"],
            constraints=[],
            difficulty="medium",
            sample_type="positive",
            max_entities=2,
        ),
        entities=[GeneratedEntity(label="PERSON", value="Lò Thị Cẩy")],
        tagged_text="Chị <PERSON>Lò Thị Cẩy</PERSON> đã gửi hồ sơ.",
        token_usage=token_usage(),
        latency_ms=3,
        model="offline-generator",
        prompt_version="generator.test",
        output_hash="hash",
        seed_validation=DeterministicValidationResult(valid=True),
        diversity_profile=DiversityProfile(),
    )


def task() -> GenerationTask:
    return GenerationTask(
        task_id="task-1",
        run_id="run-1",
        sequence_no=1,
        slot_no=1,
        language="vi",
        focus_labels=["PERSON"],
        difficulty="medium",
        sample_type="positive",
        max_entities=2,
        max_attempts=3,
        random_seed=42,
        length_target=LengthTarget(
            bucket="short",
            min_words=1,
            max_words=30,
            unit="content_units",
            min_units=3,
            max_units=5,
        ),
    )


def seed_pack() -> SeedPack:
    return SeedPack(
        seed_pack_id="seed-1",
        task_id="task-1",
        sample_type="positive",
        positive_entities=[
            PositiveEntitySeed(
                label="PERSON",
                value="Lò Thị Cẩy",
                semantic_role="customer_name",
            )
        ],
        context_frame=ContextFrame(
            frame_id="support",
            domain="support",
            document_type="note",
            tone="neutral",
            max_sentences=2,
            supported_labels=["PERSON"],
        ),
    )


class FakeVerifierClient:
    def __init__(self, decisions, repair_result=None) -> None:
        self.decisions = list(decisions)
        self.repair_result = repair_result
        self.judge_messages = []
        self.repair_messages = []

    def judge(self, messages):
        self.judge_messages.append(messages)
        return self.decisions.pop(0)

    def repair(self, messages):
        self.repair_messages.append(messages)
        if self.repair_result is None:
            raise AssertionError("repair was not expected")
        return self.repair_result


class VerifierServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.taxonomy_context = GenerationTaxonomyContext(
            taxonomy_version_id="taxonomy-v1",
            sample_type="positive",
            focus_label=LabelGenerationContext(
                label="PERSON",
                definition="Tên của một người.",
                rule="Gắn toàn bộ họ tên.",
            ),
        )

    def test_deterministic_issue_router_distinguishes_quality_routes(self) -> None:
        cases = (
            (
                DeterministicValidationResult(valid=True),
                "PASS",
                None,
            ),
            (
                DeterministicValidationResult(
                    valid=False,
                    issues=[
                        ValidationIssue(
                            type="invalid_output",
                            scope="TEXT",
                            reason="Entity metadata differs from tagged spans.",
                        ),
                        ValidationIssue(
                            type="missing_entity_metadata",
                            scope="TEXT",
                            reason="Required metadata is missing.",
                        ),
                    ],
                ),
                "FIXABLE",
                "low",
            ),
            (
                DeterministicValidationResult(
                    valid=False,
                    issues=[
                        ValidationIssue(
                            type="decoy_context_unclear",
                            scope="TEXT",
                            reason="The hard-negative cue is ambiguous.",
                        )
                    ],
                ),
                "REGENERATE",
                "high",
            ),
            (
                DeterministicValidationResult(
                    valid=False,
                    issues=[
                        ValidationIssue(
                            type="credential_risk",
                            scope="TEXT",
                            reason="A credential-like secret was detected.",
                        )
                    ],
                ),
                "REJECTED",
                "critical",
            ),
        )

        for validation, expected_route, expected_severity in cases:
            with self.subTest(route=expected_route):
                route, issues = DeterministicIssueRouter.route(validation)
                self.assertEqual(route, expected_route)
                self.assertEqual(
                    issues[0].severity if issues else None,
                    expected_severity,
                )

    def test_pass_candidate_is_returned_without_repair(self) -> None:
        client = FakeVerifierClient([decision("PASS")])
        verified, trace = VerifierService(client).verify(
            candidate=candidate(),
            task=task(),
            seed_pack=seed_pack(),
            taxonomy_context=self.taxonomy_context,
            revalidate=lambda _: DeterministicValidationResult(valid=True),
        )

        self.assertEqual(trace.outcome, "PASS")
        self.assertEqual(verified.tagged_text, candidate().tagged_text)
        self.assertEqual(client.repair_messages, [])
        system_prompt = client.judge_messages[0][0]["content"]
        compact_system_prompt = " ".join(system_prompt.split())
        user_payload = json.loads(client.judge_messages[0][1]["content"])
        self.assertIn("untrusted data", system_prompt)
        self.assertIn("deterministic_metrics", system_prompt)
        self.assertIn("ADDRESS", system_prompt)
        self.assertIn("LOCATION", system_prompt)
        self.assertIn("few-shot", system_prompt)
        self.assertIn("comma-separated", system_prompt)
        self.assertIn("unnatural", system_prompt)
        self.assertIn("at most 5 issues", system_prompt)
        self.assertIn("low, medium, high, or critical", system_prompt)
        self.assertIn(
            "minor, major, warning, error",
            compact_system_prompt,
        )
        self.assertEqual(
            set(user_payload["candidate"]),
            {"tagged_text", "entities"},
        )
        self.assertNotIn("run_id", user_payload["task"])
        self.assertNotIn("random_seed", user_payload["task"])
        self.assertNotIn("token_usage", user_payload["candidate"])
        self.assertNotIn("output_hash", user_payload["candidate"])
        self.assertLess(len(system_prompt), 3500)
        legacy_envelope = {
            "task": task().dict(),
            "seed_pack": seed_pack().dict(),
            "taxonomy_context": self.taxonomy_context.dict(),
            "deterministic_issues": [],
            "candidate": candidate().dict(),
        }
        legacy_size = len(
            json.dumps(legacy_envelope, ensure_ascii=False, default=str)
        )
        compact_size = len(client.judge_messages[0][1]["content"])
        self.assertLess(compact_size, legacy_size * 0.75)
        self.assertTrue(
            user_payload["deterministic_metrics"]["word_range_satisfied"]
        )
        self.assertTrue(
            user_payload["deterministic_metrics"]["entity_count_satisfied"]
        )
        self.assertIsNone(
            user_payload["deterministic_metrics"]["chat_turn_count"]
        )
        self.assertIn("authoritative length measurement", system_prompt)
        self.assertIn("do not invent a", system_prompt)

    def test_authoritative_metrics_override_false_judge_rejections(self) -> None:
        false_issues = [
            VerificationIssue(
                type="length_out_of_range",
                severity="high",
                field="candidate",
                reason="The valid deterministic word count was recounted incorrectly.",
                suggested_fix="Regenerate.",
            ),
            VerificationIssue(
                type="entity_count_mismatch",
                severity="high",
                field="entities",
                reason="The valid deterministic entity count was recounted incorrectly.",
                suggested_fix="Regenerate.",
            ),
        ]
        client = FakeVerifierClient([
            VerifierDecision(
                status="REGENERATE",
                score=50,
                issues=false_issues,
                token_usage=token_usage(),
                latency_ms=2,
                model="offline-verifier",
                prompt_version="judge.test",
            )
        ])

        verified, trace = VerifierService(client).verify(
            candidate=candidate(),
            task=task(),
            seed_pack=seed_pack(),
            taxonomy_context=self.taxonomy_context,
            revalidate=lambda _: DeterministicValidationResult(valid=True),
        )

        self.assertEqual(verified, candidate())
        self.assertEqual(trace.outcome, "PASS")
        self.assertEqual(trace.initial_judge.status, "PASS")
        self.assertEqual(trace.initial_judge.issues, [])

    def test_fixable_candidate_is_repaired_rechecked_and_rejudged(self) -> None:
        repaired = RepairResult(
            tagged_text="Chị <PERSON>Lò Thị Cẩy</PERSON> đã gửi hồ sơ.",
            entities=[GeneratedEntity(label="PERSON", value="Lò Thị Cẩy")],
            token_usage=token_usage(),
            latency_ms=2,
            model="offline-verifier",
            prompt_version="repair.test",
        )
        client = FakeVerifierClient(
            [decision("FIXABLE"), decision("PASS")],
            repair_result=repaired,
        )

        verified, trace = VerifierService(client).verify(
            candidate=candidate(),
            task=task(),
            seed_pack=seed_pack(),
            taxonomy_context=self.taxonomy_context,
            revalidate=lambda _: DeterministicValidationResult(valid=True),
        )

        self.assertEqual(trace.outcome, "FIXED")
        self.assertEqual(trace.repair, repaired)
        self.assertEqual(trace.final_judge.status, "PASS")
        self.assertEqual(verified.tagged_text, repaired.tagged_text)
        self.assertEqual(len(client.judge_messages), 2)
        self.assertEqual(len(client.repair_messages), 1)

    def test_regenerate_and_rejected_decisions_route_without_repair(self) -> None:
        for status in ("REGENERATE", "REJECTED"):
            with self.subTest(status=status):
                client = FakeVerifierClient([decision(status)])
                with self.assertRaises(VerificationRoutingError) as raised:
                    VerifierService(client).verify(
                        candidate=candidate(),
                        task=task(),
                        seed_pack=seed_pack(),
                        taxonomy_context=self.taxonomy_context,
                        revalidate=lambda _: DeterministicValidationResult(valid=True),
                    )
                self.assertEqual(raised.exception.status, status)
                self.assertEqual(client.repair_messages, [])

    def test_repair_that_changes_a_positive_seed_is_regenerated(self) -> None:
        unsafe_repair = RepairResult(
            tagged_text="Chị <PERSON>Nguyễn An</PERSON> đã gửi hồ sơ.",
            entities=[GeneratedEntity(label="PERSON", value="Nguyễn An")],
            token_usage=token_usage(),
            latency_ms=2,
            model="offline-verifier",
            prompt_version="repair.test",
        )
        client = FakeVerifierClient(
            [decision("FIXABLE")],
            repair_result=unsafe_repair,
        )

        with self.assertRaises(VerificationRoutingError) as raised:
            VerifierService(client).verify(
                candidate=candidate(),
                task=task(),
                seed_pack=seed_pack(),
                taxonomy_context=self.taxonomy_context,
                revalidate=lambda _: DeterministicValidationResult(valid=True),
            )

        self.assertEqual(raised.exception.status, "REGENERATE")
        self.assertEqual(len(client.judge_messages), 1)

    def test_failed_deterministic_recheck_is_regenerated(self) -> None:
        repaired = RepairResult(
            tagged_text=candidate().tagged_text,
            entities=candidate().entities,
            token_usage=token_usage(),
            latency_ms=2,
            model="offline-verifier",
            prompt_version="repair.test",
        )
        client = FakeVerifierClient(
            [decision("FIXABLE")],
            repair_result=repaired,
        )

        with self.assertRaises(VerificationRoutingError) as raised:
            VerifierService(client).verify(
                candidate=candidate(),
                task=task(),
                seed_pack=seed_pack(),
                taxonomy_context=self.taxonomy_context,
                revalidate=lambda _: DeterministicValidationResult(
                    valid=False,
                    issues=[],
                ),
            )

        self.assertEqual(raised.exception.status, "REGENERATE")
        self.assertEqual(len(client.judge_messages), 1)


if __name__ == "__main__":
    unittest.main()
