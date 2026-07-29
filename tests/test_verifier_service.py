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
    VerificationEdit,
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
        self.repair_results = (
            list(repair_result)
            if isinstance(repair_result, (list, tuple))
            else [repair_result]
        )
        self.judge_messages = []
        self.repair_messages = []

    def judge(self, messages):
        self.judge_messages.append(messages)
        return self.decisions.pop(0)

    def repair(self, messages):
        self.repair_messages.append(messages)
        if not self.repair_results or self.repair_results[0] is None:
            raise AssertionError("repair was not expected")
        if len(self.repair_results) == 1:
            return self.repair_results[0]
        return self.repair_results.pop(0)


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
                            type="positive_seed_annotation_mismatch",
                            scope="TEXT",
                            reason="positive seed surface is present but inconsistently tagged",
                            label="PERSON",
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
                            type="temporal_boundary",
                            scope="TEXT",
                            reason="Keep Ngày and GMT outside the temporal tag.",
                            label="DATE",
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
                            type="missing_positive_seed",
                            scope="TEXT",
                            reason="positive seed must appear with its exact tag",
                            label="PERSON",
                        ),
                        ValidationIssue(
                            type="missing_entity_metadata",
                            scope="TEXT",
                            reason="positive seed is missing from entities",
                            label="PERSON",
                        ),
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
        original_candidate = candidate()
        verified, trace = VerifierService(client).verify(
            candidate=original_candidate,
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
        self.assertIn("mixed_contrastive keeps tagged", system_prompt)
        self.assertIn("never apply decoy_only rules", system_prompt)
        self.assertIn("may later be called `field`", system_prompt)
        self.assertIn("at most 10 issues", system_prompt)
        self.assertIn("suggested_fix must always be a non-empty string", system_prompt)
        self.assertIn("never", system_prompt)
        self.assertIn("Every edit must contain action", system_prompt)
        self.assertIn("occurrence is a one-based integer", system_prompt)
        self.assertIn("never use 0", system_prompt)
        self.assertIn("Error examples", system_prompt)
        self.assertIn("15 tháng 5 năm nay", system_prompt)
        self.assertIn("UTC remains untagged", system_prompt)
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
        self.assertLess(len(system_prompt), 7600)
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
        self.assertIn("Minimum length is enforced", system_prompt)
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

        original_candidate = candidate()
        verified, trace = VerifierService(client).verify(
            candidate=original_candidate,
            task=task(),
            seed_pack=seed_pack(),
            taxonomy_context=self.taxonomy_context,
            revalidate=lambda _: DeterministicValidationResult(valid=True),
        )

        self.assertEqual(verified, original_candidate)
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

        with self.assertLogs(
            "pii_factory.application.verification",
            level="INFO",
        ) as captured:
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
        diagnostic_log = "\n".join(captured.output)
        self.assertIn("verifier judge feedback", diagnostic_log)
        self.assertIn("verifier original tagged_text:", diagnostic_log)
        self.assertIn("verifier fixed tagged_text:", diagnostic_log)
        self.assertIn(repaired.tagged_text, diagnostic_log)

    def test_missing_context_entity_is_fixed_and_added_to_metadata(self) -> None:
        original = candidate().copy(update={
            "tagged_text": (
                "Chị <PERSON>Lò Thị Cẩy</PERSON> hẹn tái khám ngày 15/05/2024."
            ),
        })
        repaired = RepairResult(
            tagged_text=(
                "Chị <PERSON>Lò Thị Cẩy</PERSON> hẹn tái khám ngày "
                "<DATE>15/05/2024</DATE>."
            ),
            entities=[
                GeneratedEntity(label="PERSON", value="Lò Thị Cẩy"),
                GeneratedEntity(label="DATE", value="15/05/2024"),
            ],
            token_usage=token_usage(),
            latency_ms=2,
            model="offline-verifier",
            prompt_version="repair.test",
        )
        taxonomy_context = self.taxonomy_context.copy(update={
            "available_labels": [
                LabelGenerationContext(
                    label="DATE",
                    definition="Ngày lịch cụ thể.",
                    rule="Gắn toàn bộ biểu thức ngày.",
                )
            ],
        })
        client = FakeVerifierClient(
            [
                VerifierDecision(
                    status="FIXABLE",
                    score=85,
                    issues=[VerificationIssue(
                        type="MISSING_ANNOTATION",
                        severity="low",
                        field="tagged_text",
                        reason="Ngày 15/05/2024 chưa được gắn nhãn.",
                        suggested_fix="Gắn DATE và bổ sung entity metadata.",
                    )],
                    edits=[VerificationEdit(
                        action="add_tag",
                        label="DATE",
                        value="15/05/2024",
                        occurrence=1,
                        reason="Giá trị đứng sau cụm 'hẹn tái khám ngày' nên là DATE.",
                    )],
                    token_usage=token_usage(),
                    latency_ms=2,
                    model="offline-verifier",
                    prompt_version="judge.test",
                ),
                decision("PASS"),
            ],
            repair_result=repaired,
        )

        verified, trace = VerifierService(client).verify(
            candidate=original,
            task=task().copy(update={"annotation_labels": ["PERSON", "DATE"]}),
            seed_pack=seed_pack(),
            taxonomy_context=taxonomy_context,
            revalidate=lambda _: DeterministicValidationResult(valid=True),
        )

        self.assertEqual(trace.outcome, "FIXED")
        self.assertEqual(verified.entities[-1].label, "DATE")
        judge_payload = json.loads(client.judge_messages[0][1]["content"])
        self.assertEqual(
            judge_payload["taxonomy_context"]["available_labels"][0]["label"],
            "DATE",
        )
        self.assertIn(
            "MISSING_ANNOTATION",
            client.judge_messages[0][0]["content"],
        )
        repair_payload = json.loads(client.repair_messages[0][1]["content"])
        self.assertEqual(repair_payload["edits"][0]["action"], "add_tag")
        self.assertIn("hẹn tái khám", repair_payload["edits"][0]["reason"])

    def test_second_local_repair_avoids_regeneration_and_aggregates_usage(self) -> None:
        person = seed_pack().positive_entities[0].value
        original = candidate().copy(update={
            "tagged_text": (
                f"<PERSON>{person}</PERSON> hẹn lúc 10:30 với kỹ thuật viên."
            ),
        })
        first_repair = RepairResult(
            tagged_text=(
                f"<PERSON>{person}</PERSON> hẹn lúc "
                "<TIME>10:30</TIME> với kỹ thuật viên."
            ),
            entities=[
                GeneratedEntity(label="PERSON", value=person),
                GeneratedEntity(label="TIME", value="10:30"),
            ],
            token_usage=token_usage(),
            latency_ms=2,
            model="offline-verifier",
            prompt_version="repair.test",
        )
        second_repair = RepairResult(
            tagged_text=(
                f"<PERSON>{person}</PERSON> hẹn lúc "
                "<TIME>10:30</TIME> với "
                "<JOB_TITLE>kỹ thuật viên</JOB_TITLE>."
            ),
            entities=[
                GeneratedEntity(label="PERSON", value=person),
                GeneratedEntity(label="TIME", value="10:30"),
                GeneratedEntity(
                    label="JOB_TITLE",
                    value="kỹ thuật viên",
                ),
            ],
            token_usage=token_usage(),
            latency_ms=3,
            model="offline-verifier",
            prompt_version="repair.test",
        )
        client = FakeVerifierClient(
            [
                decision("FIXABLE"),
                decision("FIXABLE"),
                decision("PASS"),
            ],
            repair_result=[first_repair, second_repair],
        )

        verified, trace = VerifierService(
            client,
            max_repairs_per_candidate=2,
        ).verify(
            candidate=original,
            task=task().copy(update={
                "annotation_labels": [
                    "PERSON",
                    "TIME",
                    "JOB_TITLE",
                ],
            }),
            seed_pack=seed_pack(),
            taxonomy_context=self.taxonomy_context,
            revalidate=lambda _: DeterministicValidationResult(valid=True),
        )

        self.assertEqual(trace.outcome, "FIXED")
        self.assertIn(
            "<JOB_TITLE>kỹ thuật viên</JOB_TITLE>",
            verified.tagged_text,
        )
        self.assertEqual(len(client.repair_messages), 2)
        self.assertEqual(len(client.judge_messages), 3)
        self.assertEqual(trace.repair.token_usage.total_tokens, 30)
        self.assertEqual(trace.final_judge.token_usage.total_tokens, 30)
        self.assertEqual(trace.repair.latency_ms, 5)

    def test_repair_restores_every_repeated_seed_occurrence(self) -> None:
        person = seed_pack().positive_entities[0].value
        original = candidate().copy(update={
            "tagged_text": (
                f"<PERSON>{person}</PERSON> gọi cho <PERSON>{person}</PERSON>, "
                f"sau đó <PERSON>{person}</PERSON> xác nhận ngày 15/05/2024."
            ),
            "entities": [
                GeneratedEntity(label="PERSON", value=person),
                GeneratedEntity(label="PERSON", value=person),
                GeneratedEntity(label="PERSON", value=person),
            ],
        })
        repaired = RepairResult(
            tagged_text=(
                f"<PERSON>{person}</PERSON> gọi cho {person}, sau đó {person} "
                "xác nhận ngày <DATE>15/05/2024</DATE>."
            ),
            entities=[
                GeneratedEntity(label="PERSON", value=person),
                GeneratedEntity(label="DATE", value="15/05/2024"),
            ],
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
            candidate=original,
            task=task().copy(update={"annotation_labels": ["PERSON", "DATE"]}),
            seed_pack=seed_pack(),
            taxonomy_context=self.taxonomy_context,
            revalidate=lambda _: DeterministicValidationResult(valid=True),
        )

        self.assertEqual(trace.outcome, "FIXED")
        self.assertEqual(
            verified.tagged_text.count(f"<PERSON>{person}</PERSON>"),
            3,
        )
        self.assertEqual(
            sum(entity.label == "PERSON" for entity in verified.entities),
            3,
        )
        self.assertEqual(verified.entities[-1].label, "DATE")

    def test_repair_restores_every_repeated_non_seed_entity_occurrence(self) -> None:
        person = seed_pack().positive_entities[0].value
        original = candidate().copy(update={
            "tagged_text": (
                f"<PERSON>{person}</PERSON> xác nhận ngày 15/05/2024, "
                "sau đó nhắc lại ngày 15/05/2024."
            ),
        })
        repaired = RepairResult(
            tagged_text=(
                f"<PERSON>{person}</PERSON> xác nhận ngày "
                "<DATE>15/05/2024</DATE>, sau đó nhắc lại ngày 15/05/2024."
            ),
            entities=[
                GeneratedEntity(label="PERSON", value=person),
                GeneratedEntity(label="DATE", value="15/05/2024"),
            ],
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
            candidate=original,
            task=task().copy(update={"annotation_labels": ["PERSON", "DATE"]}),
            seed_pack=seed_pack(),
            taxonomy_context=self.taxonomy_context,
            revalidate=lambda _: DeterministicValidationResult(valid=True),
        )

        self.assertEqual(trace.outcome, "FIXED")
        self.assertEqual(
            verified.tagged_text.count("<DATE>15/05/2024</DATE>"),
            2,
        )
        self.assertEqual(
            sum(entity.label == "DATE" for entity in verified.entities),
            2,
        )

    def test_repair_removes_unsupported_added_ticket_id_but_keeps_support_ticket(self) -> None:
        person = seed_pack().positive_entities[0].value
        original = candidate().copy(update={
            "tagged_text": (
                f"<PERSON>{person}</PERSON> kiểm tra mã đơn hàng #MED789012 "
                "và phiếu hỗ trợ SR-2026-001."
            ),
        })
        repaired = RepairResult(
            tagged_text=(
                f"<PERSON>{person}</PERSON> kiểm tra mã đơn hàng "
                "<TICKET_ID>#MED789012</TICKET_ID> và phiếu hỗ trợ "
                "<TICKET_ID>SR-2026-001</TICKET_ID>."
            ),
            entities=[
                GeneratedEntity(label="PERSON", value=person),
                GeneratedEntity(label="TICKET_ID", value="#MED789012"),
                GeneratedEntity(label="TICKET_ID", value="SR-2026-001"),
            ],
            token_usage=token_usage(),
            latency_ms=2,
            model="offline-verifier",
            prompt_version="repair.test",
        )
        client = FakeVerifierClient(
            [decision("FIXABLE"), decision("PASS")],
            repair_result=repaired,
        )

        verified, _ = VerifierService(client).verify(
            candidate=original,
            task=task().copy(update={
                "annotation_labels": ["PERSON", "TICKET_ID"],
            }),
            seed_pack=seed_pack(),
            taxonomy_context=self.taxonomy_context,
            revalidate=lambda _: DeterministicValidationResult(valid=True),
        )

        self.assertIn("mã đơn hàng #MED789012", verified.tagged_text)
        self.assertNotIn("<TICKET_ID>#MED789012</TICKET_ID>", verified.tagged_text)
        self.assertIn(
            "<TICKET_ID>SR-2026-001</TICKET_ID>",
            verified.tagged_text,
        )

    def test_repair_may_split_composite_address_without_changing_seed_surface(self) -> None:
        address = "34 Nguyễn Chí Thanh, Ba Đình, Hà Nội"
        address_seed_pack = seed_pack().copy(update={
            "positive_entities": [PositiveEntitySeed(
                label="ADDRESS",
                value=address,
                semantic_role="service_address",
            )],
        })
        original = candidate().copy(update={
            "tagged_text": f"Giao tại <ADDRESS>{address}</ADDRESS>.",
            "entities": [GeneratedEntity(label="ADDRESS", value=address)],
        })
        repaired = RepairResult(
            tagged_text=(
                "Giao tại <ADDRESS>34 Nguyễn Chí Thanh</ADDRESS>, "
                "<LOCATION>Ba Đình, Hà Nội</LOCATION>."
            ),
            entities=[
                GeneratedEntity(label="ADDRESS", value="34 Nguyễn Chí Thanh"),
                GeneratedEntity(label="LOCATION", value="Ba Đình, Hà Nội"),
            ],
            token_usage=token_usage(),
            latency_ms=2,
            model="offline-verifier",
            prompt_version="repair.test",
        )
        split_decision = VerifierDecision(
            status="FIXABLE",
            score=85,
            issues=[VerificationIssue(
                type="BOUNDARY",
                severity="low",
                field="tagged_text",
                reason="ADDRESS đang chứa địa giới hành chính.",
                suggested_fix="Tách ADDRESS và LOCATION.",
            )],
            edits=[VerificationEdit(
                action="split_tag",
                source_label="ADDRESS",
                source_value=address,
                segments=[
                    {"label": "ADDRESS", "value": "34 Nguyễn Chí Thanh"},
                    {"label": "LOCATION", "value": "Ba Đình, Hà Nội"},
                ],
                reason=(
                    "Số nhà và tên đường là ADDRESS; quận và thành phố là LOCATION."
                ),
            )],
            token_usage=token_usage(),
            latency_ms=2,
            model="offline-verifier",
            prompt_version="judge.test",
        )
        client = FakeVerifierClient(
            [split_decision, decision("PASS")],
            repair_result=repaired,
        )

        verified, trace = VerifierService(client).verify(
            candidate=original,
            task=task().copy(update={
                "focus_labels": ["ADDRESS"],
                "annotation_labels": ["ADDRESS", "LOCATION"],
            }),
            seed_pack=address_seed_pack,
            taxonomy_context=self.taxonomy_context,
            revalidate=lambda _: DeterministicValidationResult(valid=True),
        )

        self.assertEqual(trace.outcome, "FIXED")
        self.assertEqual(
            verified.tagged_text.replace("<ADDRESS>", "").replace("</ADDRESS>", "")
            .replace("<LOCATION>", "").replace("</LOCATION>", ""),
            f"Giao tại {address}.",
        )

    def test_repair_replaces_only_human_template_artifact(self) -> None:
        person = seed_pack().positive_entities[0].value
        original = candidate().copy(update={
            "tagged_text": (
                f"Bộ phận [Tên Công ty] tiếp nhận hồ sơ của "
                f"<PERSON>{person}</PERSON>."
            ),
        })
        repaired = RepairResult(
            tagged_text=(
                f"Bộ phận dịch vụ tiếp nhận hồ sơ của "
                f"<PERSON>{person}</PERSON>."
            ),
            entities=[GeneratedEntity(label="PERSON", value=person)],
            token_usage=token_usage(),
            latency_ms=2,
            model="offline-verifier",
            prompt_version="repair.test",
        )
        artifact_decision = VerifierDecision(
            status="FIXABLE",
            score=82,
            issues=[VerificationIssue(
                type="TEMPLATE_ARTIFACT",
                severity="low",
                field="tagged_text",
                reason="[Tên Công ty] là trường điền còn sót.",
                suggested_fix="Thay bằng mô tả bộ phận chung, không tạo PII mới.",
            )],
            edits=[VerificationEdit(
                action="replace_template_artifact",
                source_value="[Tên Công ty]",
                replacement="dịch vụ",
                reason="Đây là trường điền chưa hoàn tất, không phải entity hợp lệ.",
            )],
            token_usage=token_usage(),
            latency_ms=2,
            model="offline-verifier",
            prompt_version="judge.test",
        )
        client = FakeVerifierClient(
            [artifact_decision, decision("PASS")],
            repair_result=repaired,
        )

        verified, trace = VerifierService(client).verify(
            candidate=original,
            task=task(),
            seed_pack=seed_pack(),
            taxonomy_context=self.taxonomy_context,
            revalidate=lambda _: DeterministicValidationResult(valid=True),
        )

        self.assertEqual(trace.outcome, "FIXED")
        self.assertNotIn("[Tên Công ty]", verified.tagged_text)
        self.assertIn("Bộ phận dịch vụ", verified.tagged_text)

    def test_content_regenerate_and_rejected_route_without_repair(self) -> None:
        content_regenerate = VerifierDecision(
            status="REGENERATE",
            score=45,
            issues=[VerificationIssue(
                type="INSUFFICIENT_CONTEXT",
                severity="high",
                field="tagged_text",
                reason="Entity role cannot be determined from the context.",
                suggested_fix="Regenerate with a coherent event.",
            )],
            token_usage=token_usage(),
            latency_ms=2,
            model="offline-verifier",
            prompt_version="judge.test",
        )
        for expected_status, routed_decision in (
            ("REGENERATE", content_regenerate),
            ("REJECTED", decision("REJECTED")),
        ):
            with self.subTest(status=expected_status):
                client = FakeVerifierClient([routed_decision])
                with self.assertRaises(VerificationRoutingError) as raised:
                    VerifierService(client).verify(
                        candidate=candidate(),
                        task=task(),
                        seed_pack=seed_pack(),
                        taxonomy_context=self.taxonomy_context,
                        revalidate=lambda _: DeterministicValidationResult(valid=True),
                    )
                self.assertEqual(raised.exception.status, expected_status)
                self.assertEqual(client.repair_messages, [])

    def test_boundary_regenerate_is_normalized_to_fixed(self) -> None:
        repaired = RepairResult(
            tagged_text="Chị <PERSON>Lò Thị Cẩy</PERSON> đã gửi hồ sơ.",
            entities=[GeneratedEntity(label="PERSON", value="Lò Thị Cẩy")],
            token_usage=token_usage(),
            latency_ms=2,
            model="offline-verifier",
            prompt_version="repair.test",
        )
        client = FakeVerifierClient(
            [decision("REGENERATE"), decision("PASS")],
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
        self.assertEqual(verified.tagged_text, repaired.tagged_text)

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
