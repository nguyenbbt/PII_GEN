from decimal import Decimal
import unittest

from pydantic import ValidationError

from pii_factory.domain.models import (
    PipelineTokenUsage,
    TokenUsage,
    VerificationIssue,
    VerificationEdit,
    VerifierDecision,
)


def usage(input_tokens: int = 10, output_tokens: int = 5) -> TokenUsage:
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        money_cost=Decimal("0.001"),
    )


class VerifierContractTests(unittest.TestCase):
    def test_pass_requires_no_issues(self) -> None:
        decision = VerifierDecision(
            status="PASS",
            score=98,
            issues=[],
            token_usage=usage(),
            latency_ms=1,
            model="offline-verifier",
            prompt_version="judge.test",
        )

        self.assertEqual(decision.status, "PASS")

        with self.assertRaises(ValidationError):
            VerifierDecision(
                status="PASS",
                score=98,
                issues=[
                    VerificationIssue(
                        type="WRONG_LABEL",
                        severity="low",
                        field="entities",
                        reason="Sai nhãn.",
                        suggested_fix="Sửa nhãn.",
                    )
                ],
                token_usage=usage(),
                latency_ms=1,
                model="offline-verifier",
                prompt_version="judge.test",
            )

    def test_fixable_requires_only_low_severity_issues(self) -> None:
        low_issue = VerificationIssue(
            type="BOUNDARY",
            severity="low",
            field="tagged_text",
            reason="Boundary chứa dấu câu.",
            suggested_fix="Đưa dấu câu ra ngoài tag.",
        )
        decision = VerifierDecision(
            status="FIXABLE",
            score=85,
            issues=[low_issue],
            token_usage=usage(),
            latency_ms=1,
            model="offline-verifier",
            prompt_version="judge.test",
        )
        self.assertEqual(decision.issues[0].severity, "low")

        with self.assertRaises(ValidationError):
            VerifierDecision(
                status="FIXABLE",
                score=50,
                issues=[low_issue.copy(update={"severity": "high"})],
                token_usage=usage(),
                latency_ms=1,
                model="offline-verifier",
                prompt_version="judge.test",
            )

    def test_verifier_edit_requires_plain_language_reason(self) -> None:
        edit = VerificationEdit(
            action="add_tag",
            label="DATE",
            value="27/10/2023",
            occurrence=1,
            reason="Cụm này đứng sau từ 'ngày' nên là một DATE rõ ngữ cảnh.",
        )
        decision = VerifierDecision(
            status="FIXABLE",
            score=85,
            issues=[VerificationIssue(
                type="MISSING_ANNOTATION",
                severity="low",
                field="tagged_text",
                reason="Ngày chưa được gắn nhãn.",
                suggested_fix="Thêm tag DATE.",
            )],
            edits=[edit],
            token_usage=usage(),
            latency_ms=1,
            model="offline-verifier",
            prompt_version="judge.test",
        )

        self.assertEqual(decision.edits[0].reason, edit.reason)
        with self.assertRaises(ValidationError):
            VerificationEdit(
                action="add_tag",
                label="DATE",
                value="27/10/2023",
                occurrence=1,
            )

    def test_verifier_edit_normalizes_only_zero_based_first_occurrence(self) -> None:
        edit = VerificationEdit(
            action="add_tag",
            label="DATE",
            value="27/10/2023",
            occurrence=0,
            reason="The provider used zero-based indexing for the first occurrence.",
        )

        self.assertEqual(edit.occurrence, 1)
        self.assertEqual(
            VerificationEdit(
                action="add_tag",
                label="DATE",
                value="27/10/2023",
                occurrence="0",
                reason="The provider serialized the zero-based index as text.",
            ).occurrence,
            1,
        )
        with self.assertRaises(ValidationError):
            VerificationEdit(
                action="add_tag",
                label="DATE",
                value="27/10/2023",
                occurrence=-1,
                reason="Negative occurrence indexes are never valid.",
            )

    def test_verifier_contract_normalizes_harmless_json_variations(self) -> None:
        decision = VerifierDecision(
            status=" pass ",
            score=98,
            issues=None,
            edits=None,
            token_usage=usage(),
            latency_ms=1,
            model="offline-verifier",
            prompt_version="judge.test",
        )
        edit = VerificationEdit(
            action=" ADD_TAG ",
            label=" ",
            value=None,
            occurrence=" ",
            segments=None,
            reason="Repair can derive the local target from the matching issue.",
        )
        issue = VerificationIssue(
            type="BOUNDARY",
            severity=" LOW ",
            field="tagged_text",
            reason="The span has a local boundary error.",
            suggested_fix="Move only the boundary.",
        )

        self.assertEqual(decision.status, "PASS")
        self.assertEqual(decision.issues, [])
        self.assertEqual(decision.edits, [])
        self.assertEqual(edit.action, "add_tag")
        self.assertIsNone(edit.label)
        self.assertIsNone(edit.occurrence)
        self.assertEqual(edit.segments, [])
        self.assertEqual(issue.severity, "low")

    def test_verifier_issue_keeps_regenerate_issue_when_suggested_fix_is_null(self) -> None:
        issue = VerificationIssue(
            type="UNNATURAL_TEXT",
            severity="high",
            field="tagged_text",
            reason="The generated content is incoherent.",
            suggested_fix=None,
        )
        decision = VerifierDecision(
            status="REGENERATE",
            score=30,
            issues=[issue],
            edits=[],
            token_usage=usage(),
            latency_ms=1,
            model="offline-verifier",
            prompt_version="judge.test",
        )

        self.assertEqual(decision.status, "REGENERATE")
        self.assertEqual(len(decision.issues), 1)
        self.assertIn("Regenerate", decision.issues[0].suggested_fix)

    def test_pipeline_usage_sums_all_llm_calls(self) -> None:
        summary = PipelineTokenUsage.from_calls(
            generator=[usage(100, 50), usage(80, 40)],
            verifier_judge=[usage(30, 10), usage(25, 8)],
            verifier_repair=[usage(40, 20)],
            verifier_rejudge=[usage(25, 8)],
        )

        self.assertEqual(summary.total.input_tokens, 300)
        self.assertEqual(summary.total.output_tokens, 136)
        self.assertEqual(summary.total.total_tokens, 436)
        self.assertEqual(summary.total.money_cost, Decimal("0.006"))
        self.assertEqual(summary.generator.input_tokens, 180)
        self.assertEqual(summary.generator.output_tokens, 90)
        self.assertEqual(summary.verifier_total().input_tokens, 120)
        self.assertEqual(summary.verifier_total().output_tokens, 46)


if __name__ == "__main__":
    unittest.main()
