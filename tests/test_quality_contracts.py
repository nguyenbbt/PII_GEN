from decimal import Decimal
import unittest

from pydantic import ValidationError

from pii_factory.domain.models import (
    PipelineTokenUsage,
    TokenUsage,
    VerificationIssue,
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


if __name__ == "__main__":
    unittest.main()
