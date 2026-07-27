import unittest

from pii_factory.application.diversity_metrics import (
    DiversityAuditSample,
    audit_diversity,
    sentence_skeleton,
)


class DiversityMetricsTests(unittest.TestCase):
    def test_sentence_skeleton_ignores_seed_values_but_keeps_label_roles(self) -> None:
        first = sentence_skeleton("Hẹn lúc <TIME>14:30</TIME>.")
        second = sentence_skeleton("Hẹn lúc <TIME>09:15</TIME>.")

        self.assertEqual(first, second)
        self.assertIn("<time>", first)

    def test_audit_reports_entity_uniqueness_and_sentence_duplicates(self) -> None:
        samples = [
            DiversityAuditSample(
                tagged_text="Hẹn lúc <TIME>14:30</TIME>.",
                entities=[{"label": "TIME", "value": "14:30"}],
                context_frame_id="appointment",
            ),
            DiversityAuditSample(
                tagged_text="Hẹn lúc <TIME>09:15</TIME>.",
                entities=[{"label": "TIME", "value": "09:15"}],
                context_frame_id="appointment",
            ),
            DiversityAuditSample(
                tagged_text="Ca trực bắt đầu vào <TIME>14:30</TIME>.",
                entities=[{"label": "TIME", "value": "14:30"}],
                context_frame_id="shift_note",
            ),
        ]

        report = audit_diversity(samples, near_duplicate_threshold=0.8)

        self.assertEqual(report.sample_count, 3)
        self.assertAlmostEqual(report.entity_unique_ratio["TIME"], 2 / 3)
        self.assertAlmostEqual(report.exact_skeleton_duplicate_rate, 1 / 3)
        self.assertEqual(report.context_frame_distribution, {"appointment": 2, "shift_note": 1})

    def test_audit_reports_strategy_constraint_and_format_coverage(self) -> None:
        sample = DiversityAuditSample(
            tagged_text="Mã SLA SLA-TIME-03 đã được cập nhật.",
            entities=[],
            decoy_values=["SLA-TIME-03"],
            context_frame_id="support",
            strategy_ids=["time_as_sla_code"],
            constraints=["informal_chat"],
            entity_format_variants={"TIME": "24h"},
        )

        report = audit_diversity([sample])

        self.assertEqual(report.strategy_distribution, {"time_as_sla_code": 1})
        self.assertEqual(report.constraint_distribution, {"informal_chat": 1})
        self.assertEqual(report.format_variant_distribution, {"TIME:24h": 1})
        self.assertIn("<decoy>", sentence_skeleton(sample.tagged_text, sample.decoy_values))


if __name__ == "__main__":
    unittest.main()
