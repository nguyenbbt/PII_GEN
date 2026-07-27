import json
import unittest

from pii_factory.application.novelty import NoveltyGuard
from pii_factory.bootstrap import build_pipeline
from pii_factory.domain.models import (
    CreateRunRequest,
    DataGenerationResult,
    DeterministicValidationResult,
    GeneratedEntity,
    GenerationQuery,
    TokenUsage,
    RunConfig,
    TaxonomyLabel,
    TaxonomySnapshot,
)


def previous_result(text: str, label: str = "TIME", value: str = "14:30") -> DataGenerationResult:
    return DataGenerationResult(
        task_id=f"task-{value}", attempt_no=1, seed_pack_id="seed", context_frame_id="frame",
        generation_query=GenerationQuery(
            language="vi", focus_labels=[label], constraints=[], difficulty="medium",
            sample_type="positive", max_entities=1,
        ),
        entities=[GeneratedEntity(label=label, value=value)], tagged_text=text,
        token_usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2, money_cost=0),
        latency_ms=1, model="test", prompt_version="test", output_hash="hash",
        seed_validation=DeterministicValidationResult(valid=True),
        output_validation=DeterministicValidationResult(valid=True),
        sentence_skeleton="hẹn lúc <time>.",
    )


class NoveltyGuardTests(unittest.TestCase):
    def test_rejects_same_sentence_skeleton_with_different_entity_value(self) -> None:
        guard = NoveltyGuard(near_duplicate_threshold=0.9)
        previous = [previous_result("Hẹn lúc <TIME>14:30</TIME>.")]

        assessment = guard.assess(
            tagged_text="Hẹn lúc <TIME>09:15</TIME>.",
            entities=[GeneratedEntity(label="TIME", value="09:15")],
            decoy_values=[],
            previous_results=previous,
        )

        self.assertFalse(assessment.valid)
        self.assertEqual(assessment.issues[0].scope, "TEXT")
        self.assertEqual(assessment.issues[0].type, "duplicate_sentence_skeleton")

    def test_routes_reused_open_ended_entity_to_seed_regeneration(self) -> None:
        guard = NoveltyGuard()
        previous = [previous_result("Hẹn lúc <TIME>14:30</TIME>.")]

        assessment = guard.assess(
            tagged_text="Ca trực bắt đầu vào <TIME>14:30</TIME>.",
            entities=[GeneratedEntity(label="TIME", value="14:30")],
            decoy_values=[],
            previous_results=previous,
        )

        self.assertFalse(assessment.valid)
        self.assertTrue(any(issue.scope == "SEEDS" for issue in assessment.issues))

    def test_allows_repeated_categorical_values(self) -> None:
        guard = NoveltyGuard(categorical_labels=["GENDER"])
        previous = [previous_result("Giới tính: <GENDER>nữ</GENDER>.", "GENDER", "nữ")]

        assessment = guard.assess(
            tagged_text="Hồ sơ ghi nhận <GENDER>nữ</GENDER>.",
            entities=[GeneratedEntity(label="GENDER", value="nữ")],
            decoy_values=[],
            previous_results=previous,
        )

        self.assertTrue(assessment.valid)


class NoveltyPipelineTests(unittest.TestCase):
    def test_enforce_mode_retries_a_duplicate_sentence_skeleton(self) -> None:
        class DuplicateThenDiverseClient:
            def generate(self, messages):
                payload = json.loads(messages[-1]["content"].split("```json\n", 1)[1].split("\n```", 1)[0])
                seed = payload["positive_entities"][0]
                tagged = f"<{seed['label']}>{seed['value']}</{seed['label']}>"
                if payload["reflection"]:
                    text = f"Phiếu công việc ghi nhận ngày {tagged} cho bước kiểm tra tiếp theo."
                else:
                    text = f"Hẹn vào {tagged}."
                return text, [{"label": seed["label"], "value": seed["value"]}], 20, 10, 30

        pipeline, _, events = build_pipeline(offline=True)
        pipeline.generator.client = DuplicateThenDiverseClient()
        taxonomy = TaxonomySnapshot(labels=[TaxonomyLabel(code="DATE", definition="Ngày cụ thể")])
        config = RunConfig(
            num_samples=2, batch_size=2, focus_labels=["DATE"],
            sample_type_distribution={"positive": 1.0, "pure_negative": 0.0, "hard_negative": 0.0},
            validation={
                "quality_checks_enabled": True,
                "novelty_mode": "enforce",
                "near_duplicate_threshold": 0.9,
            },
        )
        run = pipeline.create_run(CreateRunRequest(taxonomy=taxonomy, config=config))

        results = pipeline.generate_pending(run.run_id, 2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[1].attempt_no, 2)
        self.assertNotEqual(results[0].sentence_skeleton, results[1].sentence_skeleton)
        rejected = [event for event in events.list_events() if event.event_type == "data.generation.rejected"]
        self.assertTrue(any(
            issue["type"] == "duplicate_sentence_skeleton"
            for event in rejected
            for issue in event.payload["output_validation"]["issues"]
        ))


if __name__ == "__main__":
    unittest.main()
