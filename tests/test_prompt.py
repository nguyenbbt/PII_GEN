import json
import unittest

from data_generator_worker.contracts import DataGenerationRequest
from data_generator_worker.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_messages, build_prompt_messages


class DataGeneratorPromptTests(unittest.TestCase):
    def test_separates_stable_system_prompt_from_dynamic_user_envelope(self) -> None:
        request = DataGenerationRequest.from_dict({
            "attempt_no": 1,
            "task": {
                "task_id": "task-1", "language": "vi", "focus_labels": ["TIME"],
                "difficulty": "medium", "sample_type": "hard_negative", "max_entities": 2,
            },
            "seed_pack": {
                "seed_pack_id": "seed-1", "task_id": "task-1", "sample_type": "hard_negative",
                "hard_negative_mode": "decoy_only", "positive_entities": [],
                "decoys": [{
                    "strategy_id": "time_as_sla_code", "target_label": "TIME", "value": "SLA-TIME-03",
                    "family": "business_identifier", "semantic_type": "service_level_code",
                    "negative_labels": ["TIME"], "possible_collision_labels": [],
                    "required_context_cues": ["mã build"], "forbidden_context_cues": ["lúc"],
                }],
                "context_frame": {
                    "frame_id": "technical_support_visit", "domain": "technical_support",
                    "document_type": "support_ticket", "tone": "formal", "max_sentences": 3,
                    "supported_labels": ["TIME"],
                },
            },
            "taxonomy_context": {
                "taxonomy_version_id": "taxonomy-v1",
                "sample_type": "hard_negative",
                "focus_label": {
                    "label": "TIME",
                    "definition": "Mốc thời gian cụ thể",
                    "rule": "Chỉ gán cho mốc giờ cụ thể.",
                    "examples": [
                        {
                            "id": f"time-hard-{index}",
                            "expected_tagged_text": f"Ví dụ {index}",
                            "rationale": "Phân biệt theo ngữ cảnh.",
                        }
                        for index in range(1, 4)
                    ],
                },
                "robin_labels": [],
            },
        })

        messages = build_messages(request)

        self.assertEqual(PROMPT_VERSION, "data-generator.v10.0.0")
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertNotIn("task-1", messages[0]["content"])
        self.assertIn("# Generation Request", messages[1]["content"])
        self.assertIn("SLA-TIME-03", messages[1]["content"])
        self.assertIn("Chỉ gán cho mốc giờ cụ thể.", messages[1]["content"])
        self.assertIn("once by default", messages[1]["content"])
        self.assertIn("confirmation, correction, quotation, or cross-reference", messages[1]["content"])
        self.assertIn("return entities as an empty array", messages[1]["content"])
        self.assertIn('"entities"', messages[0]["content"])
        self.assertIn("Few-Shot Use Policy", messages[0]["content"])
        self.assertIn("Do not copy or closely paraphrase", messages[0]["content"])
        self.assertIn("sentence structure", messages[0]["content"])
        envelope = json.loads(messages[1]["content"].split("```json\n", 1)[1].split("\n```", 1)[0])
        self.assertEqual(
            envelope["taxonomy_guidance"]["focus_label"]["label"],
            "TIME",
        )
        self.assertEqual(
            len(envelope["taxonomy_guidance"]["focus_label"]["examples"]),
            3,
        )
        self.assertNotIn("retrieved_knowledge", envelope)
        instance_rules = " ".join(envelope["hard_negative_instance_rules"])
        self.assertIn("Use it once by default", instance_rules)
        self.assertIn("copy one required_context_cue unchanged", instance_rules)
        self.assertIn("keep both occurrences in one sentence", instance_rules)

    def test_pure_negative_has_no_entity_or_tag_requirement(self) -> None:
        messages = build_prompt_messages(
            task={
                "task_id": "negative-1", "language": "vi", "focus_labels": ["TIME"],
                "difficulty": "easy", "sample_type": "pure_negative", "max_entities": 1,
            },
            seed_pack={
                "sample_type": "pure_negative", "positive_entities": [], "decoys": [],
                "content_seeds": {
                    "domain": "customer_support", "document_type": "internal_note", "tone": "neutral",
                    "generic_roles": ["bộ phận kỹ thuật"], "actions": ["kiểm tra yêu cầu"], "objects": ["biểu mẫu"],
                },
                "context_frame": {"frame_id": "generic_internal_note"},
            },
            taxonomy_context={"TIME": {"definition": "Mốc thời gian cụ thể"}},
        )

        content = messages[1]["content"]
        envelope = json.loads(content.split("```json\n", 1)[1].split("\n```", 1)[0])
        self.assertIn("Do not use XML tags", envelope["sample_type_rules"][2])
        self.assertEqual(envelope["positive_entities"], [])

    def test_mixed_hard_negative_requires_semantic_disambiguation(self) -> None:
        messages = build_prompt_messages(
            task={
                "task_id": "mixed-1", "language": "vi", "focus_labels": ["CARD_ISSUER"],
                "difficulty": "hard", "sample_type": "hard_negative", "max_entities": 2,
            },
            seed_pack={
                "sample_type": "hard_negative", "hard_negative_mode": "mixed_contrastive",
                "positive_entities": [{
                    "label": "CARD_ISSUER", "value": "Visa",
                    "semantic_role": "payment_card_network",
                }],
                "decoys": [{
                    "strategy_id": "card_issuer_as_config_flag", "target_label": "CARD_ISSUER",
                    "value": "PAYMENT-VISA-ENABLED", "semantic_type": "payment_feature_flag",
                    "required_context_cues": ["tùy chọn cấu hình"],
                    "forbidden_context_cues": ["thẻ được phát hành"],
                }],
                "context_frame": {"frame_id": "payment_dispute"},
            },
            taxonomy_context={"CARD_ISSUER": {"definition": "Tổ chức hoặc mạng phát hành thẻ"}},
        )

        envelope = json.loads(messages[1]["content"].split("```json\n", 1)[1].split("\n```", 1)[0])
        rules = " ".join(envelope["sample_type_rules"])
        instance_rules = " ".join(envelope["hard_negative_instance_rules"])

        self.assertEqual(
            envelope["positive_entities"][0]["value"],
            "[CARD_ISSUER_1]",
        )
        self.assertNotIn('"value": "Visa"', messages[1]["content"])
        self.assertIn("assigned taxonomy label", rules)
        self.assertIn("travel visa", rules)
        self.assertIn("bank-card network", rules)
        self.assertIn("single realistic", rules)
        self.assertIn("removing the decoy clause", rules)
        self.assertIn("End the decoy clause with an operational consequence", rules)
        self.assertIn("operational cause, input, or object", instance_rules)
        self.assertIn("actual record or payload processed through this decoy", instance_rules)
        self.assertIn("Do not append a disclaimer", instance_rules)
        self.assertNotIn("không phải dữ liệu cá nhân", rules + instance_rules)


if __name__ == "__main__":
    unittest.main()
