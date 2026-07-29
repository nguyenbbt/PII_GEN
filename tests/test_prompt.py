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

        self.assertEqual(PROMPT_VERSION, "data-generator.v12.0.0")
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
        self.assertIn("Never serialize entity seeds as a comma-separated list", messages[0]["content"])
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
                    "realization_plan": {
                        "blueprint_id": "card_issuer_semantic_ambiguity",
                        "family": "semantic_ambiguity",
                        "contrast_principle": "The surface names a travel document, not a card network.",
                        "anchor_label": "CARD_ISSUER",
                        "relation": "comparison",
                        "discourse_stage": "processing",
                        "evidence_cues": ["hồ sơ thị thực"],
                        "source_example_ids": [
                            "card_issuer_hard_negative_1",
                        ],
                    },
                }],
                "context_frame": {"frame_id": "payment_dispute"},
            },
            taxonomy_context={
                "focus_label": {
                    "label": "CARD_ISSUER",
                    "definition": "Tổ chức hoặc mạng phát hành thẻ",
                    "rule": "Use only in payment-card context.",
                    "examples": [],
                },
                "decoy_labels": [{
                    "label": "CARD_ISSUER",
                    "definition": "Tổ chức hoặc mạng phát hành thẻ",
                    "rule": "Use only in payment-card context.",
                    "examples": [
                        {
                            "id": f"card_issuer_hard_negative_{index}",
                            "expected_tagged_text": f"Ví dụ đối chiếu {index}",
                            "rationale": "Bề mặt giống nhãn nhưng vai trò nghiệp vụ khác.",
                        }
                        for index in range(1, 4)
                    ],
                }],
            },
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
        self.assertIn("infer why their surfaces", rules)
        self.assertIn("realization_plan", rules)
        self.assertIn("single realistic", rules)
        self.assertIn("removing the decoy clause", rules)
        self.assertIn("Do not force the event to end", rules)
        self.assertIn(
            "affirmative business role",
            rules,
        )
        self.assertIn(
            "never write 'không phải/not",
            rules,
        )
        self.assertIn("contrast_principle", instance_rules)
        self.assertIn("anchor_label=CARD_ISSUER", instance_rules)
        self.assertIn(
            "anchor_placeholder=[CARD_ISSUER_1]",
            instance_rules,
        )
        self.assertIn("relation=comparison", instance_rules)
        self.assertIn(
            "same sentence or chat turn",
            instance_rules,
        )
        self.assertIn(
            "adjacent unit only when",
            instance_rules,
        )
        self.assertIn("do not mechanically copy a cue", instance_rules)
        self.assertNotIn(
            "copy one required_context_cue unchanged",
            instance_rules,
        )
        self.assertNotIn("không phải dữ liệu cá nhân", rules + instance_rules)
        self.assertEqual(
            len(envelope["taxonomy_guidance"]["decoy_labels"][0]["examples"]),
            3,
        )
        self.assertIn("Mixed-Decoy Prohibitions", messages[0]["content"])
        self.assertIn("detached final sentence", messages[0]["content"])

    def test_prompt_enforces_numeric_length_entity_density_and_address_boundaries(self) -> None:
        messages = build_prompt_messages(
            task={
                "task_id": "quality-1",
                "language": "vi",
                "focus_labels": ["PERSON", "ADDRESS", "LOCATION", "ZIP_CODE", "EMAIL"],
                "difficulty": "hard",
                "sample_type": "positive",
                "max_entities": 8,
                "sample_structure": {"type": "contract"},
                "length_target": {
                    "bucket": "long",
                    "min_words": 260,
                    "max_words": 400,
                    "unit": "content_units",
                    "min_units": 10,
                    "max_units": 14,
                },
            },
            seed_pack={
                "sample_type": "positive",
                "positive_entities": [
                    {"label": label, "value": f"value-{index}", "semantic_role": "field"}
                    for index, label in enumerate(
                        ("PERSON", "ADDRESS", "LOCATION", "ZIP_CODE", "EMAIL"),
                        start=1,
                    )
                ],
                "decoys": [],
            },
            taxonomy_context={},
        )

        envelope = json.loads(
            messages[1]["content"].split("```json\n", 1)[1].split("\n```", 1)[0]
        )
        rules = " ".join([
            *envelope["sample_structure_rules"],
            *envelope["realization_rules"],
        ])

        self.assertEqual(envelope["required_entity_count"], 5)
        self.assertIn("260 to 400 clean-text words", rules)
        self.assertIn("exceeding 400 is allowed", rules)
        self.assertIn("10 to 14 connected content units", rules)
        self.assertIn("Additional meaningful units are allowed", rules)
        self.assertIn("multiple sentences", rules)
        self.assertIn("comma-separated", rules)
        self.assertIn("street-level", rules)
        self.assertIn("administrative", rules)
        self.assertNotRegex(rules.casefold(), r"\bcompact\b|\bconcise\b")

    def test_prompt_uses_annotation_labels_and_forbids_human_template_fields(self) -> None:
        messages = build_prompt_messages(
            task={
                "task_id": "annotation-pool",
                "language": "vi",
                "focus_labels": ["PERSON"],
                "annotation_labels": ["PERSON", "PLATE", "TICKET_ID", "JOB_TITLE"],
                "difficulty": "medium",
                "sample_type": "positive",
                "max_entities": 4,
                "sample_structure": {"type": "custom", "custom_instruction": "Viết email."},
            },
            seed_pack={
                "sample_type": "positive",
                "positive_entities": [{
                    "label": "PERSON",
                    "value": "Mai Huyền",
                    "semantic_role": "requester",
                }],
                "decoys": [],
            },
            taxonomy_context={},
        )
        envelope = json.loads(
            messages[1]["content"].split("```json\n", 1)[1].split("\n```", 1)[0]
        )

        self.assertEqual(
            envelope["allowed_labels"],
            ["PERSON", "PLATE", "TICKET_ID", "JOB_TITLE"],
        )
        self.assertIn("[Tên Công ty]", messages[0]["content"])
        self.assertIn("finished prose", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
