import json
import unittest

from data_generator_worker.prompt import build_prompt_messages
from pii_factory.infrastructure.clients import OfflineCompletionClient


def envelope_from(messages: list[dict[str, str]]) -> dict:
    content = messages[-1]["content"]
    return json.loads(content.split("```json\n", 1)[1].split("\n```", 1)[0])


def offline_sample(
    sample_structure: dict,
    document_structure: str,
) -> str:
    messages = build_prompt_messages(
        task={
            "task_id": "offline-structure",
            "language": "vi",
            "focus_labels": ["PERSON"],
            "difficulty": "medium",
            "sample_type": "positive",
            "max_entities": 1,
            "sample_structure": sample_structure,
            "diversity_profile": {
                "document_structure": document_structure,
                "language_register": "neutral",
                "speaker_role": "customer",
                "intent": "provide_update",
                "length_bucket": "medium",
            },
        },
        seed_pack={
            "sample_type": "positive",
            "positive_entities": [
                {"label": "PERSON", "value": "Lò Thị Cẩy"}
            ],
            "decoys": [],
            "context_frame": {"document_type": "support_record"},
        },
        taxonomy_context={},
    )
    tagged_text, _, _, _, _ = OfflineCompletionClient().generate(messages)
    return tagged_text


class PromptDiversityTests(unittest.TestCase):
    def test_offline_chat_has_two_distinct_speakers(self) -> None:
        tagged_text = offline_sample(
            {"type": "chat"},
            "customer_support_chat",
        )
        speaker_names = [
            line.split(":", 1)[0]
            for line in tagged_text.splitlines()
            if ":" in line
        ]

        self.assertEqual(len(tagged_text.splitlines()), 2)
        self.assertEqual(len(set(speaker_names)), 2)
        self.assertIn("Khách hàng", speaker_names)
        self.assertIn("Nhân viên hỗ trợ", speaker_names)

    def test_offline_contract_uses_selected_business_document_variant(self) -> None:
        tagged_text = offline_sample(
            {"type": "contract"},
            "handover_minutes",
        )

        self.assertTrue(tagged_text.startswith("BIÊN BẢN BÀN GIAO"))
        self.assertNotIn("Nhân viên hỗ trợ:", tagged_text)

    def test_offline_custom_uses_instruction_as_context_and_format(self) -> None:
        tagged_text = offline_sample(
            {
                "type": "custom",
                "custom_instruction": "Biên bản bàn giao thiết bị theo dạng checklist",
            },
            "custom_format",
        )

        self.assertIn("Biên bản bàn giao thiết bị theo dạng checklist", tagged_text)
        self.assertIn("<PERSON>[PERSON_1]</PERSON>", tagged_text)

    def test_contract_structure_requests_business_or_administrative_document(self) -> None:
        messages = build_prompt_messages(
            task={
                "task_id": "contract-1",
                "language": "vi",
                "focus_labels": ["PERSON"],
                "difficulty": "medium",
                "sample_type": "positive",
                "max_entities": 2,
                "sample_structure": {"type": "contract"},
                "diversity_profile": {
                    "document_structure": "administrative_record",
                },
            },
            seed_pack={
                "sample_type": "positive",
                "positive_entities": [],
                "decoys": [],
            },
            taxonomy_context={},
        )

        rules = " ".join(envelope_from(messages)["sample_structure_rules"])

        self.assertIn("business or administrative document", rules)
        self.assertIn("administrative record", rules)
        self.assertIn("not a chat conversation", rules)

    def test_chat_structure_requires_exactly_two_speakers(self) -> None:
        messages = build_prompt_messages(
            task={
                "task_id": "chat-1",
                "language": "vi",
                "focus_labels": ["PERSON"],
                "difficulty": "medium",
                "sample_type": "positive",
                "max_entities": 2,
                "sample_structure": {"type": "chat"},
                "diversity_profile": {
                    "document_structure": "customer_support_chat",
                },
            },
            seed_pack={
                "sample_type": "positive",
                "positive_entities": [],
                "decoys": [],
            },
            taxonomy_context={},
        )

        rules = " ".join(envelope_from(messages)["sample_structure_rules"])

        self.assertIn("exactly two speakers", rules)
        self.assertIn("customer-support conversation", rules)
        self.assertIn("2 to 6", rules)

    def test_custom_structure_is_context_only_and_cannot_override_annotation(self) -> None:
        instruction = "Tạo biên bản bàn giao thiết bị theo dạng checklist."
        messages = build_prompt_messages(
            task={
                "task_id": "custom-1",
                "language": "vi",
                "focus_labels": ["PERSON"],
                "difficulty": "medium",
                "sample_type": "positive",
                "max_entities": 2,
                "sample_structure": {
                    "type": "custom",
                    "custom_instruction": instruction,
                },
                "diversity_profile": {
                    "document_structure": "custom_format",
                },
            },
            seed_pack={
                "sample_type": "positive",
                "positive_entities": [],
                "decoys": [],
            },
            taxonomy_context={},
        )

        envelope = envelope_from(messages)
        rules = " ".join(envelope["sample_structure_rules"])

        self.assertEqual(
            envelope["sample_structure"]["custom_instruction"],
            instruction,
        )
        self.assertIn("context and presentation format only", rules)
        self.assertIn("cannot override", rules)
    def test_focus_label_is_central_and_robin_labels_are_supporting_context(self) -> None:
        messages = build_prompt_messages(
            task={
                "task_id": "anchor-1", "language": "vi",
                "focus_labels": ["TIME", "DATE", "PERSON"],
                "focus_label": "TIME", "robin_labels": ["DATE", "PERSON"],
                "difficulty": "medium", "sample_type": "positive", "max_entities": 3,
            },
            seed_pack={"sample_type": "positive", "positive_entities": [], "decoys": []},
            taxonomy_context={},
        )

        rules = " ".join(envelope_from(messages)["realization_rules"])

        self.assertIn("TIME is the mandatory central entity", rules)
        self.assertIn("DATE, PERSON", rules)
        self.assertIn("support the same event", rules)

    def test_profile_and_difficulty_become_explicit_realization_rules(self) -> None:
        messages = build_prompt_messages(
            task={
                "task_id": "task-1", "language": "vi", "focus_labels": ["TIME"],
                "difficulty": "hard", "sample_type": "positive", "max_entities": 2,
                "optional_constraints": ["informal_chat", "abbreviation"],
                "diversity_profile": {
                    "context_frame_id": "travel_booking",
                    "speaker_role": "customer",
                    "intent": "request_action",
                    "document_structure": "short_dialogue",
                    "language_register": "informal",
                    "length_bucket": "long",
                    "entity_format_variants": {"TIME": "12h_meridiem"},
                },
            },
            seed_pack={"sample_type": "positive", "positive_entities": [], "decoys": []},
            taxonomy_context={},
        )

        rules = envelope_from(messages)["realization_rules"]

        self.assertTrue(any("short dialogue" in rule for rule in rules))
        self.assertTrue(any("customer" in rule for rule in rules))
        self.assertTrue(any("multi-clause" in rule for rule in rules))
        self.assertTrue(any("abbreviation" in rule for rule in rules))
        self.assertTrue(any("stock opening" in rule for rule in rules))

    def test_different_profiles_produce_different_rules(self) -> None:
        base_task = {
            "task_id": "task-1", "language": "vi", "focus_labels": ["DATE"],
            "difficulty": "easy", "sample_type": "positive", "max_entities": 1,
            "optional_constraints": [],
        }
        first = envelope_from(build_prompt_messages(
            task={**base_task, "diversity_profile": {"document_structure": "single_sentence", "language_register": "formal"}},
            seed_pack={"sample_type": "positive", "positive_entities": [], "decoys": []},
            taxonomy_context={},
        ))
        second = envelope_from(build_prompt_messages(
            task={**base_task, "diversity_profile": {"document_structure": "form_like_record", "language_register": "concise_technical"}},
            seed_pack={"sample_type": "positive", "positive_entities": [], "decoys": []},
            taxonomy_context={},
        ))

        self.assertNotEqual(first["realization_rules"], second["realization_rules"])


if __name__ == "__main__":
    unittest.main()
