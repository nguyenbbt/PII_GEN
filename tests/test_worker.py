from decimal import Decimal
import unittest

from data_generator_worker.config import Settings
from data_generator_worker.contracts import DataGenerationRequest
from data_generator_worker.llm_client import CompletionResponse
from data_generator_worker.validation import validate_generated_output
from data_generator_worker.worker import DataGeneratorWorker, InMemoryAttemptStore


class FakeLLM:
    def generate(self, messages):
        return CompletionResponse(
            tagged_text="Mình sẽ gọi lúc <TIME>[TIME_1]</TIME>.",
            entities=[{"label": "TIME", "value": "[TIME_1]"}],
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )


class DataGeneratorWorkerTests(unittest.TestCase):
    def test_worker_emits_idempotent_data_generated_event(self) -> None:
        settings = Settings(
            api_key="not-used",
            base_url="https://example.openai.azure.com",
            input_token_price_per_million_usd=Decimal("2.50"),
            output_token_price_per_million_usd=Decimal("10.00"),
            model="azure/gpt-4o",
        )
        request = DataGenerationRequest.from_dict(
            {
                "attempt_no": 1,
                "task": {"task_id": "task-1", "language": "vi", "focus_labels": ["TIME"], "difficulty": "medium", "sample_type": "positive", "max_entities": 2},
                "seed_pack": {
                    "seed_pack_id": "seed-1", "task_id": "task-1", "sample_type": "positive",
                    "positive_entities": [{"label": "TIME", "value": "14:30", "semantic_role": "appointment_time"}],
                    "context_frame": {"frame_id": "support", "domain": "support", "document_type": "note", "tone": "neutral", "max_sentences": 2, "supported_labels": ["TIME"]},
                },
                "taxonomy_context": {
                    "taxonomy_version_id": "taxonomy-v1",
                    "sample_type": "positive",
                    "focus_label": {
                        "label": "TIME",
                        "definition": "Mốc thời gian cụ thể.",
                        "rule": "Gắn đúng giá trị thời gian.",
                        "examples": [],
                    },
                    "robin_labels": [],
                },
            }
        )
        worker = DataGeneratorWorker(FakeLLM(), settings, InMemoryAttemptStore())

        first = worker.process(request)
        duplicate = worker.process(request)

        self.assertEqual(first.event_type, "data.generated")
        self.assertEqual(first.payload.token_usage.total_tokens, 150)
        self.assertEqual(first.payload.token_usage.money_cost, "0.00075000")
        self.assertEqual(first.payload.entities[0].value, "14:30")
        self.assertEqual(
            first.payload.taxonomy_context_used["focus_label"]["label"],
            "TIME",
        )
        self.assertEqual(
            first.to_dict()["payload"]["taxonomy_context_used"],
            first.payload.taxonomy_context_used,
        )
        self.assertEqual(duplicate.event_id, first.event_id)

    def test_worker_accepts_entity_values_that_only_differ_by_case(self) -> None:
        class DuplicateLLM:
            def generate(self, messages):
                return CompletionResponse(
                    tagged_text="<PERSON>[PERSON_1]</PERSON> dùng tên <USERNAME>[USERNAME_1]</USERNAME>.",
                    entities=[
                        {"label": "PERSON", "value": "[PERSON_1]"},
                        {"label": "USERNAME", "value": "[USERNAME_1]"},
                    ],
                    input_tokens=1, output_tokens=1, total_tokens=2,
                )

        settings = Settings(api_key="unused", base_url="https://example.test")
        request = DataGenerationRequest.from_dict({
            "attempt_no": 1,
            "task": {
                "task_id": "duplicate-1", "language": "vi",
                "focus_labels": ["PERSON", "USERNAME"], "difficulty": "medium",
                "sample_type": "positive", "max_entities": 2,
            },
            "seed_pack": {
                "seed_pack_id": "seed-duplicate", "task_id": "duplicate-1", "sample_type": "positive",
                "positive_entities": [
                    {"label": "PERSON", "value": "Nguyễn An", "semantic_role": "name"},
                    {"label": "USERNAME", "value": "nguyễn an", "semantic_role": "username"},
                ],
                "context_frame": {"frame_id": "record", "domain": "test", "document_type": "record", "tone": "neutral", "max_sentences": 2, "supported_labels": ["PERSON", "USERNAME"]},
            },
        })

        event = DataGeneratorWorker(
            DuplicateLLM(),
            settings,
            InMemoryAttemptStore(),
        ).process(request)

        self.assertEqual(
            [entity.value for entity in event.payload.entities],
            ["Nguyễn An", "nguyễn an"],
        )

    def test_output_contract_still_rejects_exact_duplicate_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate entity value"):
            validate_generated_output(
                tagged_text=(
                    "<PERSON>Nguyễn An</PERSON> dùng tên "
                    "<USERNAME>Nguyễn An</USERNAME>."
                ),
                entities=[
                    {"label": "PERSON", "value": "Nguyễn An"},
                    {"label": "USERNAME", "value": "Nguyễn An"},
                ],
                allowed_labels=["PERSON", "USERNAME"],
                required_labels=["PERSON", "USERNAME"],
                sample_type="positive",
                max_entities=2,
            )

    def test_worker_allows_a_natural_decoy_repetition_in_decoy_only_mode(self) -> None:
        class RepeatedDecoyLLM:
            def generate(self, messages):
                return CompletionResponse(
                    tagged_text=(
                        "Mã SLA SLA-TIME-03 được ghi trong phiếu hỗ trợ. "
                        "Nhân viên xác nhận mã SLA cần đối chiếu vẫn là SLA-TIME-03."
                    ),
                    entities=[], input_tokens=20, output_tokens=15, total_tokens=35,
                )

        settings = Settings(api_key="unused", base_url="https://example.test")
        request = DataGenerationRequest.from_dict({
            "attempt_no": 1,
            "task": {
                "task_id": "repeat-decoy-1", "language": "vi", "focus_labels": ["TIME"],
                "difficulty": "hard", "sample_type": "hard_negative", "max_entities": 1,
            },
            "seed_pack": {
                "seed_pack_id": "seed-repeat", "task_id": "repeat-decoy-1",
                "sample_type": "hard_negative", "hard_negative_mode": "decoy_only",
                "decoys": [{
                    "strategy_id": "time_as_sla_code", "target_label": "TIME",
                    "value": "SLA-TIME-03", "family": "business_identifier",
                    "semantic_type": "service_level_code", "negative_labels": ["TIME"],
                    "required_context_cues": ["mã SLA"], "forbidden_context_cues": ["lúc"],
                }],
                "context_frame": {
                    "frame_id": "support", "domain": "support", "document_type": "ticket",
                    "tone": "formal", "max_sentences": 2, "supported_labels": ["TIME"],
                },
            },
        })

        event = DataGeneratorWorker(
            RepeatedDecoyLLM(), settings, InMemoryAttemptStore()
        ).process(request)

        self.assertEqual(event.payload.tagged_text.count("SLA-TIME-03"), 2)
        self.assertEqual(event.payload.entities, [])
