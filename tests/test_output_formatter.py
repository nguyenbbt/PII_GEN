import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pii_factory.application.formatting import JsonDatasetWriter, OutputFormatter
from pii_factory.domain.models import (
    FormattedSample,
    FormattedTokenUsage,
    GeneratedEntity,
)


class OutputFormatterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.formatter = OutputFormatter()

    def test_formats_vietnamese_entities_with_end_exclusive_offsets(self) -> None:
        tagged_text = (
            "Chị <PERSON>Lò Thị Cẩy</PERSON>, dân tộc <ETHNICITY>Cống</ETHNICITY>, "
            "vay vốn tại <CARD_ISSUER>Agribank</CARD_ISSUER>."
        )
        entities = [
            GeneratedEntity(label="PERSON", value="Lò Thị Cẩy"),
            GeneratedEntity(label="ETHNICITY", value="Cống"),
            GeneratedEntity(label="CARD_ISSUER", value="Agribank"),
        ]

        sample = self.formatter.format(
            tagged_text=tagged_text,
            entities=entities,
            allowed_labels=["PERSON", "ETHNICITY", "CARD_ISSUER"],
        )

        self.assertEqual(
            sample.text,
            "Chị Lò Thị Cẩy, dân tộc Cống, vay vốn tại Agribank.",
        )
        self.assertEqual(
            [entity.dict() for entity in sample.entities],
            [
                {"label": "PERSON", "start": 4, "end": 14, "text": "Lò Thị Cẩy"},
                {"label": "ETHNICITY", "start": 24, "end": 28, "text": "Cống"},
                {"label": "CARD_ISSUER", "start": 42, "end": 50, "text": "Agribank"},
            ],
        )
        for entity in sample.entities:
            self.assertEqual(sample.text[entity.start:entity.end], entity.text)

    def test_offsets_use_python_unicode_code_points_after_emoji(self) -> None:
        sample = self.formatter.format(
            tagged_text="😊 Gọi <PERSON>Hà</PERSON> ngay.",
            entities=[GeneratedEntity(label="PERSON", value="Hà")],
            allowed_labels=["PERSON"],
        )

        self.assertEqual(sample.entities[0].start, 6)
        self.assertEqual(sample.entities[0].end, 8)
        self.assertEqual(sample.text[6:8], "Hà")

    def test_offsets_are_recomputed_after_missing_annotation_is_added(self) -> None:
        sample = self.formatter.format(
            tagged_text=(
                "Chị <PERSON>Mai Huyền</PERSON> hẹn tái khám ngày "
                "<DATE>15/05/2024</DATE>."
            ),
            entities=[
                GeneratedEntity(label="PERSON", value="Mai Huyền"),
                GeneratedEntity(label="DATE", value="15/05/2024"),
            ],
            allowed_labels=["PERSON", "DATE"],
        )

        self.assertEqual(
            [(item.label, item.text) for item in sample.entities],
            [("PERSON", "Mai Huyền"), ("DATE", "15/05/2024")],
        )
        for entity in sample.entities:
            self.assertEqual(
                sample.text[entity.start:entity.end],
                entity.text,
            )

    def test_repeated_value_produces_one_span_per_occurrence(self) -> None:
        sample = self.formatter.format(
            tagged_text=(
                "<PERSON>Mai Huyền</PERSON> xác nhận hồ sơ của "
                "<PERSON>Mai Huyền</PERSON>."
            ),
            entities=[
                GeneratedEntity(label="PERSON", value="Mai Huyền"),
                GeneratedEntity(label="PERSON", value="Mai Huyền"),
            ],
            allowed_labels=["PERSON"],
        )

        self.assertEqual(len(sample.entities), 2)
        self.assertNotEqual(sample.entities[0].start, sample.entities[1].start)
        for entity in sample.entities:
            self.assertEqual(sample.text[entity.start:entity.end], "Mai Huyền")

    def test_pure_negative_has_an_empty_entity_array(self) -> None:
        sample = self.formatter.format(
            tagged_text="Bộ phận kỹ thuật đã chuyển biểu mẫu.",
            entities=[],
            allowed_labels=["PERSON"],
        )

        self.assertEqual(sample.entities, [])
        self.assertEqual(sample.text, "Bộ phận kỹ thuật đã chuyển biểu mẫu.")

    def test_rejects_nested_unknown_or_mismatched_tags(self) -> None:
        cases = (
            (
                "<PERSON>Chị <ETHNICITY>Cống</ETHNICITY></PERSON>",
                [
                    GeneratedEntity(label="PERSON", value="Chị Cống"),
                    GeneratedEntity(label="ETHNICITY", value="Cống"),
                ],
            ),
            (
                "<UNKNOWN>ABC</UNKNOWN>",
                [GeneratedEntity(label="UNKNOWN", value="ABC")],
            ),
            (
                "<PERSON>Lan</PERSON>",
                [GeneratedEntity(label="PERSON", value="Linh")],
            ),
        )
        for tagged_text, entities in cases:
            with self.subTest(tagged_text=tagged_text):
                with self.assertRaises(ValueError):
                    self.formatter.format(
                        tagged_text=tagged_text,
                        entities=entities,
                        allowed_labels=["PERSON", "ETHNICITY"],
                    )


class JsonDatasetWriterTests(unittest.TestCase):
    def test_writes_partial_then_publishes_exact_json_array(self) -> None:
        samples = [
            FormattedSample(
                entities=[],
                text="Không chứa PII.",
            )
        ]
        with TemporaryDirectory() as directory:
            writer = JsonDatasetWriter(Path(directory))

            partial_path = writer.write_partial(
                run_name="../vi data 001",
                run_id="run-123",
                samples=samples,
            )
            final_path = writer.finalize(
                run_name="../vi data 001",
                run_id="run-123",
                samples=samples,
            )

            self.assertFalse(partial_path.exists())
            self.assertTrue(final_path.exists())
            self.assertEqual(final_path.parent, Path(directory))
            self.assertNotIn("..", final_path.name)
            payload = json.loads(final_path.read_text(encoding="utf-8"))
            self.assertEqual(payload, [{"entities": [], "text": "Không chứa PII."}])

    def test_writes_per_sample_input_and_output_token_usage(self) -> None:
        samples = [
            FormattedSample(
                entities=[],
                text="Không chứa PII.",
                token_usage=FormattedTokenUsage(
                    input_tokens=120,
                    output_tokens=45,
                ),
            )
        ]
        with TemporaryDirectory() as directory:
            final_path = JsonDatasetWriter(Path(directory)).finalize(
                run_name="token-output",
                run_id="run-token",
                samples=samples,
            )

            payload = json.loads(final_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload[0]["token_usage"],
                {"input_tokens": 120, "output_tokens": 45},
            )


if __name__ == "__main__":
    unittest.main()
