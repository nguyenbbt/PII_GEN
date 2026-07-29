import unittest

from data_generator_worker.placeholders import (
    placeholder_entities,
    replace_entity_placeholders,
)
from data_generator_worker.validation import (
    validate_generated_output,
    validate_seeded_contract,
)
from pii_factory.application.formatting import OutputFormatter


class PlaceholderReplacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.positive_entities = [
            {"label": "PERSON", "value": "Nguyễn An"},
            {"label": "PERSON", "value": "Lê 🧑‍💻 Bình"},
            {"label": "EMAIL", "value": "contact@example.test"},
        ]

    def test_numbering_is_per_class_and_supports_repeated_classes(self) -> None:
        placeholders = placeholder_entities(self.positive_entities)

        self.assertEqual(
            [item["value"] for item in placeholders],
            ["[PERSON_1]", "[PERSON_2]", "[EMAIL_1]"],
        )

    def test_code_inserts_values_then_existing_validator_accepts_output(self) -> None:
        tagged_text, entities = replace_entity_placeholders(
            tagged_text=(
                "<PERSON>[PERSON_1]</PERSON> gặp "
                "<PERSON>[PERSON_2]</PERSON>; email "
                "<EMAIL>[EMAIL_1]</EMAIL>."
            ),
            entities=[
                {"label": "PERSON", "value": "[PERSON_1]"},
                {"label": "PERSON", "value": "[PERSON_2]"},
                {"label": "EMAIL", "value": "[EMAIL_1]"},
            ],
            positive_entities=self.positive_entities,
        )

        validated = validate_generated_output(
            tagged_text=tagged_text,
            entities=entities,
            allowed_labels=["PERSON", "EMAIL"],
            required_labels=["PERSON", "PERSON", "EMAIL"],
            sample_type="positive",
            max_entities=3,
        )
        self.assertIn("<PERSON>Nguyễn An</PERSON>", tagged_text)
        self.assertIn("<PERSON>Lê 🧑‍💻 Bình</PERSON>", tagged_text)
        self.assertEqual(validated, entities)

    def test_offsets_are_computed_after_value_insertion(self) -> None:
        tagged_text, entities = replace_entity_placeholders(
            tagged_text=(
                "😊 <PERSON>[PERSON_1]</PERSON> gặp "
                "<PERSON>[PERSON_2]</PERSON>."
            ),
            entities=[
                {"label": "PERSON", "value": "[PERSON_1]"},
                {"label": "PERSON", "value": "[PERSON_2]"},
            ],
            positive_entities=self.positive_entities[:2],
        )

        sample = OutputFormatter().format(
            tagged_text=tagged_text,
            entities=entities,
            allowed_labels=["PERSON"],
        )

        for entity in sample.entities:
            self.assertEqual(sample.text[entity.start:entity.end], entity.text)
        self.assertEqual(
            [entity.text for entity in sample.entities],
            ["Nguyễn An", "Lê 🧑‍💻 Bình"],
        )

    def test_unknown_placeholder_becomes_generic_non_pii_reference(self) -> None:
        tagged_text, entities = replace_entity_placeholders(
            tagged_text="<PERSON>[PERSON_1]</PERSON> [EMAIL_9]",
            entities=[{"label": "PERSON", "value": "[PERSON_1]"}],
            positive_entities=self.positive_entities[:1],
            language="en",
        )

        self.assertEqual(
            tagged_text,
            "<PERSON>Nguyễn An</PERSON> an internal reference",
        )
        self.assertEqual(
            entities,
            [{"label": "PERSON", "value": "Nguyễn An"}],
        )

    def test_known_bare_placeholder_inside_tag_is_repaired(self) -> None:
        tagged_text, entities = replace_entity_placeholders(
            tagged_text="<PERSON>PERSON_1</PERSON>",
            entities=[{"label": "PERSON", "value": "PERSON_1"}],
            positive_entities=self.positive_entities[:1],
        )

        self.assertEqual(tagged_text, "<PERSON>Nguyễn An</PERSON>")
        self.assertEqual(
            entities,
            [{"label": "PERSON", "value": "Nguyễn An"}],
        )

    def test_untagged_known_placeholder_does_not_leak_seed_value(self) -> None:
        tagged_text, entities = replace_entity_placeholders(
            tagged_text=(
                "Subject: [PERSON_1]\n"
                "Record: <PERSON>[PERSON_1]</PERSON>"
            ),
            entities=[{"label": "PERSON", "value": "[PERSON_1]"}],
            positive_entities=self.positive_entities[:1],
            language="en",
        )

        self.assertEqual(tagged_text.count("Nguyễn An"), 1)
        self.assertIn("Subject: the referenced person", tagged_text)
        validate_seeded_contract(
            tagged_text=tagged_text,
            entities=entities,
            positive_entities=self.positive_entities[:1],
            decoys=[],
        )

    def test_entity_metadata_is_synchronized_from_repeated_tags(self) -> None:
        tagged_text, entities = replace_entity_placeholders(
            tagged_text=(
                "<PERSON>[PERSON_1]</PERSON> confirmed; "
                "<PERSON>[PERSON_1]</PERSON> signed."
            ),
            entities=[{"label": "PERSON", "value": "[PERSON_1]"}],
            positive_entities=self.positive_entities[:1],
            language="en",
        )

        self.assertEqual(len(entities), 2)
        validate_seeded_contract(
            tagged_text=tagged_text,
            entities=entities,
            positive_entities=self.positive_entities[:1],
            decoys=[],
        )

    def test_existing_seed_validator_still_rejects_missing_placeholder(self) -> None:
        bound_text, bound_entities = replace_entity_placeholders(
            tagged_text="Không có người trong hồ sơ.",
            entities=[],
            positive_entities=self.positive_entities[:1],
        )
        with self.assertRaises(ValueError):
            validate_seeded_contract(
                tagged_text=bound_text,
                entities=bound_entities,
                positive_entities=self.positive_entities[:1],
                decoys=[],
            )


if __name__ == "__main__":
    unittest.main()
