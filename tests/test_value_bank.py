import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pii_factory.application.value_bank import (
    InvalidValueBankError,
    ValueBankClassError,
    ValueBankEmptyClassError,
    ValueBankEntityProvider,
    ValueBankLanguageError,
)


class ValueBankEntityProviderTests(unittest.TestCase):
    def test_current_banks_cover_all_taxonomy_classes_in_three_languages(self) -> None:
        provider = ValueBankEntityProvider("PII_Value_Bank")
        taxonomy = json.loads(
            Path("pii_taxonomy_rules.json").read_text(encoding="utf-8")
        )
        expected_classes = {item["LABEL"] for item in taxonomy}

        for language in ("vi", "en", "de"):
            with self.subTest(language=language):
                available = {
                    label
                    for label in expected_classes
                    if provider.values_for(language, label)
                }
                self.assertEqual(available, expected_classes)

    def test_sampling_is_reproducible_and_uses_requested_language_and_class(self) -> None:
        provider = ValueBankEntityProvider("PII_Value_Bank")

        for language in ("vi", "en", "de"):
            for label in ("PERSON", "ADDRESS", "PHONE"):
                with self.subTest(language=language, label=label):
                    first = provider.generate(label, language, random.Random(174))
                    repeated = provider.generate(label, language, random.Random(174))
                    self.assertEqual(first, repeated)
                    self.assertIn(first, provider.values_for(language, label))

    def test_exclusions_prevent_duplicate_values_in_one_sample(self) -> None:
        provider = ValueBankEntityProvider("PII_Value_Bank")
        rng = random.Random(42)
        first = provider.generate("PERSON", "vi", rng)
        second = provider.generate(
            "PERSON",
            "vi",
            rng,
            excluded_values={first},
        )

        self.assertNotEqual(first, second)

    def test_source_duplicates_and_case_variants_are_retained_without_editing_source(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "vi_pii_value_pools.json"
            self._write_bank(
                path,
                {"PERSON": ["Visa", "VISA", "Visa"]},
            )
            before = path.read_bytes()

            values = ValueBankEntityProvider(directory).values_for("vi", "PERSON")

            self.assertEqual(values, ("Visa", "VISA", "Visa"))
            self.assertEqual(path.read_bytes(), before)

    def test_exclusions_are_case_sensitive(self) -> None:
        with TemporaryDirectory() as directory:
            self._write_bank(
                Path(directory) / "vi_pii_value_pools.json",
                {"CARD_ISSUER": ["Visa", "VISA"]},
            )
            provider = ValueBankEntityProvider(directory)

            selected = provider.generate(
                "CARD_ISSUER",
                "vi",
                random.Random(1),
                excluded_values={"Visa"},
            )

            self.assertEqual(selected, "VISA")

    def test_language_file_mapping_can_override_default_filename(self) -> None:
        with TemporaryDirectory() as directory:
            custom_path = Path(directory) / "english-bank.json"
            custom_path.write_text(
                json.dumps({
                    "version": 1,
                    "entity_values": {
                        "PERSON": [{
                            "value": "Configured English Name",
                            "locale": "en",
                        }]
                    },
                }),
                encoding="utf-8",
            )
            provider = ValueBankEntityProvider(
                directory,
                {"en": "english-bank.json"},
            )

            self.assertEqual(
                provider.values_for("en", "PERSON"),
                ("Configured English Name",),
            )
            with self.assertRaisesRegex(
                ValueBankLanguageError,
                "no configured file",
            ):
                provider.values_for("de", "PERSON")

    def test_absolute_language_file_does_not_require_base_directory(self) -> None:
        with TemporaryDirectory() as directory:
            custom_path = Path(directory) / "english-bank.json"
            custom_path.write_text(
                json.dumps({
                    "version": 1,
                    "entity_values": {
                        "PERSON": [{
                            "value": "Absolute English Name",
                            "locale": "en",
                        }]
                    },
                }),
                encoding="utf-8",
            )
            provider = ValueBankEntityProvider(
                Path(directory) / "missing-base",
                {"en": str(custom_path.resolve())},
            )

            self.assertEqual(
                provider.values_for("en", "PERSON"),
                ("Absolute English Name",),
            )

    def test_missing_directory_language_and_class_have_explicit_errors(self) -> None:
        with TemporaryDirectory() as directory:
            missing_directory = Path(directory) / "missing"
            with self.assertRaisesRegex(
                ValueBankLanguageError,
                "directory does not exist",
            ):
                ValueBankEntityProvider(missing_directory).values_for("vi", "PERSON")

            provider = ValueBankEntityProvider(directory)
            with self.assertRaisesRegex(ValueBankLanguageError, "no file"):
                provider.values_for("de", "PERSON")

            self._write_bank(
                Path(directory) / "vi_pii_value_pools.json",
                {"PERSON": ["Nguyễn An"]},
            )
            with self.assertRaisesRegex(ValueBankClassError, "does not define class"):
                provider.values_for("vi", "EMAIL")

    def test_empty_class_invalid_json_and_locale_mismatch_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "vi_pii_value_pools.json"

            path.write_text(
                json.dumps(
                    {"version": 1, "entity_values": {"PERSON": []}},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueBankEmptyClassError, "has no values"):
                ValueBankEntityProvider(directory).values_for("vi", "PERSON")

            path.write_text("{not-json", encoding="utf-8")
            with self.assertRaisesRegex(InvalidValueBankError, "valid JSON"):
                ValueBankEntityProvider(directory).values_for("vi", "PERSON")

            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "entity_values": {
                            "PERSON": [{"value": "Jane Doe", "locale": "en"}]
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(InvalidValueBankError, "expected 'vi'"):
                ValueBankEntityProvider(directory).values_for("vi", "PERSON")

    @staticmethod
    def _write_bank(path: Path, values_by_class: dict[str, list[str]]) -> None:
        payload = {
            "version": 1,
            "entity_values": {
                label: [
                    {"value": value, "locale": "vi"}
                    for value in values
                ]
                for label, values in values_by_class.items()
            },
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
