import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pii_factory.application.value_bank import (
    InvalidValueBankError,
    ValueBankClassError,
    ValueBankEmptyClassError,
    ValueBankEntityProvider,
    ValueBankLanguageError,
    resolve_value_bank_path,
)


class ValueBankEntityProviderTests(unittest.TestCase):
    def test_default_bank_path_resolves_outside_the_process_working_directory(self) -> None:
        expected = Path(__file__).resolve().parents[1] / "PII_Value_Bank"
        with TemporaryDirectory() as directory, patch(
            "pii_factory.application.value_bank.Path.cwd",
            return_value=Path(directory),
        ):
            resolved = resolve_value_bank_path("PII_Value_Bank")

        self.assertEqual(resolved, expected)

    def test_validate_required_classes_fails_before_sampling(self) -> None:
        with TemporaryDirectory() as directory:
            self._write_bank(
                Path(directory) / "vi_pii_value_pools.json",
                {"PERSON": ["Nguyễn An"]},
            )

            with self.assertRaisesRegex(ValueBankClassError, "EMAIL"):
                ValueBankEntityProvider(directory).validate(
                    language="vi",
                    required_labels=["PERSON", "EMAIL"],
                )

    def test_stable_partitions_do_not_share_person_values(self) -> None:
        full_values = set(
            ValueBankEntityProvider("PII_Value_Bank").values_for(
                "vi",
                "PERSON",
            )
        )
        partition_values = [
            set(ValueBankEntityProvider(
                "PII_Value_Bank",
                partition_index=index,
                partition_count=5,
            ).values_for("vi", "PERSON"))
            for index in range(5)
        ]

        self.assertEqual(set().union(*partition_values), full_values)
        for left in range(len(partition_values)):
            for right in range(left + 1, len(partition_values)):
                self.assertTrue(
                    partition_values[left].isdisjoint(
                        partition_values[right]
                    )
                )

    def test_empty_partition_falls_back_to_the_full_class_pool(self) -> None:
        with TemporaryDirectory() as directory:
            self._write_bank(
                Path(directory) / "vi_pii_value_pools.json",
                {"PREFIX": ["Ông"]},
            )
            provider = ValueBankEntityProvider(
                directory,
                partition_index=9,
                partition_count=10,
            )

            self.assertEqual(provider.values_for("vi", "PREFIX"), ("Ông",))
            self.assertEqual(
                provider.generate("PREFIX", "vi", random.Random(174)),
                "Ông",
            )

    def test_partition_exhaustion_uses_unused_values_from_the_full_pool(self) -> None:
        with TemporaryDirectory() as directory:
            self._write_bank(
                Path(directory) / "vi_pii_value_pools.json",
                {"PERSON": ["An", "Em", "Bình"]},
            )
            provider = ValueBankEntityProvider(
                directory,
                partition_index=0,
                partition_count=2,
            )
            partition_values = tuple(provider.values_for("vi", "PERSON"))
            self.assertTrue(partition_values)
            excluded = set(partition_values)

            selected = provider.generate(
                "PERSON",
                "vi",
                random.Random(174),
                excluded_values=excluded,
            )

            self.assertIn(selected, {"An", "Em", "Bình"} - excluded)

    def test_partition_fallback_is_reproducible(self) -> None:
        with TemporaryDirectory() as directory:
            self._write_bank(
                Path(directory) / "vi_pii_value_pools.json",
                {"ETHNICITY": ["Kinh", "Tày"]},
            )
            provider = ValueBankEntityProvider(
                directory,
                partition_index=7,
                partition_count=10,
            )

            first = provider.generate(
                "ETHNICITY",
                "vi",
                random.Random(174),
            )
            repeated = provider.generate(
                "ETHNICITY",
                "vi",
                random.Random(174),
            )

            self.assertEqual(first, repeated)

    def test_mixed_spelled_and_numeric_pin_or_cvv_values_are_filtered(self) -> None:
        with TemporaryDirectory() as directory:
            self._write_bank(
                Path(directory) / "vi_pii_value_pools.json",
                {
                    "PIN": ["một tám 42", "0-1-3-7", "một tám bốn hai"],
                    "CVV": ["tám 013", "9 0 3", "chín không ba"],
                },
            )
            provider = ValueBankEntityProvider(directory)

            self.assertEqual(
                provider.values_for("vi", "PIN"),
                ("0-1-3-7", "một tám bốn hai"),
            )
            self.assertEqual(
                provider.values_for("vi", "CVV"),
                ("9 0 3", "chín không ba"),
            )

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

    def test_small_real_classes_are_available_in_all_ten_shards(self) -> None:
        for label in ("PREFIX", "ETHNICITY"):
            for partition_index in range(10):
                with self.subTest(
                    label=label,
                    partition_index=partition_index,
                ):
                    provider = ValueBankEntityProvider(
                        "PII_Value_Bank",
                        partition_index=partition_index,
                        partition_count=10,
                    )
                    values = provider.values_for("vi", label)
                    first = provider.generate(
                        label,
                        "vi",
                        random.Random(174),
                    )
                    repeated = provider.generate(
                        label,
                        "vi",
                        random.Random(174),
                    )

                    self.assertTrue(values)
                    self.assertIn(first, values)
                    self.assertEqual(first, repeated)

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

    def test_address_sampling_excludes_embedded_location_suffixes(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "vi_pii_value_pools.json"
            self._write_bank(
                path,
                {
                    "ADDRESS": [
                        "15 Thái Hà, Đống Đa, Hà Nội",
                        "Phòng 512, 15 Thái Hà",
                        "28 đường Nguyễn Văn Linh",
                    ],
                    "LOCATION": [
                        "Đống Đa",
                        "Hà Nội",
                        "Đống Đa, Hà Nội",
                    ],
                },
            )
            before = path.read_bytes()

            values = ValueBankEntityProvider(directory).values_for(
                "vi",
                "ADDRESS",
            )

            self.assertEqual(
                values,
                (
                    "Phòng 512, 15 Thái Hà",
                    "28 đường Nguyễn Văn Linh",
                ),
            )
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
