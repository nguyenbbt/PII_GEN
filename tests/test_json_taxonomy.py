import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from pii_factory.application.taxonomy_context import TaxonomyContextSelector
from pii_factory.domain.models import GenerationTask
from pii_factory.infrastructure.json_taxonomy import JsonTaxonomyParser


class JsonTaxonomyParserTests(unittest.TestCase):
    def test_parses_rules_and_all_example_groups(self) -> None:
        taxonomy = JsonTaxonomyParser().parse_file(
            Path("pii_taxonomy_rules.json")
        )

        self.assertEqual(len(taxonomy.labels), 44)
        api_key = next(
            label for label in taxonomy.labels if label.code == "API_KEY"
        )
        self.assertTrue(api_key.definition)
        self.assertEqual(len(api_key.rules), 1)
        self.assertEqual(len(api_key.examples.positive), 3)
        self.assertEqual(len(api_key.examples.pure_negative), 3)
        self.assertEqual(len(api_key.examples.hard_negative), 3)
        self.assertEqual(
            api_key.examples.positive[0].id,
            "api_key_positive_1",
        )
        date = next(
            label for label in taxonomy.labels if label.code == "DATE"
        )
        time = next(
            label for label in taxonomy.labels if label.code == "TIME"
        )
        self.assertIn("15 tháng 5 năm nay", date.definition)
        self.assertIn("Exclude leading cue words", date.rules[0])
        self.assertIn("sáng", time.definition)
        self.assertIn("Always exclude UTC, GMT", time.rules[0])

    def test_rejects_duplicate_labels(self) -> None:
        source = json.loads(
            Path("pii_taxonomy_rules.json").read_text(encoding="utf-8")
        )
        source.append(source[0])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(
                json.dumps(source, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                JsonTaxonomyParser().parse_file(path)

    def test_rejects_duplicate_example_ids_within_a_group(self) -> None:
        source = json.loads(
            Path("pii_taxonomy_rules.json").read_text(encoding="utf-8")
        )
        source[0]["EXAMPLES"]["POSITIVE"][1]["id"] = (
            source[0]["EXAMPLES"]["POSITIVE"][0]["id"]
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate-example.json"
            path.write_text(
                json.dumps(source, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                JsonTaxonomyParser().parse_file(path)


class TaxonomyContextSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.taxonomy = JsonTaxonomyParser().parse_file(
            Path("pii_taxonomy_rules.json")
        )

    def test_focus_gets_examples_for_resolved_sample_type_only(self) -> None:
        api_key = next(
            label for label in self.taxonomy.labels
            if label.code == "API_KEY"
        )
        expected_groups = {
            "positive": api_key.examples.positive,
            "pure_negative": api_key.examples.pure_negative,
            "hard_negative": api_key.examples.hard_negative,
        }
        for sample_type, expected_examples in expected_groups.items():
            with self.subTest(sample_type=sample_type):
                task = self._task(sample_type=sample_type)

                context = TaxonomyContextSelector().select(
                    self.taxonomy,
                    task,
                )

                self.assertEqual(context.focus_label.label, "API_KEY")
                self.assertTrue(context.focus_label.definition)
                self.assertTrue(context.focus_label.rule)
                self.assertEqual(len(context.focus_label.examples), 3)
                self.assertEqual(
                    context.focus_label.examples,
                    expected_examples,
                )

    def test_robin_labels_receive_rules_without_examples(self) -> None:
        context = TaxonomyContextSelector().select(
            self.taxonomy,
            self._task(sample_type="positive"),
        )

        self.assertEqual(
            [label.label for label in context.robin_labels],
            ["EMAIL", "PHONE"],
        )
        self.assertTrue(all(label.definition for label in context.robin_labels))
        self.assertTrue(all(label.rule for label in context.robin_labels))
        self.assertTrue(all(not label.examples for label in context.robin_labels))

    def test_selection_is_reproducible_for_the_same_task_seed(self) -> None:
        task = self._task(sample_type="positive")
        selector = TaxonomyContextSelector()

        first = selector.select(self.taxonomy, task)
        second = selector.select(self.taxonomy, task)

        self.assertEqual(first, second)

    @staticmethod
    def _task(*, sample_type: str) -> GenerationTask:
        return GenerationTask(
            run_id="run-1",
            sequence_no=1,
            language="vi",
            focus_labels=["API_KEY", "EMAIL", "PHONE"],
            focus_label="API_KEY",
            robin_labels=["EMAIL", "PHONE"],
            difficulty="medium",
            sample_type=sample_type,
            max_entities=3,
            max_attempts=3,
            random_seed=2468,
        )


if __name__ == "__main__":
    unittest.main()
