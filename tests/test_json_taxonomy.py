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
        self.assertGreaterEqual(len(api_key.decoy_blueprints), 3)
        self.assertGreaterEqual(
            sum(not blueprint.technical for blueprint in api_key.decoy_blueprints),
            2,
        )
        hard_negative_ids = {
            example.id for example in api_key.examples.hard_negative
        }
        self.assertTrue(all(
            set(blueprint.source_example_ids) <= hard_negative_ids
            for blueprint in api_key.decoy_blueprints
        ))

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

    def test_rejects_blueprint_with_unknown_source_example(self) -> None:
        source = json.loads(
            Path("pii_taxonomy_rules.json").read_text(encoding="utf-8")
        )
        source[0]["DECOY_BLUEPRINTS"][0]["source_example_ids"] = [
            "missing-hard-negative-example"
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-blueprint-reference.json"
            path.write_text(
                json.dumps(source, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                JsonTaxonomyParser().parse_file(path)

    def test_rejects_blueprint_that_bypasses_technical_quota(self) -> None:
        source = json.loads(
            Path("pii_taxonomy_rules.json").read_text(encoding="utf-8")
        )
        blueprint = source[0]["DECOY_BLUEPRINTS"][0]
        blueprint["family"] = "technical_schema"
        blueprint["technical"] = False

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-technical-flag.json"
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
        self.assertEqual(context.decoy_labels, [])

    def test_selected_decoy_target_gets_its_three_hard_negative_examples(self) -> None:
        context = TaxonomyContextSelector().select(
            self.taxonomy,
            self._task(sample_type="hard_negative"),
            decoy_target_codes=["EMAIL"],
        )
        email = next(
            label for label in self.taxonomy.labels if label.code == "EMAIL"
        )

        self.assertEqual(
            [label.label for label in context.decoy_labels],
            ["EMAIL"],
        )
        self.assertEqual(
            context.decoy_labels[0].examples,
            email.examples.hard_negative,
        )
        self.assertEqual(
            [label.label for label in context.robin_labels],
            ["EMAIL", "PHONE"],
        )
        self.assertTrue(all(
            not label.examples for label in context.robin_labels
        ))

    def test_selected_blueprint_source_example_is_never_sampled_out(self) -> None:
        label = next(
            item
            for item in self.taxonomy.labels
            if item.code == "EMAIL"
        )
        extra = label.examples.hard_negative[0].copy(update={
            "id": "email_hard_negative_extra",
            "expected_tagged_text": "Một ví dụ bổ sung hoàn toàn khác.",
        })
        expanded_label = label.copy(update={
            "examples": label.examples.copy(update={
                "hard_negative": [
                    *label.examples.hard_negative,
                    extra,
                ],
            }),
        })
        snapshot = self.taxonomy.copy(update={
            "labels": [
                expanded_label
                if item.code == "EMAIL"
                else item
                for item in self.taxonomy.labels
            ],
        })

        context = TaxonomyContextSelector().select(
            snapshot,
            self._task(sample_type="hard_negative"),
            decoy_target_codes=["EMAIL"],
            decoy_source_example_ids={
                "EMAIL": ["email_hard_negative_extra"],
            },
        )

        selected_ids = {
            example.id
            for example in context.decoy_labels[0].examples
        }
        self.assertEqual(len(selected_ids), 3)
        self.assertIn("email_hard_negative_extra", selected_ids)

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
