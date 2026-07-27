import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from pii_factory.bootstrap import build_pipeline
from pii_factory.domain.models import CreateRunRequest, RunConfig


class DistributionRunConfigTests(unittest.TestCase):
    def test_sample_structure_supports_contract_chat_and_custom(self) -> None:
        contract = RunConfig(
            num_samples=1,
            sample_structure={"type": "contract"},
        )
        chat = RunConfig(
            num_samples=1,
            sample_structure={"type": "chat"},
        )
        custom = RunConfig(
            num_samples=1,
            sample_structure={
                "type": "custom",
                "custom_instruction": "Biên bản bàn giao thiết bị giữa nhân viên và công ty.",
            },
        )

        self.assertEqual(contract.sample_structure.type, "contract")
        self.assertEqual(chat.sample_structure.type, "chat")
        self.assertEqual(custom.sample_structure.type, "custom")
        self.assertIn("bàn giao thiết bị", custom.sample_structure.custom_instruction)

    def test_custom_structure_requires_instruction_and_presets_reject_it(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "custom_instruction is required",
        ):
            RunConfig(
                num_samples=1,
                sample_structure={"type": "custom"},
            )

        with self.assertRaisesRegex(
            ValidationError,
            "custom_instruction is only allowed",
        ):
            RunConfig(
                num_samples=1,
                sample_structure={
                    "type": "chat",
                    "custom_instruction": "Ignore the chat preset.",
                },
            )

    def test_quality_checks_default_off_while_legacy_validation_fields_parse(self) -> None:
        config = RunConfig(
            num_samples=1,
            validation={
                "novelty_mode": "enforce",
                "near_duplicate_threshold": 0.8,
            },
        )

        self.assertFalse(config.validation.quality_checks_enabled)
        self.assertEqual(config.validation.novelty_mode, "enforce")
        self.assertEqual(config.validation.near_duplicate_threshold, 0.8)

    def test_parses_documented_config_and_normalises_language(self) -> None:
        raw_config = json.loads(Path("configs/run_config.example.json").read_text(encoding="utf-8"))
        config = RunConfig.parse_obj(raw_config)

        self.assertEqual(config.run_name, "vi_data_001")
        self.assertEqual(config.num_samples, raw_config["num_samples"])
        self.assertEqual(config.language, "vi")
        self.assertIsNone(config.focus_labels)
        self.assertEqual(config.focus_label, "PERSON")
        self.assertEqual(config.robin_labels, [
            "PHONE", "EMAIL", "ADDRESS", "DATE", "TIME", "MONEY", "URL",
            "IP", "CARD_NUMBER", "PLATE", "PASSPORT", "MEDICAL_INFO", "LOCATION",
        ])
        self.assertEqual(config.robin_selection.min_per_sample, 2)
        self.assertEqual(config.robin_selection.max_per_sample, 4)
        self.assertEqual(config.max_attempts, 3)
        self.assertEqual(config.faker.locale, "vi_VN")
        self.assertEqual(config.hard_negative.max_decoys, 1)
        self.assertEqual(config.hard_negative.mode, "mixed_contrastive")
        self.assertEqual(config.sample_structure.type, "contract")
        self.assertFalse(config.validation.quality_checks_enabled)
        self.assertNotIn(
            "informal_chat",
            config.optional_constraint_distribution,
        )

    def test_random_task_decisions_are_reproducible_and_use_selected_labels(self) -> None:
        config = RunConfig.parse_obj(json.loads(Path("configs/run_config.example.json").read_text(encoding="utf-8")))
        decisions = []
        for _ in range(2):
            pipeline, repository, _ = build_pipeline(offline=True)
            taxonomy = pipeline.taxonomy_service.import_json(
                Path("pii_taxonomy_rules.json")
            )
            run = pipeline.create_run(CreateRunRequest(taxonomy_version_id=taxonomy.version_id, config=config))
            decisions.append([
                (
                    task.focus_labels, task.difficulty, task.sample_type,
                    task.optional_constraints, task.max_entities, task.max_attempts, task.random_seed,
                )
                for task in repository.list_tasks(run.run_id)
            ])

        self.assertEqual(decisions[0], decisions[1])
        selected = set(config.label_pool or [])
        self.assertTrue(all(set(task[0]).issubset(selected) for task in decisions[0]))
        self.assertTrue(any(set(task[0]) - {"TIME"} for task in decisions[0]))

    def test_focus_label_is_anchored_while_robin_labels_vary_reproducibly(self) -> None:
        config = RunConfig(
            num_samples=20,
            minimum_per_label=3,
            focus_label="TIME",
            robin_labels=["PERSON", "DATE", "EMAIL", "PHONE"],
            robin_selection={"min_per_sample": 1, "max_per_sample": 2},
            difficulty_distribution={"easy": 0.0, "medium": 1.0, "hard": 0.0},
            sample_type_distribution={"positive": 1.0, "pure_negative": 0.0, "hard_negative": 0.0},
            max_entities={"easy": 3, "medium": 3, "hard": 3},
            complexity_limits={"positive": 3, "pure_negative": 1, "hard_negative": 4},
            random_seed=174,
        )

        decisions = []
        for _ in range(2):
            pipeline, repository, _ = build_pipeline(offline=True)
            taxonomy = pipeline.taxonomy_service.import_json(
                Path("pii_taxonomy_rules.json")
            )
            run = pipeline.create_run(CreateRunRequest(taxonomy_version_id=taxonomy.version_id, config=config))
            decisions.append([
                (task.focus_label, task.robin_labels, task.focus_labels)
                for task in repository.list_tasks(run.run_id)
            ])

        self.assertEqual(decisions[0], decisions[1])
        self.assertTrue(all(item[0] == "TIME" for item in decisions[0]))
        self.assertTrue(all(item[2][0] == "TIME" for item in decisions[0]))
        self.assertTrue(all(1 <= len(item[1]) <= 2 for item in decisions[0]))
        self.assertTrue(all(set(item[1]).issubset({"PERSON", "DATE", "EMAIL", "PHONE"}) for item in decisions[0]))
        self.assertGreater(len({tuple(item[1]) for item in decisions[0]}), 1)

    def test_chat_structure_flows_to_tasks_with_reproducible_variants(self) -> None:
        config = RunConfig(
            num_samples=12,
            focus_labels=["PERSON"],
            sample_structure={"type": "chat"},
            sample_type_distribution={
                "positive": 1.0,
                "pure_negative": 0.0,
                "hard_negative": 0.0,
            },
            random_seed=174,
        )

        planned_variants = []
        for _ in range(2):
            pipeline, repository, _ = build_pipeline(offline=True)
            taxonomy = pipeline.taxonomy_service.import_json(
                Path("pii_taxonomy_rules.json")
            )
            run = pipeline.create_run(
                CreateRunRequest(
                    taxonomy_version_id=taxonomy.version_id,
                    config=config,
                )
            )
            tasks = repository.list_tasks(run.run_id)
            self.assertTrue(
                all(task.sample_structure.type == "chat" for task in tasks)
            )
            planned_variants.append(
                [task.diversity_profile.document_structure for task in tasks]
            )

        self.assertEqual(planned_variants[0], planned_variants[1])
        self.assertEqual(
            set(planned_variants[0]),
            {"friend_chat", "customer_support_chat"},
        )

    def test_focus_and_robin_labels_flow_to_seed_and_result(self) -> None:
        pipeline, repository, _ = build_pipeline(offline=True)
        taxonomy = pipeline.taxonomy_service.import_json(
            Path("pii_taxonomy_rules.json")
        )
        config = RunConfig(
            num_samples=1,
            focus_label="TIME",
            robin_labels=["DATE"],
            robin_selection={"min_per_sample": 1, "max_per_sample": 1},
            sample_type_distribution={"positive": 1.0, "pure_negative": 0.0, "hard_negative": 0.0},
            max_entities={"easy": 2, "medium": 2, "hard": 2},
            complexity_limits={"positive": 2, "pure_negative": 1, "hard_negative": 4},
        )
        run = pipeline.create_run(CreateRunRequest(taxonomy_version_id=taxonomy.version_id, config=config))

        result = pipeline.generate_pending(run.run_id, limit=1)[0]
        task = repository.list_tasks(run.run_id)[0]

        self.assertEqual(task.focus_label, "TIME")
        self.assertEqual(task.robin_labels, ["DATE"])
        self.assertEqual([entity.label for entity in result.entities], ["TIME", "DATE"])
        self.assertEqual(result.generation_query.focus_label, "TIME")
        self.assertEqual(result.generation_query.robin_labels, ["DATE"])

    def test_focus_label_remains_positive_in_mixed_hard_negative_samples(self) -> None:
        pipeline, repository, _ = build_pipeline(offline=True)
        taxonomy = pipeline.taxonomy_service.import_json(
            Path("pii_taxonomy_rules.json")
        )
        config = RunConfig(
            num_samples=6,
            batch_size=6,
            minimum_per_label=3,
            focus_label="PERSON",
            robin_labels=["PHONE", "EMAIL", "DATE"],
            robin_selection={"min_per_sample": 1, "max_per_sample": 2},
            difficulty_distribution={"easy": 0.0, "medium": 1.0, "hard": 0.0},
            sample_type_distribution={"positive": 0.0, "pure_negative": 0.0, "hard_negative": 1.0},
            max_entities={"easy": 3, "medium": 3, "hard": 3},
            hard_negative={
                "mode": "mixed_contrastive", "min_decoys": 1, "max_decoys": 1,
                "max_focus_labels": 3, "unsupported_label_policy": "rebuild_task",
            },
            complexity_limits={"positive": 3, "pure_negative": 1, "hard_negative": 4},
            random_seed=174,
        )
        run = pipeline.create_run(CreateRunRequest(taxonomy_version_id=taxonomy.version_id, config=config))

        results = pipeline.generate_pending(run.run_id, limit=6)
        tasks = repository.list_tasks(run.run_id)

        self.assertEqual(len(results), 6)
        self.assertTrue(all(task.focus_labels[0] == "PERSON" for task in tasks))
        self.assertTrue(all(result.entities[0].label == "PERSON" for result in results))
        self.assertTrue(all(result.generation_query.robin_labels for result in results))

    def test_focus_label_mode_rejects_sample_types_that_cannot_emit_the_anchor(self) -> None:
        with self.assertRaisesRegex(ValidationError, "pure_negative must be 0"):
            RunConfig(
                num_samples=10,
                focus_label="TIME",
                robin_labels=["DATE"],
                sample_type_distribution={"positive": 0.9, "pure_negative": 0.1, "hard_negative": 0.0},
            )

        with self.assertRaisesRegex(ValidationError, "mixed_contrastive"):
            RunConfig(
                num_samples=10,
                focus_label="TIME",
                robin_labels=["DATE"],
                sample_type_distribution={"positive": 0.0, "pure_negative": 0.0, "hard_negative": 1.0},
                hard_negative={"mode": "decoy_only"},
            )

    def test_rejects_distribution_that_does_not_sum_to_one(self) -> None:
        with self.assertRaises(ValidationError):
            RunConfig(
                num_samples=10,
                difficulty_distribution={"easy": 0.3, "medium": 0.3, "hard": 0.3},
            )

    def test_legacy_auxiliary_date_policy_is_migrated_without_injecting_date(self) -> None:
        config = RunConfig(
            num_samples=1,
            hard_negative={"unsupported_label_policy": "auxiliary_date"},
        )

        self.assertEqual(config.hard_negative.unsupported_label_policy, "rebuild_task")

    def test_offline_llm_path_uses_validated_faker_seed_pack(self) -> None:
        pipeline, _, _ = build_pipeline(offline=True)
        taxonomy = pipeline.taxonomy_service.import_json(
            Path("pii_taxonomy_rules.json")
        )
        config = RunConfig(
            num_samples=1, focus_labels=["PERSON", "EMAIL"],
            sample_type_distribution={"positive": 1.0, "pure_negative": 0.0, "hard_negative": 0.0},
            max_entities={"easy": 2, "medium": 2, "hard": 2},
        )
        run = pipeline.create_run(CreateRunRequest(taxonomy_version_id=taxonomy.version_id, config=config))

        result = pipeline.generate_pending(run.run_id, limit=1)[0]

        self.assertTrue(result.entities)
        self.assertEqual(len({entity.value.casefold() for entity in result.entities}), len(result.entities))
        self.assertTrue(result.seed_pack_id)

    def test_omitted_focus_labels_uses_taxonomy_and_batch_size_caps_generation(self) -> None:
        pipeline, _, _ = build_pipeline(offline=True)
        taxonomy = pipeline.taxonomy_service.import_json(
            Path("pii_taxonomy_rules.json")
        )
        config = RunConfig(num_samples=4, batch_size=2, focus_labels=None, random_seed=42)
        run = pipeline.create_run(CreateRunRequest(taxonomy_version_id=taxonomy.version_id, config=config))

        first_batch = pipeline.generate_pending(run.run_id, limit=100)

        self.assertEqual(len(run.config.focus_labels or []), 44)
        self.assertEqual(len(first_batch), 2)

    def test_hard_negative_tasks_respect_complexity_budget(self) -> None:
        pipeline, repository, _ = build_pipeline(offline=True)
        taxonomy = pipeline.taxonomy_service.import_json(
            Path("pii_taxonomy_rules.json")
        )
        config = RunConfig(
            num_samples=10, focus_labels=["PERSON", "ADDRESS", "DATE", "EMAIL", "PHONE"],
            sample_type_distribution={"positive": 0.0, "pure_negative": 0.0, "hard_negative": 1.0},
            optional_constraint_distribution={"informal_chat": 1.0, "light_typo": 1.0},
        )
        run = pipeline.create_run(CreateRunRequest(taxonomy_version_id=taxonomy.version_id, config=config))

        for task in repository.list_tasks(run.run_id):
            score = config.hard_negative.min_decoys + len(task.optional_constraints)
            self.assertLessEqual(score, config.complexity_limits.hard_negative)
            self.assertEqual(len(task.focus_labels), 1)

    def test_run_rejects_label_alias_absent_from_taxonomy(self) -> None:
        pipeline, _, _ = build_pipeline(offline=True)
        taxonomy = pipeline.taxonomy_service.import_json(
            Path("pii_taxonomy_rules.json")
        )
        config = RunConfig(num_samples=1, focus_labels=["PHONE_NUMBER"])

        with self.assertRaisesRegex(ValueError, "absent from taxonomy"):
            pipeline.create_run(CreateRunRequest(
                taxonomy_version_id=taxonomy.version_id,
                config=config,
            ))


if __name__ == "__main__":
    unittest.main()
