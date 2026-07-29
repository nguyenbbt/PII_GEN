import unittest
import json
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from pii_factory.domain.models import RunConfig
from pii_factory.parallel import (
    ParallelGenerationError,
    _run_shard,
    build_shard_configs,
    merge_shard_payloads,
    run_parallel_generation,
)


class ParallelGenerationTests(unittest.TestCase):
    def test_online_50_robin_minimum_is_distributed_across_valid_shards(
        self,
    ) -> None:
        config = RunConfig.parse_obj(json.loads(
            Path("configs/run_config.online-50.json").read_text(
                encoding="utf-8"
            )
        ))

        shards = build_shard_configs(config)

        self.assertEqual(len(shards), 5)
        self.assertEqual(
            sum(
                shard.robin_selection.minimum_per_label
                for shard in shards
            ),
            config.robin_selection.minimum_per_label,
        )
        self.assertEqual(
            [
                shard.robin_selection.minimum_per_label
                for shard in shards
            ],
            [2, 2, 2, 1, 1],
        )
        self.assertEqual(
            [shard.value_bank.partition_index for shard in shards],
            [0, 1, 2, 3, 4],
        )
        self.assertTrue(
            all(shard.value_bank.partition_count == 5 for shard in shards)
        )
        for shard in shards:
            RunConfig.parse_obj(shard.dict())

    def test_builds_unique_seeded_shards_covering_exact_target(self) -> None:
        config = RunConfig(
            run_name="parallel-100",
            num_samples=100,
            batch_size=5,
            focus_label="PERSON",
            robin_labels=["EMAIL", "PHONE", "DATE", "ADDRESS"],
            robin_selection={
                "min_per_sample": 4,
                "max_per_sample": 4,
            },
            sample_type_distribution={
                "positive": 1.0,
                "pure_negative": 0.0,
                "hard_negative": 0.0,
            },
            max_entities={
                "easy": 5,
                "medium": 5,
                "hard": 5,
            },
            complexity_limits={
                "positive": 5,
                "pure_negative": 1,
                "hard_negative": 6,
            },
            parallel_generation={
                "workers": 10,
                "shard_size": 10,
                "max_shard_retries": 2,
            },
            random_seed=174,
        )

        shards = build_shard_configs(config)

        self.assertEqual(len(shards), 10)
        self.assertEqual(sum(shard.num_samples for shard in shards), 100)
        self.assertEqual({shard.num_samples for shard in shards}, {10})
        self.assertEqual(len({shard.random_seed for shard in shards}), 10)
        self.assertEqual(len({shard.run_name for shard in shards}), 10)
        self.assertTrue(all(len(shard.run_name) <= 25 for shard in shards))
        self.assertTrue(
            all(shard.batch_size <= shard.num_samples for shard in shards)
        )
        self.assertTrue(
            all(
                shard.sample_length_distribution
                == config.sample_length_distribution
                for shard in shards
            )
        )

    def test_last_shard_uses_only_remaining_sample_capacity(self) -> None:
        config = RunConfig(
            num_samples=23,
            batch_size=5,
            focus_labels=["PERSON"],
            parallel_generation={
                "workers": 3,
                "shard_size": 10,
                "max_shard_retries": 1,
            },
        )

        shards = build_shard_configs(config)

        self.assertEqual([shard.num_samples for shard in shards], [10, 10, 3])
        self.assertEqual([shard.batch_size for shard in shards], [5, 5, 3])

    def test_shard_retry_accumulates_usage_from_failed_paid_attempt(self) -> None:
        config = RunConfig(
            num_samples=1,
            batch_size=1,
            focus_labels=["PERSON"],
            parallel_generation={
                "workers": 1,
                "shard_size": 1,
                "max_shard_retries": 1,
            },
        )
        failed_payload = {
            "status": "FAILED",
            "accepted_samples": 0,
            "token_usage": self._usage_payload(10, "0.01"),
        }
        successful_payload = {
            "status": "COMPLETED",
            "accepted_samples": 1,
            "token_usage": self._usage_payload(20, "0.02"),
        }
        completed_calls = [
            SimpleNamespace(
                returncode=1,
                stdout=json.dumps(failed_payload),
                stderr="failed after a paid call",
            ),
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(successful_payload),
                stderr="",
            ),
        ]

        with TemporaryDirectory() as directory, patch(
            "pii_factory.parallel.subprocess.run",
            side_effect=completed_calls,
        ):
            root = Path(directory)
            payload = _run_shard(
                shard=config,
                shard_index=1,
                taxonomy_path=root / "taxonomy.json",
                work_directory=root,
                shard_artifact_directory=root / "artifacts",
                offline=False,
            )

        self.assertEqual(payload["token_usage"]["total_tokens"], 30)
        self.assertEqual(payload["token_usage"]["money_cost"], "0.03")
        self.assertEqual(
            payload["token_usage"]["generator"]["total_tokens"],
            18,
        )
        self.assertEqual(
            payload["token_usage"]["verifier"]["total_tokens"],
            12,
        )

    @staticmethod
    def _usage_payload(total_tokens: int, money_cost: str) -> dict:
        generator_tokens = total_tokens * 3 // 5
        verifier_tokens = total_tokens - generator_tokens
        total_cost = Decimal(money_cost)
        return {
            "input_tokens": total_tokens * 2 // 3,
            "output_tokens": total_tokens - (total_tokens * 2 // 3),
            "total_tokens": total_tokens,
            "money_cost": money_cost,
            "generator": {
                "input_tokens": generator_tokens,
                "output_tokens": 0,
                "total_tokens": generator_tokens,
                "money_cost": str(
                    total_cost * generator_tokens / total_tokens
                ),
            },
            "verifier": {
                "input_tokens": verifier_tokens,
                "output_tokens": 0,
                "total_tokens": verifier_tokens,
                "money_cost": str(
                    total_cost * verifier_tokens / total_tokens
                ),
            },
        }

    def test_merge_requires_complete_unique_samples_with_valid_offsets(self) -> None:
        payloads = [
            {
                "status": "COMPLETED",
                "accepted_samples": 1,
                "samples": [
                    {
                        "entities": [
                            {
                                "label": "PERSON",
                                "start": 4,
                                "end": 14,
                                "text": "Lê Minh An",
                            }
                        ],
                        "text": "Chị Lê Minh An đã xác nhận.",
                    }
                ],
                "token_usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "money_cost": "0.01",
                    "generator": {
                        "input_tokens": 6,
                        "output_tokens": 3,
                        "total_tokens": 9,
                        "money_cost": "0.006",
                    },
                    "verifier": {
                        "input_tokens": 4,
                        "output_tokens": 2,
                        "total_tokens": 6,
                        "money_cost": "0.004",
                    },
                },
                "diagnostics": {
                    "generated_candidates": 1,
                    "verifier_rejections": 0,
                    "verifier_candidate_pass_rate": 1.0,
                    "verification_outcomes": {"PASS": 1},
                },
            },
            {
                "status": "COMPLETED",
                "accepted_samples": 1,
                "samples": [
                    {
                        "entities": [],
                        "text": "Không có thông tin định danh.",
                    }
                ],
                "token_usage": {
                    "input_tokens": 20,
                    "output_tokens": 8,
                    "total_tokens": 28,
                    "money_cost": "0.02",
                    "generator": {
                        "input_tokens": 12,
                        "output_tokens": 5,
                        "total_tokens": 17,
                        "money_cost": "0.012",
                    },
                    "verifier": {
                        "input_tokens": 8,
                        "output_tokens": 3,
                        "total_tokens": 11,
                        "money_cost": "0.008",
                    },
                },
                "diagnostics": {
                    "generated_candidates": 2,
                    "verifier_rejections": 1,
                    "verifier_candidate_pass_rate": 0.5,
                    "verification_outcomes": {"PASS": 1},
                },
            },
        ]

        merged = merge_shard_payloads(payloads, expected_samples=2)

        self.assertEqual(len(merged["samples"]), 2)
        self.assertEqual(merged["token_usage"]["input_tokens"], 30)
        self.assertEqual(merged["token_usage"]["money_cost"], "0.03")
        self.assertEqual(
            merged["token_usage"]["generator"]["input_tokens"],
            18,
        )
        self.assertEqual(
            merged["token_usage"]["verifier"]["output_tokens"],
            5,
        )
        self.assertEqual(
            merged["diagnostics"]["generated_candidates"],
            3,
        )
        self.assertAlmostEqual(
            merged["diagnostics"]["verifier_candidate_pass_rate"],
            2 / 3,
        )

    def test_merge_rejects_failed_duplicate_or_invalid_shard_output(self) -> None:
        valid_sample = {
            "entities": [],
            "text": "Một văn bản duy nhất.",
        }
        cases = (
            [
                {
                    "status": "FAILED",
                    "accepted_samples": 0,
                    "samples": [],
                }
            ],
            [
                {
                    "status": "COMPLETED",
                    "accepted_samples": 1,
                    "samples": [valid_sample],
                },
                {
                    "status": "COMPLETED",
                    "accepted_samples": 1,
                    "samples": [valid_sample],
                },
            ],
            [
                {
                    "status": "COMPLETED",
                    "accepted_samples": 1,
                    "samples": [
                        {
                            "entities": [
                                {
                                    "label": "PERSON",
                                    "start": 0,
                                    "end": 3,
                                    "text": "Sai",
                                }
                            ],
                            "text": "Đúng offset",
                        }
                    ],
                }
            ],
            [
                {
                    "status": "COMPLETED",
                    "accepted_samples": 1,
                    "samples": [{
                        "entities": [{
                            "label": "PERSON",
                            "start": 4,
                            "end": 14,
                            "text": "Lê Minh An",
                        }],
                        "text": "Chị Lê Minh An xác nhận.",
                    }],
                },
                {
                    "status": "COMPLETED",
                    "accepted_samples": 1,
                    "samples": [{
                        "entities": [{
                            "label": "PERSON",
                            "start": 4,
                            "end": 14,
                            "text": "Lê Minh An",
                        }],
                        "text": "Anh Lê Minh An phản hồi.",
                    }],
                },
            ],
            [
                {
                    "status": "COMPLETED",
                    "accepted_samples": 1,
                    "samples": [{
                        "entities": [{
                            "label": "PERSON",
                            "start": 4,
                            "end": 6,
                            "text": "An",
                        }],
                        "text": "Chị An xác nhận.",
                    }],
                },
                {
                    "status": "COMPLETED",
                    "accepted_samples": 1,
                    "samples": [{
                        "entities": [{
                            "label": "PERSON",
                            "start": 4,
                            "end": 8,
                            "text": "Bình",
                        }],
                        "text": "Chị Bình xác nhận.",
                    }],
                },
            ],
        )

        for payloads in cases:
            with self.subTest(payloads=payloads):
                with self.assertRaises(ParallelGenerationError):
                    merge_shard_payloads(
                        payloads,
                        expected_samples=sum(
                            item.get("accepted_samples", 0)
                            for item in payloads
                        ),
                    )

    def test_parallel_runner_merges_offline_shards_end_to_end(self) -> None:
        config = RunConfig(
            run_name="offline-parallel",
            num_samples=4,
            batch_size=2,
            focus_labels=["PERSON"],
            difficulty_distribution={
                "easy": 1.0,
                "medium": 0.0,
                "hard": 0.0,
            },
            sample_type_distribution={
                "positive": 1.0,
                "pure_negative": 0.0,
                "hard_negative": 0.0,
            },
            sample_length_distribution={
                "short": 0.5,
                "medium": 0.5,
                "long": 0.0,
            },
            max_entities={"easy": 1, "medium": 1, "hard": 1},
            complexity_limits={
                "positive": 1,
                "pure_negative": 1,
                "hard_negative": 2,
            },
            parallel_generation={
                "workers": 2,
                "shard_size": 2,
                "max_shard_retries": 0,
            },
            random_seed=174,
        )

        with TemporaryDirectory() as directory:
            summary = run_parallel_generation(
                config=config,
                taxonomy_path=Path("pii_taxonomy_rules.json"),
                output_directory=Path(directory),
                offline=True,
            )

            output_path = Path(summary["output_path"])
            dataset = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "COMPLETED")
            self.assertEqual(summary["accepted_samples"], 4)
            self.assertEqual(summary["completed_shards"], 2)
            self.assertTrue(Path(summary["summary_path"]).is_file())
            self.assertTrue(
                Path(summary["shard_artifact_directory"]).is_dir()
            )
            self.assertEqual(len(summary["shard_logs"]), 2)
            self.assertTrue(all(
                Path(item["console_log_path"]).is_file()
                for item in summary["shard_logs"]
            ))
            self.assertEqual(len(dataset), 4)
            self.assertEqual(
                len({sample["text"] for sample in dataset}),
                4,
            )
            self.assertTrue(all(
                set(sample["token_usage"]) == {
                    "input_tokens",
                    "output_tokens",
                    "generator",
                    "verifier",
                }
                for sample in dataset
            ))

    def test_parallel_runner_writes_failed_summary_for_exhausted_shard(
        self,
    ) -> None:
        config = RunConfig(
            run_name="failed-parallel",
            num_samples=1,
            focus_labels=["PERSON"],
            parallel_generation={
                "workers": 1,
                "shard_size": 1,
                "max_shard_retries": 0,
            },
        )
        failed_usage = self._usage_payload(30, "0.03")
        with TemporaryDirectory() as directory, patch(
            "pii_factory.parallel._run_shard",
            side_effect=ParallelGenerationError(
                "simulated shard failure",
                token_usage=failed_usage,
            ),
        ):
            output_directory = Path(directory)
            with self.assertRaisesRegex(
                ParallelGenerationError,
                "1 shard\\(s\\) failed permanently",
            ):
                run_parallel_generation(
                    config=config,
                    taxonomy_path=Path("pii_taxonomy_rules.json"),
                    output_directory=output_directory,
                    offline=True,
                )

            summaries = list(
                output_directory.glob("*-failed-summary.json")
            )
            self.assertEqual(len(summaries), 1)
            payload = json.loads(
                summaries[0].read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "FAILED")
            self.assertEqual(payload["completed_shards"], 0)
            self.assertIn("1", payload["failed_shards"])
            self.assertEqual(payload["token_usage"]["total_tokens"], 30)
            self.assertEqual(payload["token_usage"]["money_cost"], "0.03")


if __name__ == "__main__":
    unittest.main()
