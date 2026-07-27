import unittest

from pii_factory.domain.models import RunConfig
from pii_factory.parallel import (
    ParallelGenerationError,
    build_shard_configs,
    merge_shard_payloads,
)


class ParallelGenerationTests(unittest.TestCase):
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
                },
                "diagnostics": {"generated_candidates": 1},
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
                },
                "diagnostics": {"generated_candidates": 2},
            },
        ]

        merged = merge_shard_payloads(payloads, expected_samples=2)

        self.assertEqual(len(merged["samples"]), 2)
        self.assertEqual(merged["token_usage"]["input_tokens"], 30)
        self.assertEqual(merged["token_usage"]["money_cost"], "0.03")
        self.assertEqual(
            merged["diagnostics"]["generated_candidates"],
            3,
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


if __name__ == "__main__":
    unittest.main()
