import unittest

from pii_factory.domain.models import RunConfig
from pii_factory.parallel import build_shard_configs


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


if __name__ == "__main__":
    unittest.main()
