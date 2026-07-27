from __future__ import annotations

import random

from .domain.models import RunConfig


def build_shard_configs(config: RunConfig) -> list[RunConfig]:
    """Split one logical run into deterministic, independently seeded shards."""
    shard_size = config.parallel_generation.shard_size
    shards: list[RunConfig] = []
    remaining = config.num_samples
    shard_index = 0
    while remaining:
        shard_index += 1
        sample_count = min(shard_size, remaining)
        seed = random.Random(
            config.random_seed ^ (shard_index * 0x9E37_79B1)
        ).randint(1, 2_147_483_647)
        base_name = config.run_name[:100].rstrip("-")
        shards.append(config.copy(update={
            "run_name": f"{base_name}-shard-{shard_index:02d}",
            "num_samples": sample_count,
            "batch_size": min(config.batch_size, sample_count),
            "random_seed": seed,
        }))
        remaining -= sample_count
    return shards
