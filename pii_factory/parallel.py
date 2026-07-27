from __future__ import annotations

import random
from collections import Counter
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from .domain.models import FormattedSample, RunConfig


class ParallelGenerationError(RuntimeError):
    """A shard set cannot be safely published as one complete dataset."""


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


def merge_shard_payloads(
    payloads: list[dict[str, Any]],
    *,
    expected_samples: int,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    input_tokens = output_tokens = total_tokens = 0
    money_cost = Decimal("0")
    diagnostics: Counter[str] = Counter()
    nested_diagnostics: dict[str, Counter[str]] = {}

    for payload in payloads:
        shard_samples = payload.get("samples")
        if (
            payload.get("status") != "COMPLETED"
            or not isinstance(shard_samples, list)
            or payload.get("accepted_samples") != len(shard_samples)
        ):
            raise ParallelGenerationError(
                "every shard must be COMPLETED with all accepted samples"
            )
        for raw_sample in shard_samples:
            try:
                sample = FormattedSample.parse_obj(raw_sample)
            except (TypeError, ValueError, ValidationError) as exc:
                raise ParallelGenerationError(
                    "shard contains an invalid formatted sample"
                ) from exc
            if sample.text in seen_texts:
                raise ParallelGenerationError(
                    "parallel shards contain duplicate sample text"
                )
            seen_texts.add(sample.text)
            samples.append(sample.dict())

        usage = payload.get("token_usage") or {}
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))
        total_tokens += int(usage.get("total_tokens", 0))
        money_cost += Decimal(str(usage.get("money_cost", "0")))

        for name, value in (payload.get("diagnostics") or {}).items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                diagnostics[name] += value
            elif isinstance(value, dict):
                bucket = nested_diagnostics.setdefault(name, Counter())
                bucket.update({
                    str(key): int(count)
                    for key, count in value.items()
                    if isinstance(count, (int, float))
                })

    if len(samples) != expected_samples:
        raise ParallelGenerationError(
            f"merged sample count {len(samples)} != {expected_samples}"
        )

    return {
        "samples": samples,
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "money_cost": str(money_cost),
        },
        "diagnostics": {
            **dict(diagnostics),
            **{
                name: dict(counts)
                for name, counts in nested_diagnostics.items()
            },
        },
    }
