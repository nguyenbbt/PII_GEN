from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

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
            samples.append(sample.dict(exclude_none=True))

        usage = payload.get("token_usage") or {}
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))
        total_tokens += int(usage.get("total_tokens", 0))
        money_cost += Decimal(str(usage.get("money_cost", "0")))

        for name, value in (payload.get("diagnostics") or {}).items():
            if name == "verifier_candidate_pass_rate":
                continue
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

    verification_outcomes = nested_diagnostics.get(
        "verification_outcomes",
        Counter(),
    )
    verifier_accepts = sum(verification_outcomes.values())
    verifier_rejections = diagnostics.get("verifier_rejections", 0)
    verifier_decisions = verifier_accepts + verifier_rejections
    merged_diagnostics: dict[str, Any] = {
        **dict(diagnostics),
        **{
            name: dict(counts)
            for name, counts in nested_diagnostics.items()
        },
    }
    merged_diagnostics["verifier_candidate_pass_rate"] = (
        verifier_accepts / verifier_decisions
        if verifier_decisions
        else None
    )

    return {
        "samples": samples,
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "money_cost": str(money_cost),
        },
        "diagnostics": merged_diagnostics,
    }


def _retry_config(
    shard: RunConfig,
    *,
    shard_index: int,
    attempt: int,
) -> RunConfig:
    if attempt == 0:
        return shard
    retry_seed = random.Random(
        shard.random_seed ^ (attempt * 0xA11C_E5ED)
    ).randint(1, 2_147_483_647)
    return shard.copy(update={
        "run_name": f"{shard.run_name[:108]}-retry-{attempt}",
        "random_seed": retry_seed,
    })


def _run_shard(
    *,
    shard: RunConfig,
    shard_index: int,
    taxonomy_path: Path,
    work_directory: Path,
    offline: bool,
) -> dict[str, Any]:
    last_message = "no provider response"
    for attempt in range(
        shard.parallel_generation.max_shard_retries + 1
    ):
        attempt_config = _retry_config(
            shard,
            shard_index=shard_index,
            attempt=attempt,
        )
        config_path = work_directory / (
            f"shard-{shard_index:03d}-attempt-{attempt}.json"
        )
        config_path.write_text(
            json.dumps(
                attempt_config.dict(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        shard_output = work_directory / (
            f"output-{shard_index:03d}-attempt-{attempt}"
        )
        command = [
            sys.executable,
            "-m",
            "pii_factory.main",
            "--config",
            str(config_path),
            "--taxonomy-json",
            str(taxonomy_path),
            "--output-dir",
            str(shard_output),
        ]
        if offline:
            command.append("--offline")
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        try:
            payload = json.loads(completed.stdout)
        except (TypeError, ValueError, json.JSONDecodeError):
            last_message = (
                completed.stderr.strip()[-500:]
                or f"child exited with code {completed.returncode}"
            )
            continue
        if (
            completed.returncode == 0
            and payload.get("status") == "COMPLETED"
            and payload.get("accepted_samples")
            == attempt_config.num_samples
        ):
            return payload
        last_message = (
            f"status={payload.get('status')}, "
            f"accepted={payload.get('accepted_samples')}"
        )
    raise ParallelGenerationError(
        f"shard {shard_index} exhausted retries: {last_message}"
    )


def run_parallel_generation(
    *,
    config: RunConfig,
    taxonomy_path: Path,
    output_directory: Path,
    offline: bool = False,
) -> dict[str, Any]:
    taxonomy_path = taxonomy_path.resolve()
    if not taxonomy_path.is_file():
        raise ParallelGenerationError(
            f"taxonomy file does not exist: {taxonomy_path}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    shards = build_shard_configs(config)
    workers = min(
        config.parallel_generation.workers,
        len(shards),
    )
    payloads_by_index: dict[int, dict[str, Any]] = {}
    with TemporaryDirectory(
        prefix=".parallel-",
        dir=output_directory,
    ) as temporary:
        work_directory = Path(temporary)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_shard,
                    shard=shard,
                    shard_index=index,
                    taxonomy_path=taxonomy_path,
                    work_directory=work_directory,
                    offline=offline,
                ): index
                for index, shard in enumerate(shards, start=1)
            }
            for future in as_completed(futures):
                index = futures[future]
                payloads_by_index[index] = future.result()

    payloads = [
        payloads_by_index[index]
        for index in range(1, len(shards) + 1)
    ]
    merged = merge_shard_payloads(
        payloads,
        expected_samples=config.num_samples,
    )
    safe_name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "-",
        config.run_name,
    ).strip("._-") or "pii-run"
    output_path = output_directory / (
        f"{safe_name}-parallel-{uuid4()}.json"
    )
    temporary_path = output_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(
            merged["samples"],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return {
        "status": "COMPLETED",
        "output_path": str(output_path.resolve()),
        "accepted_samples": len(merged["samples"]),
        "completed_shards": len(shards),
        "workers": workers,
        "token_usage": merged["token_usage"],
        "diagnostics": merged["diagnostics"],
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Run PII generation shards concurrently and merge safely"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--taxonomy-json",
        type=Path,
        default=Path("pii_taxonomy_rules.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gen_data"),
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = RunConfig.parse_obj(json.loads(
        args.config.read_text(encoding="utf-8-sig")
    ))
    summary = run_parallel_generation(
        config=config,
        taxonomy_path=args.taxonomy_json,
        output_directory=args.output_dir,
        offline=args.offline,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
