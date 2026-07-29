from __future__ import annotations

import argparse
import json
import logging
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


logger = logging.getLogger(__name__)
_MODEL_IDENTITY_LOG = re.compile(
    r"\[llm\s+(?P<role>generator|verifier)\]\s+response received.*?"
    r"requested_model=(?P<requested>\S+)\s+"
    r"response_model=(?P<response>\S+)\s+"
    r"model_status=(?P<status>\S+)"
)


class ParallelGenerationError(RuntimeError):
    """A shard set cannot be safely published as one complete dataset."""


def _model_identity_summary(console_output: str) -> dict[str, list[str]]:
    """Collect unique provider-reported model mappings from a child shard."""
    identities: dict[str, list[str]] = {}
    for match in _MODEL_IDENTITY_LOG.finditer(console_output):
        role = match.group("role")
        identity = (
            f"{match.group('requested')}->{match.group('response')}"
            f"({match.group('status')})"
        )
        values = identities.setdefault(role, [])
        if identity not in values:
            values.append(identity)
    return identities


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
        # Child output filenames also contain a UUID and ``-summary.json.tmp``.
        # Keep their run names deliberately short so retries remain below the
        # legacy Windows MAX_PATH limit in deep OneDrive workspaces.
        base_name = re.sub(
            r"[^A-Za-z0-9._-]+",
            "-",
            config.run_name,
        ).strip("._-")[:20].rstrip("-") or "pii"
        shards.append(config.copy(update={
            "run_name": f"{base_name}-s{shard_index:03d}",
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
    role_usage: dict[str, dict[str, int | Decimal]] = {
        role: {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "money_cost": Decimal("0"),
        }
        for role in ("generator", "verifier")
    }
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
        for role in ("generator", "verifier"):
            shard_role_usage = usage.get(role) or {}
            role_usage[role]["input_tokens"] += int(
                shard_role_usage.get("input_tokens", 0)
            )
            role_usage[role]["output_tokens"] += int(
                shard_role_usage.get("output_tokens", 0)
            )
            role_usage[role]["total_tokens"] += int(
                shard_role_usage.get("total_tokens", 0)
            )
            role_usage[role]["money_cost"] += Decimal(
                str(shard_role_usage.get("money_cost", "0"))
            )

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
            **{
                role: {
                    "input_tokens": values["input_tokens"],
                    "output_tokens": values["output_tokens"],
                    "total_tokens": values["total_tokens"],
                    "money_cost": str(values["money_cost"]),
                }
                for role, values in role_usage.items()
            },
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
        "run_name": f"{shard.run_name}-r{attempt}",
        "random_seed": retry_seed,
    })


def _run_shard(
    *,
    shard: RunConfig,
    shard_index: int,
    taxonomy_path: Path,
    work_directory: Path,
    shard_artifact_directory: Path,
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
        # Keep child paths short for Windows installations under deep OneDrive
        # workspaces, where the legacy MAX_PATH limit may still apply.
        attempt_directory = shard_artifact_directory / (
            f"s{shard_index:03d}-a{attempt}"
        )
        attempt_directory.mkdir(parents=True, exist_ok=True)
        shard_output = attempt_directory / "output"
        console_log_path = attempt_directory / "console.log"
        logger.info(
            "[parallel shard %s] attempt %s/%s started samples=%s",
            shard_index,
            attempt + 1,
            shard.parallel_generation.max_shard_retries + 1,
            attempt_config.num_samples,
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
        console_log_path.write_text(
            completed.stderr,
            encoding="utf-8",
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
            model_identities = _model_identity_summary(completed.stderr)
            logger.info(
                "[parallel shard %s] model identity generator=%s verifier=%s",
                shard_index,
                ",".join(model_identities.get("generator", []))
                or "<not-observed>",
                ",".join(model_identities.get("verifier", []))
                or "<not-observed>",
            )
            logger.info(
                "[parallel shard %s] completed attempt=%s accepted=%s "
                "input_tokens=%s output_tokens=%s generator_input=%s "
                "generator_output=%s verifier_input=%s verifier_output=%s",
                shard_index,
                attempt + 1,
                payload.get("accepted_samples"),
                (payload.get("token_usage") or {}).get("input_tokens", 0),
                (payload.get("token_usage") or {}).get("output_tokens", 0),
                (
                    (payload.get("token_usage") or {}).get("generator") or {}
                ).get("input_tokens", 0),
                (
                    (payload.get("token_usage") or {}).get("generator") or {}
                ).get("output_tokens", 0),
                (
                    (payload.get("token_usage") or {}).get("verifier") or {}
                ).get("input_tokens", 0),
                (
                    (payload.get("token_usage") or {}).get("verifier") or {}
                ).get("output_tokens", 0),
            )
            return {
                **payload,
                "_shard_index": shard_index,
                "_console_log_path": str(console_log_path.resolve()),
                "_diagnostic_log_path": payload.get("diagnostic_log_path"),
            }
        last_message = (
            f"status={payload.get('status')}, "
            f"accepted={payload.get('accepted_samples')}"
        )
        logger.warning(
            "[parallel shard %s] attempt %s failed: %s",
            shard_index,
            attempt + 1,
            last_message,
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
    safe_name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "-",
        config.run_name,
    ).strip("._-") or "pii-run"
    parallel_run_id = str(uuid4())
    shard_artifact_directory = output_directory / (
        f"shards-{parallel_run_id[:8]}"
    )
    shard_artifact_directory.mkdir(parents=True, exist_ok=True)
    shards = build_shard_configs(config)
    workers = min(
        config.parallel_generation.workers,
        len(shards),
    )
    payloads_by_index: dict[int, dict[str, Any]] = {}
    failures_by_index: dict[int, str] = {}
    logger.info(
        "[parallel] started name=%s samples=%s workers=%s shards=%s "
        "shard_size=%s artifacts=%s",
        config.run_name,
        config.num_samples,
        workers,
        len(shards),
        config.parallel_generation.shard_size,
        shard_artifact_directory.resolve(),
    )
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
                    shard_artifact_directory=shard_artifact_directory,
                    offline=offline,
                ): index
                for index, shard in enumerate(shards, start=1)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    payloads_by_index[index] = future.result()
                except Exception as exc:
                    failures_by_index[index] = str(exc)
                    logger.error(
                        "[parallel shard %s] permanently failed: %s",
                        index,
                        exc,
                    )
                    continue
                logger.info(
                    "[parallel] progress completed_shards=%s/%s "
                    "accepted_samples=%s/%s",
                    len(payloads_by_index),
                    len(shards),
                    sum(
                        int(payload.get("accepted_samples", 0))
                        for payload in payloads_by_index.values()
                    ),
                    config.num_samples,
                )

    if failures_by_index:
        failed_summary_path = output_directory / (
            f"{safe_name}-parallel-{parallel_run_id}-failed-summary.json"
        )
        failed_summary = {
            "status": "FAILED",
            "output_path": None,
            "summary_path": str(failed_summary_path.resolve()),
            "shard_artifact_directory": str(
                shard_artifact_directory.resolve()
            ),
            "accepted_samples": sum(
                int(payload.get("accepted_samples", 0))
                for payload in payloads_by_index.values()
            ),
            "completed_shards": len(payloads_by_index),
            "expected_shards": len(shards),
            "failed_shards": {
                str(index): message
                for index, message in sorted(failures_by_index.items())
            },
        }
        temporary_failed_summary = failed_summary_path.with_suffix(
            f"{failed_summary_path.suffix}.tmp"
        )
        temporary_failed_summary.write_text(
            json.dumps(failed_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_failed_summary.replace(failed_summary_path)
        raise ParallelGenerationError(
            f"{len(failures_by_index)} shard(s) failed permanently; "
            f"completed={len(payloads_by_index)}/{len(shards)}; "
            f"summary={failed_summary_path.resolve()}; "
            f"artifacts={shard_artifact_directory.resolve()}"
        )

    payloads = [
        payloads_by_index[index]
        for index in range(1, len(shards) + 1)
    ]
    merged = merge_shard_payloads(
        payloads,
        expected_samples=config.num_samples,
    )
    output_path = output_directory / (
        f"{safe_name}-parallel-{parallel_run_id}.json"
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
    summary_path = output_path.with_name(
        f"{output_path.stem}-summary.json"
    )
    summary = {
        "status": "COMPLETED",
        "output_path": str(output_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "shard_artifact_directory": str(
            shard_artifact_directory.resolve()
        ),
        "shard_logs": [
            {
                "shard": payload.get("_shard_index"),
                "console_log_path": payload.get("_console_log_path"),
                "diagnostic_log_path": payload.get(
                    "_diagnostic_log_path"
                ),
            }
            for payload in payloads
        ],
        "accepted_samples": len(merged["samples"]),
        "completed_shards": len(shards),
        "workers": workers,
        "token_usage": merged["token_usage"],
        "diagnostics": merged["diagnostics"],
    }
    summary_temporary = summary_path.with_suffix(
        f"{summary_path.suffix}.tmp"
    )
    summary_temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_temporary.replace(summary_path)
    logger.info(
        "[parallel] token summary input_tokens=%s output_tokens=%s "
        "total_tokens=%s generator_input=%s generator_output=%s "
        "verifier_input=%s verifier_output=%s summary=%s",
        merged["token_usage"]["input_tokens"],
        merged["token_usage"]["output_tokens"],
        merged["token_usage"]["total_tokens"],
        merged["token_usage"]["generator"]["input_tokens"],
        merged["token_usage"]["generator"]["output_tokens"],
        merged["token_usage"]["verifier"]["input_tokens"],
        merged["token_usage"]["verifier"]["output_tokens"],
        summary_path.resolve(),
    )
    logger.info(
        "[parallel] finished accepted=%s/%s output=%s",
        len(merged["samples"]),
        config.num_samples,
        output_path.resolve(),
    )
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
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
    try:
        summary = run_parallel_generation(
            config=config,
            taxonomy_path=args.taxonomy_json,
            output_directory=args.output_dir,
            offline=args.offline,
        )
    except KeyboardInterrupt:
        logger.error(
            "[parallel] interrupted by user; completed shard artifacts remain "
            "inside %s",
            args.output_dir.resolve(),
        )
        raise SystemExit(130)
    except ParallelGenerationError as exc:
        logger.error("[parallel] failed: %s", exc)
        raise SystemExit(1)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
