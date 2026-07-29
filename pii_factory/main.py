from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
from datetime import datetime
from pathlib import Path
import sys

import uvicorn

from .api import create_app
from .bootstrap import build_pipeline
from .domain.models import CreateRunRequest, RunConfig, TokenUsage


DEFAULT_TAXONOMY_PATH = Path(
    os.getenv("PII_TAXONOMY_PATH", "pii_taxonomy_rules.json")
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="PII Data Factory quality-first generation pipeline"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic clients without making paid network calls",
    )
    parser.add_argument(
        "--taxonomy-json",
        type=Path,
        help=(
            "Canonical taxonomy rules JSON "
            "(default: pii_taxonomy_rules.json)"
        ),
    )
    parser.add_argument("--config", type=Path, help="Run config JSON using the documented distribution schema")
    parser.add_argument("--samples", default=2, type=int, help="Samples to generate with --taxonomy-json")
    parser.add_argument("--focus-labels", default="TIME", help="Comma-separated labels for --taxonomy-json")
    parser.add_argument("--hard-negative", action="store_true", help="Use hard-negative generation for --taxonomy-json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for partial/final accepted JSON datasets (default: GEN_DATA_DIR or gen_data)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help=(
            "Detailed UTF-8 diagnostic log path. By default a timestamped log "
            "is created inside --output-dir."
        ),
    )
    args = parser.parse_args()
    if args.config or args.taxonomy_json:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stderr,
            force=True,
        )
        if args.offline:
            print(
                "WARNING: offline output is a smoke-test artifact, not a dataset.",
                file=sys.stderr,
            )
        taxonomy_path = args.taxonomy_json or DEFAULT_TAXONOMY_PATH
        pipeline, repository, event_bus = build_pipeline(
            offline=args.offline,
            output_directory=args.output_dir,
        )
        taxonomy = pipeline.taxonomy_service.import_json(taxonomy_path)
        if args.config:
            config_text = args.config.read_text(encoding="utf-8-sig").replace("\u2028", "\n").replace("\u2029", "\n")
            config = RunConfig.parse_obj(json.loads(config_text))
        else:
            focus_labels = [label.strip().upper() for label in args.focus_labels.split(",") if label.strip()]
            sample_type = "hard_negative" if args.hard_negative else "positive"
            config = RunConfig(
                run_name=f"json-{taxonomy_path.stem}",
                num_samples=args.samples,
                focus_labels=focus_labels,
                sample_type_distribution={
                    "positive": float(sample_type == "positive"),
                    "pure_negative": 0.0,
                    "hard_negative": float(sample_type == "hard_negative"),
                },
            )
        log_directory = args.output_dir or Path(
            os.getenv("GEN_DATA_DIR", "gen_data")
        )
        log_path = args.log_file or (
            log_directory
            / f"{config.run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path,
            mode="w",
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logging.getLogger().addHandler(file_handler)
        run = pipeline.create_run(CreateRunRequest(
            run_name=config.run_name, taxonomy_version_id=taxonomy.version_id, config=config,
        ))
        logging.getLogger(__name__).info(
            "[run] started name=%s samples=%s language=%s verifier=%s",
            config.run_name,
            config.num_samples,
            config.language,
            "on" if config.verifier.enabled else "off",
        )
        logging.getLogger(__name__).info(
            "[run] detailed diagnostic log=%s",
            log_path.resolve(),
        )
        results = []
        while len(results) < config.num_samples:
            logging.getLogger(__name__).info(
                "[run] scheduling next batch accepted=%s/%s",
                len(results),
                config.num_samples,
            )
            batch = pipeline.generate_pending(
                run.run_id,
                limit=min(config.batch_size, config.num_samples - len(results)),
            )
            if not batch:
                break
            results.extend(batch)
            logging.getLogger(__name__).info(
                "[run] batch finished accepted=%s/%s",
                len(results),
                config.num_samples,
            )
        final_run = repository.get_run(run.run_id)
        pipeline_usages = [
            result.pipeline_token_usage
            for result in results
            if result.pipeline_token_usage is not None
        ]
        generator_usage = TokenUsage.combine([
            usage.generator for usage in pipeline_usages
        ])
        verifier_usage = TokenUsage.combine([
            usage.verifier_total() for usage in pipeline_usages
        ])
        combined_usage = TokenUsage.combine(
            (generator_usage, verifier_usage)
        )
        run_events = [
            event
            for event in event_bus.list_events()
            if event.correlation_id == run.run_id
        ]
        event_counts = Counter(event.event_type for event in run_events)
        issue_counts: Counter[str] = Counter()
        for event in run_events:
            if event.event_type == "data.generation.rejected":
                issues = (
                    event.payload.get("output_validation", {}).get("issues", [])
                )
            elif event.event_type == "data.verification.rejected":
                issues = event.payload.get("issues", [])
            else:
                issues = []
            issue_counts.update(
                str(issue.get("type", "unknown"))
                for issue in issues
                if isinstance(issue, dict)
            )
        verification_outcomes = Counter(
            result.verification_trace.outcome
            for result in results
            if result.verification_trace is not None
        )
        generated_candidates = event_counts["data.generated"]
        accepted_with_verifier = sum(verification_outcomes.values())
        verifier_rejections = event_counts["data.verification.rejected"]
        verifier_decisions = accepted_with_verifier + verifier_rejections
        summary_payload = {
            "taxonomy_version_id": taxonomy.version_id,
            "labels": len(taxonomy.labels),
            "run_id": run.run_id,
            "status": final_run.status,
            "output_path": final_run.output_path,
            "diagnostic_log_path": str(log_path.resolve()),
            "accepted_samples": len(results),
            "token_usage": {
                "input_tokens": combined_usage.input_tokens,
                "output_tokens": combined_usage.output_tokens,
                "total_tokens": combined_usage.total_tokens,
                "money_cost": str(combined_usage.money_cost),
                "generator": {
                    "input_tokens": generator_usage.input_tokens,
                    "output_tokens": generator_usage.output_tokens,
                    "total_tokens": generator_usage.total_tokens,
                    "money_cost": str(generator_usage.money_cost),
                },
                "verifier": {
                    "input_tokens": verifier_usage.input_tokens,
                    "output_tokens": verifier_usage.output_tokens,
                    "total_tokens": verifier_usage.total_tokens,
                    "money_cost": str(verifier_usage.money_cost),
                },
            },
            "diagnostics": {
                "generated_candidates": generated_candidates,
                "discarded_candidates": max(
                    0,
                    generated_candidates - len(results),
                ),
                "deterministic_rejections": event_counts[
                    "data.generation.rejected"
                ],
                "verifier_rejections": verifier_rejections,
                "task_replacements": event_counts[
                    "generation.task.replaced"
                ],
                "fallback_accepts": event_counts[
                    "sample.fallback_accepted"
                ],
                "verification_outcomes": dict(verification_outcomes),
                "verifier_candidate_pass_rate": (
                    accepted_with_verifier / verifier_decisions
                    if verifier_decisions else None
                ),
                "rejection_issue_types": dict(issue_counts),
            },
        }
        if final_run.output_path:
            dataset_path = Path(final_run.output_path)
            summary_path = dataset_path.with_name(
                f"{dataset_path.stem}-summary.json"
            )
        else:
            summary_path = log_path.with_suffix(".summary.json")
        summary_payload["summary_path"] = str(summary_path.resolve())
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_temporary = summary_path.with_suffix(
            f"{summary_path.suffix}.tmp"
        )
        summary_temporary.write_text(
            json.dumps(
                summary_payload,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        summary_temporary.replace(summary_path)
        logging.getLogger(__name__).info(
            "[run] token summary input_tokens=%s output_tokens=%s "
            "total_tokens=%s generator_input=%s generator_output=%s "
            "verifier_input=%s verifier_output=%s money_cost=%s summary=%s",
            combined_usage.input_tokens,
            combined_usage.output_tokens,
            combined_usage.total_tokens,
            generator_usage.input_tokens,
            generator_usage.output_tokens,
            verifier_usage.input_tokens,
            verifier_usage.output_tokens,
            combined_usage.money_cost,
            summary_path.resolve(),
        )
        print(json.dumps({
            **summary_payload,
            "samples": [
                result.formatted_sample.dict(exclude_none=True)
                for result in results
                if result.formatted_sample is not None
            ],
        }, ensure_ascii=False, indent=2, default=str))
        logging.getLogger(__name__).info(
            "[run] finished status=%s accepted=%s/%s output=%s",
            final_run.status,
            len(results),
            config.num_samples,
            final_run.output_path,
        )
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        if final_run.status == "FAILED":
            raise SystemExit(1)
        return
    uvicorn.run(create_app(offline=args.offline), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
