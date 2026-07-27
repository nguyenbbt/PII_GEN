from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import uvicorn

from .api import create_app
from .bootstrap import build_pipeline
from .domain.models import CreateRunRequest, RunConfig


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
    args = parser.parse_args()
    if args.config or args.taxonomy_json:
        taxonomy_path = args.taxonomy_json or DEFAULT_TAXONOMY_PATH
        pipeline, repository, _ = build_pipeline(
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
        run = pipeline.create_run(CreateRunRequest(
            run_name=config.run_name, taxonomy_version_id=taxonomy.version_id, config=config,
        ))
        results = []
        while len(results) < config.num_samples:
            batch = pipeline.generate_pending(
                run.run_id,
                limit=min(config.batch_size, config.num_samples - len(results)),
            )
            if not batch:
                break
            results.extend(batch)
        final_run = repository.get_run(run.run_id)
        pipeline_usages = [
            result.pipeline_token_usage.total
            for result in results
            if result.pipeline_token_usage is not None
        ]
        total_input_tokens = sum(usage.input_tokens for usage in pipeline_usages)
        total_output_tokens = sum(usage.output_tokens for usage in pipeline_usages)
        total_tokens = sum(usage.total_tokens for usage in pipeline_usages)
        total_money_cost = sum(
            (usage.money_cost for usage in pipeline_usages),
            start=0,
        )
        print(json.dumps({
            "taxonomy_version_id": taxonomy.version_id,
            "labels": len(taxonomy.labels),
            "run_id": run.run_id,
            "status": final_run.status,
            "output_path": final_run.output_path,
            "accepted_samples": len(results),
            "token_usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "money_cost": str(total_money_cost),
            },
            "samples": [
                result.formatted_sample.dict()
                for result in results
                if result.formatted_sample is not None
            ],
        }, ensure_ascii=False, indent=2, default=str))
        if final_run.status == "FAILED":
            raise SystemExit(1)
        return
    uvicorn.run(create_app(offline=args.offline), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
