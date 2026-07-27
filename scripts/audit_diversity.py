from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

from pii_factory.application.diversity_metrics import DiversityAuditSample, audit_diversity
from pii_factory.bootstrap import build_pipeline
from pii_factory.domain.models import CreateRunRequest, RunConfig


def run_offline_audit(config_path: Path, taxonomy_path: Path, sample_count: int) -> dict:
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_config["num_samples"] = sample_count
    raw_config["batch_size"] = min(sample_count, 1_000)
    raw_config.setdefault("validation", {})["novelty_mode"] = "audit"
    config = RunConfig.parse_obj(raw_config)
    pipeline, _, events = build_pipeline(offline=True)
    taxonomy = pipeline.taxonomy_service.import_json(taxonomy_path)
    run = pipeline.create_run(CreateRunRequest(taxonomy_version_id=taxonomy.version_id, config=config))

    results = []
    while len(results) < sample_count:
        batch = pipeline.generate_pending(run.run_id, config.batch_size)
        if not batch:
            break
        results.extend(batch)

    seed_packs = {
        event.payload["task_id"]: event.payload["seed_pack"]
        for event in events.list_events()
        if event.event_type == "seed.validated"
    }
    samples = []
    for result in results:
        seed_pack = seed_packs[result.task_id]
        samples.append(DiversityAuditSample(
            tagged_text=result.tagged_text,
            entities=[entity.dict() for entity in result.entities],
            decoy_values=[decoy["value"] for decoy in seed_pack["decoys"]],
            context_frame_id=result.context_frame_id,
            strategy_ids=[decoy["strategy_id"] for decoy in seed_pack["decoys"]],
            constraints=result.generation_query.constraints,
            entity_format_variants=result.diversity_profile.entity_format_variants,
        ))
    report = audit_diversity(samples)
    total_cost = sum((result.token_usage.money_cost for result in results), Decimal("0"))
    return {
        **report.dict(),
        "requested_samples": sample_count,
        "generated_samples": len(results),
        "total_money_cost": str(total_cost),
        "prompt_version": results[0].prompt_version if results else None,
        "random_seed": config.random_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PII generation diversity without calling Azure OpenAI.")
    parser.add_argument("--config", type=Path, default=Path("configs/run_config.example.json"))
    parser.add_argument(
        "--taxonomy-json",
        type=Path,
        default=Path("pii_taxonomy_rules.json"),
    )
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be greater than zero")
    print(json.dumps(
        run_offline_audit(args.config, args.taxonomy_json, args.samples),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
