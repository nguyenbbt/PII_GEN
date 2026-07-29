from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path

from .application.services import CostRates, CoverageController, DataGenerator, Pipeline, RunOrchestrator
from .application.formatting import JsonDatasetWriter, OutputFormatter
from .application.taxonomy_service import TaxonomyService
from .application.verification import VerifierService
from .infrastructure.clients import (
    AzureOpenAICompletionClient,
    AzureOpenAISettings,
    AzureOpenAIVerifierClient,
    OfflineCompletionClient,
    OfflineVerifierClient,
    load_local_environment,
)
from .infrastructure.memory import InMemoryEventBus, InMemoryRepository


_DEFAULT_MODEL_RATES = {
    "GENERATOR": CostRates(
        input_per_million_usd=Decimal("0.30"),
        output_per_million_usd=Decimal("2.50"),
    ),
    "VERIFIER": CostRates(
        input_per_million_usd=Decimal("1.25"),
        output_per_million_usd=Decimal("10.00"),
    ),
}


def _role_cost_rates(role: str, fallback: CostRates) -> CostRates:
    prefix = role.upper()
    return CostRates(
        input_per_million_usd=Decimal(os.getenv(
            f"{prefix}_INPUT_TOKEN_PRICE_PER_MILLION_USD",
            str(fallback.input_per_million_usd),
        )),
        output_per_million_usd=Decimal(os.getenv(
            f"{prefix}_OUTPUT_TOKEN_PRICE_PER_MILLION_USD",
            str(fallback.output_per_million_usd),
        )),
    )


def _cost_rates(role: str) -> CostRates:
    role_name = role.upper()
    defaults = _DEFAULT_MODEL_RATES.get(role_name, CostRates())
    fallback = CostRates(
        input_per_million_usd=Decimal(os.getenv(
            "INPUT_TOKEN_PRICE_PER_MILLION_USD",
            str(defaults.input_per_million_usd),
        )),
        output_per_million_usd=Decimal(os.getenv(
            "OUTPUT_TOKEN_PRICE_PER_MILLION_USD",
            str(defaults.output_per_million_usd),
        )),
    )
    return _role_cost_rates(role_name, fallback)


def build_pipeline(
    offline: bool = False,
    output_directory: Path | str | None = None,
) -> tuple[Pipeline, InMemoryRepository, InMemoryEventBus]:
    load_local_environment()
    repository = InMemoryRepository()
    event_bus = InMemoryEventBus()
    orchestrator = RunOrchestrator(repository, event_bus)
    taxonomy_service = TaxonomyService(repository, event_bus)
    coverage = CoverageController(
        repository,
        event_bus,
        taxonomy_for_run=orchestrator.taxonomy_for_run,
    )
    generator_rates = _cost_rates("GENERATOR")
    verifier_rates = _cost_rates("VERIFIER")
    if offline:
        client = OfflineCompletionClient()
        verifier_client = OfflineVerifierClient()
        model = "offline-demo"
    else:
        settings = AzureOpenAISettings.from_environment()
        client = AzureOpenAICompletionClient(settings)
        verifier_client = AzureOpenAIVerifierClient(
            settings,
            input_price_per_million=verifier_rates.input_per_million_usd,
            output_price_per_million=verifier_rates.output_per_million_usd,
        )
        model = settings.effective_generator_model
    generator = DataGenerator(
        repository,
        event_bus,
        client,
        model,
        generator_rates,
    )
    verifier = VerifierService(verifier_client)
    formatter = OutputFormatter()
    output_root = Path(
        output_directory or Path(os.getenv("GEN_DATA_DIR", "gen_data"))
    )
    if offline:
        output_root /= "offline-smoke"
    writer = JsonDatasetWriter(output_root)
    return Pipeline(
        orchestrator,
        coverage,
        generator,
        verifier,
        formatter,
        writer,
        taxonomy_service,
        repository,
        event_bus,
        enforce_quality_targets=not offline,
    ), repository, event_bus
