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
)
from .infrastructure.memory import InMemoryEventBus, InMemoryRepository


def build_pipeline(
    offline: bool = False,
    output_directory: Path | str | None = None,
) -> tuple[Pipeline, InMemoryRepository, InMemoryEventBus]:
    repository = InMemoryRepository()
    event_bus = InMemoryEventBus()
    orchestrator = RunOrchestrator(repository, event_bus)
    taxonomy_service = TaxonomyService(repository, event_bus)
    coverage = CoverageController(
        repository,
        event_bus,
        taxonomy_for_run=orchestrator.taxonomy_for_run,
    )
    rates = CostRates(
        input_per_million_usd=Decimal(os.getenv("INPUT_TOKEN_PRICE_PER_MILLION_USD", "2.50")),
        output_per_million_usd=Decimal(os.getenv("OUTPUT_TOKEN_PRICE_PER_MILLION_USD", "10.00")),
    )
    if offline:
        client = OfflineCompletionClient()
        verifier_client = OfflineVerifierClient()
        model = "offline-demo"
    else:
        settings = AzureOpenAISettings.from_environment()
        client = AzureOpenAICompletionClient(settings)
        verifier_client = AzureOpenAIVerifierClient(
            settings,
            input_price_per_million=rates.input_per_million_usd,
            output_price_per_million=rates.output_per_million_usd,
        )
        model = settings.effective_generator_model
    generator = DataGenerator(repository, event_bus, client, model, rates)
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
