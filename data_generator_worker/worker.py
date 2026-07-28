from __future__ import annotations

import hashlib
import logging
import time
from typing import Protocol

from .config import Settings
from .contracts import DataGenerationRequest, DataGenerationResult, EventEnvelope, GeneratedEntity, GenerationQuery
from .cost import CostCalculator
from .llm_client import LLMClient
from .placeholders import replace_entity_placeholders
from .prompt import PROMPT_VERSION, build_messages
from .validation import validate_generated_output, validate_seeded_contract

logger = logging.getLogger("data_generator_worker")


class AttemptStore(Protocol):
    def get_by_idempotency_key(self, key: str) -> EventEnvelope | None: ...

    def save(self, event: EventEnvelope) -> None: ...


class InMemoryAttemptStore:
    """Local adapter for development/tests; replace with transactional PostgreSQL persistence in production."""

    def __init__(self) -> None:
        self._events: dict[str, EventEnvelope] = {}

    def get_by_idempotency_key(self, key: str) -> EventEnvelope | None:
        return self._events.get(key)

    def save(self, event: EventEnvelope) -> None:
        self._events[event.idempotency_key] = event


class DataGeneratorWorker:
    def __init__(self, llm: LLMClient, settings: Settings, store: AttemptStore) -> None:
        self.llm = llm
        self.settings = settings
        self.store = store
        self.cost_calculator = CostCalculator(
            settings.input_token_price_per_million_usd,
            settings.output_token_price_per_million_usd,
        )

    def process(self, request: DataGenerationRequest) -> EventEnvelope:
        idempotency_key = f"{request.task.task_id}:attempt:{request.attempt_no}:data-generation"
        existing = self.store.get_by_idempotency_key(idempotency_key)
        if existing:
            logger.info("duplicate delivery ignored", extra={"task_id": request.task.task_id, "attempt_no": request.attempt_no})
            return existing

        started = time.perf_counter()
        completion = self.llm.generate(build_messages(request))
        tagged_text, raw_entities = replace_entity_placeholders(
            tagged_text=completion.tagged_text,
            entities=completion.entities,
            positive_entities=[
                vars(seed) for seed in request.seed_pack.positive_entities
            ],
        )
        latency_ms = round((time.perf_counter() - started) * 1000)
        token_usage = self.cost_calculator.calculate(
            completion.input_tokens, completion.output_tokens, completion.total_tokens
        )
        validated_entities = validate_generated_output(
            tagged_text=tagged_text,
            entities=raw_entities,
            allowed_labels=request.task.focus_labels,
            required_labels=request.task.focus_labels,
            sample_type=(
                "pure_negative"
                if request.task.sample_type == "hard_negative"
                and request.seed_pack.hard_negative_mode == "decoy_only"
                else request.task.sample_type
            ),
            max_entities=request.task.max_entities,
        )
        validate_seeded_contract(
            tagged_text=tagged_text,
            entities=validated_entities,
            positive_entities=[vars(seed) for seed in request.seed_pack.positive_entities],
            decoys=[vars(decoy) for decoy in request.seed_pack.decoys],
            max_decoy_occurrences=(
                2 if request.seed_pack.hard_negative_mode == "decoy_only" else 1
            ),
        )
        query = GenerationQuery(
            language=request.task.language,
            constraints=request.task.optional_constraints,
            focus_labels=request.task.focus_labels,
            difficulty=request.task.difficulty,
            sample_type=request.task.sample_type,
            max_entities=request.task.max_entities,
            length_target=request.task.length_target,
        )
        result = DataGenerationResult(
            task_id=request.task.task_id,
            attempt_no=request.attempt_no,
            seed_pack_id=request.seed_pack.seed_pack_id,
            context_frame_id=request.seed_pack.context_frame.frame_id,
            generation_query=query,
            entities=[GeneratedEntity(**entity) for entity in validated_entities],
            tagged_text=tagged_text,
            token_usage=token_usage,
            latency_ms=latency_ms,
            model=self.settings.effective_generator_model,
            prompt_version=PROMPT_VERSION,
            output_hash=hashlib.sha256(tagged_text.encode("utf-8")).hexdigest(),
            taxonomy_context_used=dict(request.taxonomy_context),
            diversity_profile=dict(request.task.diversity_profile),
        )
        event = EventEnvelope(
            correlation_id=request.correlation_id or request.task.run_id,
            causation_id=request.causation_id or request.event_id,
            idempotency_key=idempotency_key,
            payload=result,
        )
        self.store.save(event)
        logger.info(
            "generation completed",
            extra={
                "task_id": request.task.task_id,
                "attempt_no": request.attempt_no,
                "input_tokens": token_usage.input_tokens,
                "output_tokens": token_usage.output_tokens,
                "total_tokens": token_usage.total_tokens,
                "money_cost": token_usage.money_cost,
            },
        )
        return event
