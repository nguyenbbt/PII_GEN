from __future__ import annotations

import hashlib
import logging
import random
import time
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List
from uuid import uuid4

from data_generator_worker.placeholders import replace_entity_placeholders
from data_generator_worker.prompt import PROMPT_VERSION, build_prompt_messages

from ..domain.models import (
    CreateRunRequest,
    DataGenerationRequest,
    DataGenerationResult,
    DeterministicValidationResult,
    EventEnvelope,
    GenerationCandidate,
    GeneratedEntity,
    GenerationQuery,
    GenerationTask,
    NoveltyAssessment,
    PipelineTokenUsage,
    ReflectionContext,
    Run,
    RunStatus,
    SampleType,
    SeedPack,
    TaskStatus,
    TaxonomyLabel,
    TokenUsage,
    ValidationIssue,
    VerificationTrace,
)
from ..ports import CompletionClient, EventBus, RunRepository
from .formatting import JsonDatasetWriter, OutputFormatter
from .few_shot_similarity import FewShotImitationGuard
from .diversity import DiversityPlanner, resolve_length_target
from .context_catalog import compatible_context_frames
from .novelty import NoveltyGuard
from .retry import RegenerationRouter
from .seed_generation import (
    ContextFrameSelector,
    ContextSelectionError,
    UnsupportedDecoyError,
    build_sample_type_router,
)
from .taxonomy_service import TaxonomyService
from .validators import DeterministicOutputValidator, SeedPackValidator
from .value_bank import ValueBankError
from .verification import (
    DeterministicIssueRouter,
    VerificationRoutingError,
    VerifierInfrastructureError,
    VerifierService,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CostRates:
    input_per_million_usd: Decimal = Decimal("2.50")
    output_per_million_usd: Decimal = Decimal("10.00")

    def calculate(self, input_tokens: int, output_tokens: int, total_tokens: int) -> TokenUsage:
        cost = (
            Decimal(input_tokens) * self.input_per_million_usd / Decimal(1_000_000)
            + Decimal(output_tokens) * self.output_per_million_usd / Decimal(1_000_000)
        ).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            money_cost=cost,
        )


class OutputValidationError(ValueError):
    def __init__(self, result: DeterministicValidationResult) -> None:
        self.result = result
        scopes = {issue.scope for issue in result.issues}
        self.regeneration_scope = next(
            (scope for scope in ("SEEDS", "CONTEXT", "TEXT") if scope in scopes),
            "TEXT",
        )
        super().__init__("; ".join(issue.reason for issue in result.issues))


class SeedPlanningError(ValueError):
    def __init__(self, scope: str, issues: List[ValidationIssue]) -> None:
        self.regeneration_scope = scope
        self.issues = issues
        super().__init__("; ".join(issue.reason for issue in issues))


class RunOrchestrator:
    def __init__(self, repository: RunRepository, event_bus: EventBus) -> None:
        self.repository = repository
        self.event_bus = event_bus
        self._taxonomies: Dict[str, List[TaxonomyLabel]] = {}

    def create_run(
        self, request: CreateRunRequest, taxonomy: List[TaxonomyLabel], taxonomy_version_id: str
    ) -> Run:
        known_labels = {label.code for label in taxonomy}
        selected_labels = request.config.label_pool or [label.code for label in taxonomy]
        missing = set(selected_labels) - known_labels
        if missing:
            raise ValueError(f"focus labels absent from taxonomy: {sorted(missing)}")
        required_samples = (
            request.config.minimum_per_label
            if request.config.focus_label
            else request.config.minimum_per_label * len(selected_labels)
        )
        if required_samples > request.config.num_samples:
            raise ValueError("num_samples is too small for minimum_per_label across selected focus_labels")
        resolved_config = request.config.copy(update=(
            {} if request.config.focus_label else {"focus_labels": selected_labels}
        ))
        run = Run(
            name=request.run_name or resolved_config.run_name,
            config=resolved_config,
            taxonomy_version_id=taxonomy_version_id,
        )
        self.repository.add_run(run)
        self._taxonomies[run.run_id] = taxonomy
        self.event_bus.publish(EventEnvelope(
            event_type="run.created", correlation_id=run.run_id,
            idempotency_key=f"run:{run.run_id}:created", payload={"run_id": run.run_id},
        ))
        return run

    def taxonomy_for_run(self, run_id: str) -> List[TaxonomyLabel]:
        return self._taxonomies[run_id]


class CoverageController:
    def __init__(self, repository: RunRepository, event_bus: EventBus) -> None:
        self.repository = repository
        self.event_bus = event_bus

    def create_tasks(self, run: Run) -> List[GenerationTask]:
        rng = random.Random(run.config.random_seed)
        diversity_planner = DiversityPlanner(
            run.config.random_seed,
            run.config.sample_length_distribution,
            run.config.num_samples,
            run.config.sample_structures,
        )
        tasks: List[GenerationTask] = []
        labels = run.config.label_pool or []
        mandatory_labels = (
            [] if run.config.focus_label
            else [label for label in labels for _ in range(run.config.minimum_per_label)]
        )
        difficulties = ["easy", "medium", "hard"]
        sample_types = ["positive", "pure_negative", "hard_negative"]
        for sequence_no in range(1, run.config.num_samples + 1):
            difficulty = rng.choices(
                difficulties,
                weights=[run.config.difficulty_distribution[name] for name in difficulties], k=1,
            )[0]
            sample_type = rng.choices(
                sample_types,
                weights=[run.config.sample_type_distribution[name] for name in sample_types], k=1,
            )[0]
            entity_limit = run.config.max_entities[difficulty]
            planned_constraints = [
                name for name, probability in run.config.optional_constraint_distribution.items()
                if rng.random() < probability
            ]
            complexity_limit = getattr(run.config.complexity_limits, sample_type)
            reserved_decoys = run.config.hard_negative.min_decoys if sample_type == "hard_negative" else 0
            if sample_type == "pure_negative":
                max_focus = min(entity_limit, len(labels))
                optional_constraints = planned_constraints[:complexity_limit]
            elif sample_type == "hard_negative" and run.config.hard_negative.mode == "decoy_only":
                max_focus = 1
                optional_budget = max(0, complexity_limit - reserved_decoys)
                optional_constraints = planned_constraints[:optional_budget]
            else:
                max_focus = min(entity_limit, len(labels), max(1, complexity_limit - reserved_decoys))
                if sample_type == "hard_negative":
                    max_focus = min(max_focus, run.config.hard_negative.max_focus_labels)
                optional_budget = max(0, complexity_limit - max_focus - reserved_decoys)
                optional_constraints = planned_constraints[:optional_budget]
            focus_labels = self._select_labels(
                run, labels, mandatory_labels, sequence_no, max_focus, rng
            )
            selected_robin_labels = focus_labels[1:] if run.config.focus_label else []
            sample_structure = diversity_planner.select_sample_structure(
                run.config.sample_structure
            )
            diversity_profile = diversity_planner.plan(
                focus_labels,
                sample_structure,
            )
            task = GenerationTask(
                run_id=run.run_id, sequence_no=sequence_no, language=run.config.language,
                slot_no=sequence_no,
                focus_labels=focus_labels,
                annotation_labels=(
                    labels
                    if run.config.value_bank.allow_additional_unseeded_pii
                    else focus_labels
                ),
                difficulty=difficulty, sample_type=sample_type,
                sample_structure=sample_structure,
                focus_label=run.config.focus_label, robin_labels=selected_robin_labels,
                optional_constraints=optional_constraints, max_entities=entity_limit,
                max_attempts=run.config.max_attempts, random_seed=rng.randint(1, 2_147_483_647),
                diversity_profile=diversity_profile,
                length_target=resolve_length_target(
                    sample_structure,
                    diversity_profile.length_bucket,
                ),
            )
            self.repository.add_task(task)
            self.event_bus.publish(EventEnvelope(
                event_type="generation.task.created", correlation_id=run.run_id,
                idempotency_key=f"task:{task.task_id}:created", payload=task.dict(),
            ))
            tasks.append(task)
        return tasks

    @staticmethod
    def _select_labels(
        run: Run,
        labels: List[str],
        mandatory_labels: List[str],
        sequence_no: int,
        max_focus: int,
        rng: random.Random,
    ) -> List[str]:
        if run.config.focus_label:
            selection = run.config.robin_selection
            max_robin = min(
                selection.max_per_sample,
                len(run.config.robin_labels),
                max(0, max_focus - 1),
            )
            robin_count = rng.randint(selection.min_per_sample, max_robin)
            return [
                run.config.focus_label,
                *rng.sample(run.config.robin_labels, k=robin_count),
            ]

        primary_label = (
            mandatory_labels[sequence_no - 1]
            if sequence_no <= len(mandatory_labels) else rng.choice(labels)
        )
        remaining_labels = [label for label in labels if label != primary_label]
        focus_count = rng.randint(1, max_focus)
        return [primary_label, *rng.sample(remaining_labels, k=focus_count - 1)]


class DataGenerator:
    def __init__(
        self,
        repository: RunRepository,
        event_bus: EventBus,
        client: CompletionClient,
        model: str,
        cost_rates: CostRates,
    ) -> None:
        self.repository = repository
        self.event_bus = event_bus
        self.client = client
        self.model = model
        self.cost_rates = cost_rates

    def generate(
        self,
        request: DataGenerationRequest,
        _deprecated_validator: DeterministicOutputValidator | None = None,
    ) -> GenerationCandidate:
        """Generate a raw candidate.

        The optional validator argument is retained only for source compatibility
        with older callers; validation now belongs to the downstream quality gate.
        """
        existing = self.repository.get_candidate(request.task.task_id, request.attempt_no)
        if existing:
            return existing
        idempotency_key = f"{request.task.task_id}:attempt:{request.attempt_no}:data-generation"
        request.task.status = TaskStatus.GENERATING
        request.task.current_attempt = request.attempt_no
        self.repository.update_task(request.task)
        started = time.perf_counter()
        try:
            tagged_text, raw_entities, input_tokens, output_tokens, total_tokens = self.client.generate(
                self._messages(request)
            )
            logger.info(
                "[sample %s attempt %s] LLM raw tagged_text:\n%s",
                request.task.slot_no or request.task.sequence_no,
                request.attempt_no,
                tagged_text,
            )
            tagged_text, raw_entities = replace_entity_placeholders(
                tagged_text=tagged_text,
                entities=raw_entities,
                positive_entities=[
                    entity.dict() for entity in request.seed_pack.positive_entities
                ],
            )
            logger.info(
                "[sample %s attempt %s] candidate after Value Bank binding:\n%s",
                request.task.slot_no or request.task.sequence_no,
                request.attempt_no,
                tagged_text,
            )
        except ValueError as exc:
            logger.warning(
                "[sample %s attempt %s] output binding failed: %s",
                request.task.slot_no or request.task.sequence_no,
                request.attempt_no,
                exc,
            )
            raise OutputValidationError(DeterministicValidationResult(
                valid=False,
                issues=[ValidationIssue(type="invalid_output", scope="TEXT", reason=str(exc))],
            )) from exc
        latency_ms = round((time.perf_counter() - started) * 1000)
        entities = [GeneratedEntity(**entity) for entity in raw_entities]
        candidate = GenerationCandidate(
            task_id=request.task.task_id,
            attempt_no=request.attempt_no,
            seed_pack_id=request.seed_pack.seed_pack_id,
            context_frame_id=request.seed_pack.context_frame.frame_id,
            generation_query=GenerationQuery(
                language=request.task.language, focus_labels=request.task.focus_labels,
                focus_label=request.task.focus_label, robin_labels=request.task.robin_labels,
                constraints=request.task.optional_constraints, difficulty=request.task.difficulty,
                sample_type=request.task.sample_type, max_entities=request.task.max_entities,
                sample_structure=request.task.sample_structure,
                length_target=request.task.length_target,
            ),
            entities=entities,
            tagged_text=tagged_text,
            token_usage=self.cost_rates.calculate(input_tokens, output_tokens, total_tokens),
            latency_ms=latency_ms,
            model=self.model,
            prompt_version=PROMPT_VERSION,
            output_hash=hashlib.sha256(tagged_text.encode("utf-8")).hexdigest(),
            seed_validation=request.seed_validation,
            diversity_profile=request.task.diversity_profile,
            taxonomy_context_used=request.taxonomy_context,
        )
        self.repository.add_candidate(candidate)
        self.repository.mark_consumed(idempotency_key)
        request.task.status = TaskStatus.GENERATED
        self.repository.update_task(request.task)
        self.event_bus.publish(EventEnvelope(
            event_type="data.generated", correlation_id=request.correlation_id or request.task.run_id,
            causation_id=request.causation_id, idempotency_key=idempotency_key, payload=candidate.dict(),
        ))
        return candidate

    @staticmethod
    def _messages(request: DataGenerationRequest) -> List[dict[str, str]]:
        return build_prompt_messages(
            task=request.task.dict(),
            seed_pack=request.seed_pack.dict(),
            taxonomy_context=request.taxonomy_context.dict(),
            dynamic_prompt_additions=request.dynamic_prompt_additions,
            reflection=request.reflection.dict() if request.reflection else None,
        )


class Pipeline:
    """Synchronous quality-first coordinator through accepted JSON samples."""

    def __init__(
        self,
        orchestrator: RunOrchestrator,
        coverage: CoverageController,
        generator: DataGenerator,
        verifier: VerifierService,
        formatter: OutputFormatter,
        dataset_writer: JsonDatasetWriter,
        taxonomy_service: TaxonomyService,
        repository: RunRepository,
        event_bus: EventBus,
        enforce_quality_targets: bool = True,
    ) -> None:
        self.orchestrator = orchestrator
        self.coverage = coverage
        self.generator = generator
        self.verifier = verifier
        self.formatter = formatter
        self.dataset_writer = dataset_writer
        self.taxonomy_service = taxonomy_service
        self.repository = repository
        self.event_bus = event_bus
        self.enforce_quality_targets = enforce_quality_targets
        self._seed_states: Dict[str, tuple[SeedPack, DeterministicValidationResult]] = {}
        self._reflections: Dict[str, ReflectionContext] = {}
        self._usage_calls: Dict[tuple[str, int], Dict[str, List[TokenUsage]]] = {}
        self._recorded_calls: set[tuple[str, int, str]] = set()

    def create_run(self, request: CreateRunRequest) -> Run:
        taxonomy = (
            self.taxonomy_service.register(request.taxonomy)
            if request.taxonomy else self.taxonomy_service.get(request.taxonomy_version_id or "")
        )
        run = self.orchestrator.create_run(request, taxonomy.labels, taxonomy.version_id)
        self.coverage.create_tasks(run)
        return run

    def _build_valid_seed_pack(
        self, task: GenerationTask, taxonomy: List[TaxonomyLabel], run: Run
    ) -> tuple[SeedPack, DeterministicValidationResult]:
        rng = random.Random(task.random_seed)
        router = build_sample_type_router(
            run.config.value_bank,
            run.config.hard_negative,
        )
        validator = SeedPackValidator(run.config.hard_negative, run.config.validation)
        last_issues: List[ValidationIssue] = []
        last_scope = "SEEDS"
        for seed_attempt in range(
            1,
            run.config.value_bank.max_seed_pack_attempts + 1,
        ):
            terminal_value_bank_error = False
            try:
                pack = router.build_seed_pack(task, taxonomy, rng)
                validation = validator.validate(pack, task.focus_labels, taxonomy)
                if validation.valid:
                    self.event_bus.publish(EventEnvelope(
                        event_type="seed.validated", correlation_id=task.run_id,
                        idempotency_key=f"task:{task.task_id}:seed:{pack.seed_pack_id}",
                        payload={
                            "task_id": task.task_id, "seed_pack": pack.dict(),
                            "seed_validation": validation.dict(), "seed_attempt": seed_attempt,
                        },
                    ))
                    return pack, validation
                last_issues = validation.issues
                last_scope = "SEEDS"
            except ContextSelectionError as exc:
                last_scope = "CONTEXT"
                last_issues = [ValidationIssue(type="incompatible_context_frame", scope="CONTEXT", reason=str(exc))]
            except UnsupportedDecoyError as exc:
                last_scope = "SEEDS"
                last_issues = [ValidationIssue(type="unsupported_decoy", scope="SEEDS", reason=str(exc))]
            except ValueBankError as exc:
                last_scope = "SEEDS"
                last_issues = [
                    ValidationIssue(
                        type="value_bank_error",
                        scope="SEEDS",
                        reason=str(exc),
                    )
                ]
                terminal_value_bank_error = True
            self.event_bus.publish(EventEnvelope(
                event_type="seed.rejected", correlation_id=task.run_id,
                idempotency_key=f"task:{task.task_id}:seed-attempt:{seed_attempt}:rejected",
                payload={
                    "task_id": task.task_id, "seed_attempt": seed_attempt,
                    "issues": [issue.dict() for issue in last_issues], "regeneration_scope": last_scope,
                },
            ))
            if terminal_value_bank_error:
                break
        raise SeedPlanningError(last_scope, last_issues)

    def generate_pending(self, run_id: str, limit: int) -> List[DataGenerationResult]:
        run = self.repository.get_run(run_id)
        if run.status in {RunStatus.COMPLETED, RunStatus.FAILED}:
            return []
        run.status = RunStatus.RUNNING
        self.repository.update_run(run)
        self.verifier.max_repairs_per_candidate = (
            run.config.verifier.max_repairs_per_candidate
        )
        taxonomy = self.taxonomy_service.get(run.taxonomy_version_id).labels
        output_validator = DeterministicOutputValidator(run.config.validation)
        effective_limit = min(limit, run.config.batch_size)
        results: List[DataGenerationResult] = []
        while len(results) < effective_limit and run.status == RunStatus.RUNNING:
            pending = [
                task for task in self.repository.list_tasks(run_id)
                if task.status not in {TaskStatus.ACCEPTED, TaskStatus.REJECTED}
            ]
            if not pending:
                break
            for task in pending:
                if len(results) >= effective_limit or run.status != RunStatus.RUNNING:
                    break
                result = self._process_task(
                    run=run,
                    task=task,
                    taxonomy=taxonomy,
                    output_validator=output_validator,
                )
                if result is not None:
                    results.append(result)
        if results and run.status != RunStatus.COMPLETED:
            path = self.dataset_writer.write_partial(
                run_name=run.name,
                run_id=run.run_id,
                samples=self.repository.list_formatted_samples(run.run_id),
            )
            run.output_path = str(path.resolve())
            self.repository.update_run(run)
        return results

    def _process_task(
        self,
        *,
        run: Run,
        task: GenerationTask,
        taxonomy: List[TaxonomyLabel],
        output_validator: DeterministicOutputValidator,
    ) -> DataGenerationResult | None:
        slot_no = task.slot_no or task.sequence_no
        logger.info(
            "[sample %s/%s] start type=%s structure=%s variant=%s length=%s labels=%s",
            slot_no,
            run.config.num_samples,
            task.sample_type,
            task.sample_structure.type,
            task.diversity_profile.document_structure,
            task.length_target.bucket,
            ",".join(task.focus_labels),
        )
        try:
            seed_pack, seed_validation = self._seed_states.get(task.task_id) or (
                self._build_valid_seed_pack(task, taxonomy, run)
            )
            self._seed_states[task.task_id] = (seed_pack, seed_validation)
            logger.info(
                "[sample %s/%s] seed ready positives=%s decoys=%s",
                slot_no,
                run.config.num_samples,
                len(seed_pack.positive_entities),
                len(seed_pack.decoys),
            )
        except SeedPlanningError as exc:
            logger.error(
                "[sample %s/%s] seed planning failed scope=%s issues=%s",
                slot_no,
                run.config.num_samples,
                exc.regeneration_scope,
                len(exc.issues),
            )
            self._reject_task(run, task, exc.regeneration_scope, exc.issues)
            return None

        generation_taxonomy_context = self.taxonomy_service.generation_context(
            run.taxonomy_version_id,
            task,
        )
        retry_rng = random.Random(task.random_seed ^ 0x5EED_77)
        retry_router = RegenerationRouter(
            build_sample_type_router(
                run.config.value_bank,
                run.config.hard_negative,
            ),
            ContextFrameSelector(),
        )
        seed_validator = SeedPackValidator(run.config.hard_negative, run.config.validation)
        start_attempt = max(1, task.current_attempt or 1)
        if task.status not in {
            TaskStatus.GENERATED,
            TaskStatus.VALIDATING,
            TaskStatus.VERIFYING,
            TaskStatus.REPAIRING,
        }:
            start_attempt = max(1, task.current_attempt + 1)

        for attempt_no in range(start_attempt, task.max_attempts + 1):
            try:
                reflection = self._reflections.get(task.task_id)
                if reflection is not None:
                    logger.info(
                        "[sample %s/%s] retry feedback for attempt %s: %s",
                        slot_no,
                        run.config.num_samples,
                        attempt_no,
                        " | ".join(reflection.mandatory_repairs),
                    )
                logger.info(
                    "[sample %s/%s] generator attempt %s/%s started",
                    slot_no,
                    run.config.num_samples,
                    attempt_no,
                    task.max_attempts,
                )
                candidate = self.generator.generate(
                    DataGenerationRequest(
                        task=task,
                        seed_pack=seed_pack,
                        seed_validation=seed_validation,
                        taxonomy_context=generation_taxonomy_context,
                        attempt_no=attempt_no,
                        reflection=self._reflections.get(task.task_id),
                        correlation_id=run.run_id,
                    ),
                    output_validator,
                )
                self._record_usage(
                    run.run_id,
                    task.slot_no or task.sequence_no,
                    "generator",
                    candidate.token_usage,
                    task.task_id,
                    attempt_no,
                )
                logger.info(
                    "[sample %s/%s] generator finished tokens=%s",
                    slot_no,
                    run.config.num_samples,
                    candidate.token_usage.total_tokens,
                )
                task.status = TaskStatus.VALIDATING
                self.repository.update_task(task)
                validation, novelty = self._validate_candidate(
                    candidate, seed_pack, task, output_validator
                )
                deterministic_route, deterministic_issues = (
                    DeterministicIssueRouter.route(validation)
                )
                logger.info(
                    "[sample %s/%s] deterministic validation route=%s issues=%s",
                    slot_no,
                    run.config.num_samples,
                    deterministic_route,
                    len(validation.issues),
                )
                self._log_validation_feedback(
                    task=task,
                    attempt_no=attempt_no,
                    source="deterministic",
                    issues=validation.issues,
                )
                self.event_bus.publish(EventEnvelope(
                    event_type="data.deterministic.validated",
                    correlation_id=run.run_id,
                    idempotency_key=(
                        f"{task.task_id}:attempt:{attempt_no}:deterministic-evaluated"
                    ),
                    payload={
                        "task_id": task.task_id,
                        "attempt_no": attempt_no,
                        "route": deterministic_route,
                        "validation": validation.dict(),
                    },
                ))
                if deterministic_route == "REJECTED":
                    self._reject_task(run, task, "TEXT", validation.issues)
                    return None
                if deterministic_route == "REGENERATE":
                    raise OutputValidationError(validation)
                verifier_enabled = run.config.verifier.enabled
                if not verifier_enabled:
                    if deterministic_route in {"REGENERATE", "FIXABLE"}:
                        raise OutputValidationError(validation)
                    return self._accept_candidate(
                        run=run,
                        task=task,
                        candidate=candidate,
                        validation=validation,
                        novelty=novelty,
                        trace=None,
                    )

                task.status = TaskStatus.VERIFYING
                self.repository.update_task(task)
                logger.info(
                    "[sample %s/%s] verifier started",
                    slot_no,
                    run.config.num_samples,
                )
                verified, trace = self.verifier.verify(
                    candidate=candidate,
                    task=task,
                    seed_pack=seed_pack,
                    taxonomy_context=generation_taxonomy_context,
                    revalidate=lambda repaired: self._validate_candidate(
                        repaired, seed_pack, task, output_validator
                    )[0],
                    deterministic_issues=deterministic_issues,
                )
                logger.info(
                    "[sample %s/%s] verifier finished outcome=%s",
                    slot_no,
                    run.config.num_samples,
                    trace.outcome,
                )
                self._record_verifier_trace(run.run_id, task, attempt_no, trace)
                final_validation, final_novelty = self._validate_candidate(
                    verified, seed_pack, task, output_validator
                )
                if not final_validation.valid:
                    raise OutputValidationError(final_validation)
                return self._accept_candidate(
                    run=run,
                    task=task,
                    candidate=verified,
                    validation=final_validation,
                    novelty=final_novelty,
                    trace=trace,
                )
            except VerificationRoutingError as exc:
                logger.warning(
                    "[sample %s/%s] verifier route=%s issues=%s attempt=%s/%s",
                    slot_no,
                    run.config.num_samples,
                    exc.status,
                    len(exc.issues),
                    attempt_no,
                    task.max_attempts,
                )
                for index, issue in enumerate(exc.issues, start=1):
                    logger.warning(
                        "[sample %s/%s] verifier feedback %s/%s "
                        "type=%s severity=%s field=%s reason=%s suggested_fix=%s",
                        slot_no,
                        run.config.num_samples,
                        index,
                        len(exc.issues),
                        issue.type,
                        issue.severity,
                        issue.field,
                        issue.reason,
                        issue.suggested_fix,
                    )
                if exc.trace is not None:
                    self._record_verifier_trace(
                        run.run_id, task, attempt_no, exc.trace
                    )
                    self.event_bus.publish(EventEnvelope(
                        event_type="data.verification.rejected",
                        correlation_id=run.run_id,
                        idempotency_key=(
                            f"{task.task_id}:attempt:{attempt_no}:verification-rejected"
                        ),
                        payload={
                            "task_id": task.task_id,
                            "attempt_no": attempt_no,
                            "status": exc.status,
                            "issues": [issue.dict() for issue in exc.issues],
                        },
                    ))
                validation_issues = [
                    ValidationIssue(
                        type=issue.type,
                        scope="TEXT",
                        reason=issue.reason,
                    )
                    for issue in exc.issues
                ]
                if exc.status == "REJECTED":
                    self._reject_task(run, task, "TEXT", validation_issues)
                    return None
                error = OutputValidationError(
                    DeterministicValidationResult(
                        valid=False,
                        issues=validation_issues,
                    )
                )
                self._set_reflection(task, error.result)
                if attempt_no == task.max_attempts:
                    self._reject_task(run, task, error.regeneration_scope, error.result.issues)
                    return None
            except VerifierInfrastructureError as exc:
                logger.error(
                    "[sample %s/%s] verifier infrastructure failure stage=%s: %s",
                    slot_no,
                    run.config.num_samples,
                    exc.stage or "unknown",
                    exc,
                )
                if exc.token_usage is not None:
                    category = (
                        "verifier_repair"
                        if exc.stage == "repair"
                        else "verifier_judge"
                    )
                    self._usage_bucket(
                        run.run_id, task.slot_no or task.sequence_no
                    )[category].append(exc.token_usage)
                task.status = TaskStatus.VERIFYING
                self.repository.update_task(task)
                raise
            except OutputValidationError as exc:
                logger.warning(
                    "[sample %s/%s] regeneration requested scope=%s issues=%s attempt=%s/%s",
                    slot_no,
                    run.config.num_samples,
                    exc.regeneration_scope,
                    len(exc.result.issues),
                    attempt_no,
                    task.max_attempts,
                )
                self._log_validation_feedback(
                    task=task,
                    attempt_no=attempt_no,
                    source="regeneration",
                    issues=exc.result.issues,
                )
                self._publish_generation_rejection(
                    run.run_id, task, attempt_no, exc.result
                )
                self._set_reflection(task, exc.result)
                if attempt_no < task.max_attempts and exc.regeneration_scope != "TEXT":
                    seed_pack = retry_router.route(
                        exc.regeneration_scope,
                        task=task,
                        taxonomy=taxonomy,
                        rng=retry_rng,
                        current=seed_pack,
                    )
                    seed_validation = seed_validator.validate(
                        seed_pack, task.focus_labels, taxonomy
                    )
                    if not seed_validation.valid:
                        self._reject_task(
                            run, task, exc.regeneration_scope, seed_validation.issues
                        )
                        return None
                    self._seed_states[task.task_id] = (seed_pack, seed_validation)
                if attempt_no == task.max_attempts:
                    self._reject_task(
                        run, task, exc.regeneration_scope, exc.result.issues
                    )
                    return None
            task.status = TaskStatus.GENERATING
            self.repository.update_task(task)
        return None

    @staticmethod
    def _log_validation_feedback(
        *,
        task: GenerationTask,
        attempt_no: int,
        source: str,
        issues: List[ValidationIssue],
    ) -> None:
        for index, issue in enumerate(issues, start=1):
            logger.warning(
                "[sample %s attempt %s] %s feedback %s/%s "
                "type=%s scope=%s label=%s reason=%s",
                task.slot_no or task.sequence_no,
                attempt_no,
                source,
                index,
                len(issues),
                issue.type,
                issue.scope,
                issue.label or "-",
                issue.reason,
            )

    def _validate_candidate(
        self,
        candidate: GenerationCandidate,
        seed_pack: SeedPack,
        task: GenerationTask,
        output_validator: DeterministicOutputValidator,
    ) -> tuple[DeterministicValidationResult, NoveltyAssessment]:
        validation = output_validator.validate(
            tagged_text=candidate.tagged_text,
            entities=candidate.entities,
            seed_pack=seed_pack,
            focus_labels=task.focus_labels,
            allowed_labels=(task.annotation_labels or task.focus_labels),
            max_entities=max(task.max_entities, len(candidate.entities)),
            length_target=(
                task.length_target if self.enforce_quality_targets else None
            ),
        )
        if candidate.taxonomy_context_used is not None:
            imitation_issue = FewShotImitationGuard().find_imitation(
                candidate.tagged_text,
                candidate.taxonomy_context_used.focus_label.examples,
            )
            if imitation_issue is not None:
                validation = DeterministicValidationResult(
                    valid=False,
                    issues=[*validation.issues, imitation_issue],
                )
        novelty_config = output_validator.config
        if not novelty_config.quality_checks_enabled:
            return validation, NoveltyAssessment()
        novelty_guard = NoveltyGuard(
            near_duplicate_threshold=novelty_config.near_duplicate_threshold,
            recent_window=novelty_config.novelty_recent_window,
            max_feedback=novelty_config.max_novelty_feedback,
        )
        novelty = novelty_guard.assess(
            tagged_text=candidate.tagged_text,
            entities=candidate.entities,
            decoy_values=[decoy.value for decoy in seed_pack.decoys],
            previous_results=(
                [] if novelty_config.novelty_mode == "off"
                else self.repository.list_results(task.run_id)
            ),
        )
        if (
            validation.valid
            and novelty_config.novelty_mode == "enforce"
            and not novelty.valid
        ):
            validation = DeterministicValidationResult(
                valid=False,
                issues=novelty.issues,
            )
        return validation, novelty

    def _accept_candidate(
        self,
        *,
        run: Run,
        task: GenerationTask,
        candidate: GenerationCandidate,
        validation: DeterministicValidationResult,
        novelty: NoveltyAssessment,
        trace: VerificationTrace | None,
    ) -> DataGenerationResult:
        if trace is not None:
            self.event_bus.publish(EventEnvelope(
                event_type="data.verification.judged",
                correlation_id=run.run_id,
                idempotency_key=(
                    f"{task.task_id}:attempt:{candidate.attempt_no}:verification-pass"
                ),
                payload={
                    "task_id": task.task_id,
                    "attempt_no": candidate.attempt_no,
                    "verification_trace": trace.dict(),
                },
            ))
        task.status = TaskStatus.FORMATTING
        self.repository.update_task(task)
        formatted = self.formatter.format(
            tagged_text=candidate.tagged_text,
            entities=candidate.entities,
            allowed_labels=(task.annotation_labels or task.focus_labels),
        )
        result = DataGenerationResult(
            **candidate.dict(exclude={"created_at"}),
            output_validation=validation,
            sentence_skeleton=novelty.sentence_skeleton,
            novelty_assessment=novelty,
            verification_trace=trace,
            formatted_sample=formatted,
            pipeline_token_usage=self._pipeline_usage(
                run.run_id, task.slot_no or task.sequence_no
            ),
        )
        self.repository.add_result(result)
        self.repository.add_formatted_sample(task.task_id, formatted)
        task.status = TaskStatus.ACCEPTED
        self.repository.update_task(task)
        self.event_bus.publish(EventEnvelope(
            event_type="sample.formatted",
            correlation_id=run.run_id,
            idempotency_key=f"task:{task.task_id}:sample-formatted",
            payload={"task_id": task.task_id, "sample": formatted.dict()},
        ))
        self.event_bus.publish(EventEnvelope(
            event_type="sample.accepted",
            correlation_id=run.run_id,
            idempotency_key=f"task:{task.task_id}:sample-accepted",
            payload={"task_id": task.task_id, "slot_no": task.slot_no},
        ))

        samples = self.repository.list_formatted_samples(run.run_id)
        logger.info(
            "[sample %s/%s] accepted progress=%s/%s verifier=%s",
            task.slot_no or task.sequence_no,
            run.config.num_samples,
            len(samples),
            run.config.num_samples,
            trace.outcome if trace is not None else "disabled",
        )
        if len(samples) >= run.config.num_samples:
            path = self.dataset_writer.finalize(
                run_name=run.name,
                run_id=run.run_id,
                samples=samples,
            )
            run.status = RunStatus.COMPLETED
            run.output_path = str(path.resolve())
            self.event_bus.publish(EventEnvelope(
                event_type="run.completed",
                correlation_id=run.run_id,
                idempotency_key=f"run:{run.run_id}:completed",
                payload={
                    "run_id": run.run_id,
                    "samples": len(samples),
                    "output_path": run.output_path,
                },
            ))
        self.repository.update_run(run)
        return result

    def _reject_task(
        self,
        run: Run,
        task: GenerationTask,
        scope: str,
        issues: List[ValidationIssue],
    ) -> None:
        logger.error(
            "[sample %s/%s] rejected scope=%s issues=%s replacement=%s/%s",
            task.slot_no or task.sequence_no,
            run.config.num_samples,
            scope,
            len(issues),
            task.replacement_no,
            run.config.max_task_replacements,
        )
        task.status = TaskStatus.REJECTED
        self.repository.update_task(task)
        self.event_bus.publish(EventEnvelope(
            event_type="generation.task.rejected",
            correlation_id=run.run_id,
            idempotency_key=(
                f"task:{task.task_id}:rejected:{scope}:{task.replacement_no}"
            ),
            payload={
                "task_id": task.task_id,
                "slot_no": task.slot_no,
                "regeneration_scope": scope,
                "issues": [issue.dict() for issue in issues],
            },
        ))
        if task.replacement_no < run.config.max_task_replacements:
            sequence_no = max(
                item.sequence_no for item in self.repository.list_tasks(run.run_id)
            ) + 1
            replacement_seed = random.Random(
                task.random_seed ^ sequence_no ^ 0xA11CE
            ).randint(1, 2_147_483_647)
            diversity_profile = DiversityPlanner(replacement_seed).plan(
                task.focus_labels,
                task.sample_structure,
            )
            alternative_frames = compatible_context_frames(
                task.focus_labels,
                excluded_frame_ids=(
                    (task.diversity_profile.context_frame_id,)
                    if task.diversity_profile.context_frame_id
                    else ()
                ),
            )
            if alternative_frames:
                replacement_rng = random.Random(replacement_seed ^ 0xC07E87)
                diversity_profile = diversity_profile.copy(update={
                    "context_frame_id": replacement_rng.choice(
                        alternative_frames
                    ).frame_id,
                })
            replacement = task.copy(update={
                "task_id": str(uuid4()),
                "sequence_no": sequence_no,
                "replacement_no": task.replacement_no + 1,
                "parent_task_id": task.task_id,
                "random_seed": replacement_seed,
                "diversity_profile": diversity_profile,
                "current_attempt": 0,
                "status": TaskStatus.CREATED,
            })
            self.repository.add_task(replacement)
            logger.warning(
                "[sample %s/%s] replacement scheduled %s/%s",
                task.slot_no or task.sequence_no,
                run.config.num_samples,
                replacement.replacement_no,
                run.config.max_task_replacements,
            )
            self.event_bus.publish(EventEnvelope(
                event_type="generation.task.replaced",
                correlation_id=run.run_id,
                idempotency_key=f"task:{task.task_id}:replacement:{replacement.task_id}",
                payload={
                    "rejected_task_id": task.task_id,
                    "replacement_task": replacement.dict(),
                },
            ))
            return
        run.status = RunStatus.FAILED
        self.repository.update_run(run)
        self.event_bus.publish(EventEnvelope(
            event_type="run.failed",
            correlation_id=run.run_id,
            idempotency_key=f"run:{run.run_id}:failed:slot:{task.slot_no}",
            payload={
                "run_id": run.run_id,
                "slot_no": task.slot_no,
                "reason": "task replacement budget exhausted",
            },
        ))

    def _set_reflection(
        self,
        task: GenerationTask,
        validation: DeterministicValidationResult,
    ) -> None:
        self._reflections[task.task_id] = ReflectionContext(
            mandatory_repairs=[issue.reason for issue in validation.issues],
            must_not_repeat=[
                {"type": issue.type, "reason": issue.reason}
                for issue in validation.issues
            ],
        )

    def _publish_generation_rejection(
        self,
        run_id: str,
        task: GenerationTask,
        attempt_no: int,
        validation: DeterministicValidationResult,
    ) -> None:
        self.event_bus.publish(EventEnvelope(
            event_type="data.generation.rejected",
            correlation_id=run_id,
            idempotency_key=(
                f"{task.task_id}:attempt:{attempt_no}:deterministic-rejected"
            ),
            payload={
                "task_id": task.task_id,
                "attempt_no": attempt_no,
                "seed_pack_id": self._seed_states[task.task_id][0].seed_pack_id,
                "context_frame_id": (
                    self._seed_states[task.task_id][0].context_frame.frame_id
                ),
                "output_validation": validation.dict(),
                "regeneration_scope": OutputValidationError(
                    validation
                ).regeneration_scope,
            },
        ))

    def _usage_bucket(self, run_id: str, slot_no: int) -> Dict[str, List[TokenUsage]]:
        return self._usage_calls.setdefault(
            (run_id, slot_no),
            {
                "generator": [],
                "verifier_judge": [],
                "verifier_repair": [],
                "verifier_rejudge": [],
            },
        )

    def _record_usage(
        self,
        run_id: str,
        slot_no: int,
        category: str,
        usage: TokenUsage,
        task_id: str,
        attempt_no: int,
    ) -> None:
        key = (task_id, attempt_no, category)
        if key in self._recorded_calls:
            return
        self._recorded_calls.add(key)
        self._usage_bucket(run_id, slot_no)[category].append(usage)

    def _record_verifier_trace(
        self,
        run_id: str,
        task: GenerationTask,
        attempt_no: int,
        trace: VerificationTrace,
    ) -> None:
        slot_no = task.slot_no or task.sequence_no
        self._record_usage(
            run_id,
            slot_no,
            "verifier_judge",
            trace.initial_judge.token_usage,
            task.task_id,
            attempt_no,
        )
        if trace.repair is not None:
            self._record_usage(
                run_id,
                slot_no,
                "verifier_repair",
                trace.repair.token_usage,
                task.task_id,
                attempt_no,
            )
        if trace.final_judge is not None:
            self._record_usage(
                run_id,
                slot_no,
                "verifier_rejudge",
                trace.final_judge.token_usage,
                task.task_id,
                attempt_no,
            )

    def _pipeline_usage(self, run_id: str, slot_no: int) -> PipelineTokenUsage:
        calls = self._usage_bucket(run_id, slot_no)
        return PipelineTokenUsage.from_calls(**calls)
