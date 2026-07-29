from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Protocol

from .domain.models import (
    DataGenerationResult,
    EventEnvelope,
    FormattedSample,
    GenerationCandidate,
    GenerationTask,
    RepairResult,
    Run,
    TaxonomySnapshot,
    VerifierDecision,
)


class RunRepository(Protocol):
    def add_taxonomy(self, taxonomy: TaxonomySnapshot) -> None: ...
    def get_taxonomy(self, version_id: str) -> TaxonomySnapshot: ...
    def list_taxonomies(self) -> List[TaxonomySnapshot]: ...
    def add_run(self, run: Run) -> None: ...
    def get_run(self, run_id: str) -> Run: ...
    def update_run(self, run: Run) -> None: ...
    def add_task(self, task: GenerationTask) -> None: ...
    def list_tasks(self, run_id: str) -> List[GenerationTask]: ...
    def update_task(self, task: GenerationTask) -> None: ...
    def add_candidate(self, candidate: GenerationCandidate) -> None: ...
    def get_candidate(self, task_id: str, attempt_no: int) -> Optional[GenerationCandidate]: ...
    def add_result(self, result: DataGenerationResult) -> None: ...
    def get_result(self, task_id: str, attempt_no: int) -> Optional[DataGenerationResult]: ...
    def list_results(self, run_id: str) -> List[DataGenerationResult]: ...
    def add_formatted_sample(self, task_id: str, sample: FormattedSample) -> None: ...
    def list_formatted_samples(self, run_id: str) -> List[FormattedSample]: ...
    def mark_consumed(self, idempotency_key: str) -> bool: ...


class EventBus(Protocol):
    def publish(self, event: EventEnvelope) -> None: ...
    def list_events(self) -> List[EventEnvelope]: ...


@dataclass(frozen=True)
class CompletionResult:
    tagged_text: str
    entities: List[dict[str, str]]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str | None = None
    latency_ms: int | None = None

    @classmethod
    def coerce(
        cls,
        value: "CompletionResult | tuple[str, List[dict[str, str]], int, int, int]",
    ) -> "CompletionResult":
        if isinstance(value, cls):
            return value
        return cls(*value)

    def __iter__(self) -> Iterator[object]:
        """Keep positional unpacking compatible while callers migrate."""
        yield self.tagged_text
        yield self.entities
        yield self.input_tokens
        yield self.output_tokens
        yield self.total_tokens


class CompletionClientError(RuntimeError):
    """Generator content failure that retains usage from paid HTTP responses."""

    def __init__(
        self,
        message: str,
        *,
        raw_usage: tuple[int, int, int],
    ) -> None:
        self.raw_usage = raw_usage
        super().__init__(message)


class CompletionClient(Protocol):
    def generate(self, messages: List[dict[str, str]]) -> CompletionResult: ...


class VerifierClient(Protocol):
    def judge(self, messages: List[dict[str, str]]) -> VerifierDecision: ...
    def repair(self, messages: List[dict[str, str]]) -> RepairResult: ...
