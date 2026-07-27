from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Dict, List, Optional, Tuple

from ..domain.models import (
    DataGenerationResult,
    EventEnvelope,
    FormattedSample,
    GenerationCandidate,
    GenerationTask,
    Run,
    TaxonomySnapshot,
)


class InMemoryRepository:
    """Thread-safe adapter for local development and tests.

    The application only depends on the repository port, so this can be replaced by
    PostgreSQL tables and transactional outbox in production.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._runs: Dict[str, Run] = {}
        self._taxonomies: Dict[str, TaxonomySnapshot] = {}
        self._tasks: Dict[str, GenerationTask] = {}
        self._candidates: Dict[Tuple[str, int], GenerationCandidate] = {}
        self._results: Dict[Tuple[str, int], DataGenerationResult] = {}
        self._formatted_samples: Dict[str, FormattedSample] = {}
        self._consumed: set[str] = set()

    def add_run(self, run: Run) -> None:
        with self._lock:
            self._runs[run.run_id] = run

    def add_taxonomy(self, taxonomy: TaxonomySnapshot) -> None:
        with self._lock:
            if taxonomy.version_id in self._taxonomies:
                raise ValueError(f"taxonomy version already exists: {taxonomy.version_id}")
            self._taxonomies[taxonomy.version_id] = taxonomy

    def get_taxonomy(self, version_id: str) -> TaxonomySnapshot:
        try:
            return self._taxonomies[version_id]
        except KeyError as exc:
            raise KeyError(f"taxonomy not found: {version_id}") from exc

    def list_taxonomies(self) -> List[TaxonomySnapshot]:
        return sorted(self._taxonomies.values(), key=lambda taxonomy: taxonomy.created_at, reverse=True)

    def get_run(self, run_id: str) -> Run:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"run not found: {run_id}") from exc

    def update_run(self, run: Run) -> None:
        with self._lock:
            if run.run_id not in self._runs:
                raise KeyError(f"run not found: {run.run_id}")
            self._runs[run.run_id] = run

    def add_task(self, task: GenerationTask) -> None:
        with self._lock:
            self._tasks[task.task_id] = task

    def list_tasks(self, run_id: str) -> List[GenerationTask]:
        return sorted((task for task in self._tasks.values() if task.run_id == run_id), key=lambda task: task.sequence_no)

    def update_task(self, task: GenerationTask) -> None:
        with self._lock:
            if task.task_id not in self._tasks:
                raise KeyError(f"task not found: {task.task_id}")
            self._tasks[task.task_id] = task

    def add_candidate(self, candidate: GenerationCandidate) -> None:
        with self._lock:
            self._candidates[(candidate.task_id, candidate.attempt_no)] = candidate

    def get_candidate(self, task_id: str, attempt_no: int) -> Optional[GenerationCandidate]:
        return self._candidates.get((task_id, attempt_no))

    def add_result(self, result: DataGenerationResult) -> None:
        with self._lock:
            self._results[(result.task_id, result.attempt_no)] = result

    def get_result(self, task_id: str, attempt_no: int) -> Optional[DataGenerationResult]:
        return self._results.get((task_id, attempt_no))

    def list_results(self, run_id: str) -> List[DataGenerationResult]:
        task_ids = {task.task_id for task in self._tasks.values() if task.run_id == run_id}
        return sorted(
            (result for result in self._results.values() if result.task_id in task_ids),
            key=lambda result: (self._tasks[result.task_id].sequence_no, result.attempt_no),
        )

    def add_formatted_sample(self, task_id: str, sample: FormattedSample) -> None:
        with self._lock:
            self._formatted_samples[task_id] = sample

    def list_formatted_samples(self, run_id: str) -> List[FormattedSample]:
        accepted = [
            task for task in self._tasks.values()
            if task.run_id == run_id and task.task_id in self._formatted_samples
        ]
        return [
            self._formatted_samples[task.task_id]
            for task in sorted(
                accepted,
                key=lambda item: (
                    item.slot_no or item.sequence_no,
                    item.replacement_no,
                ),
            )
        ]

    def mark_consumed(self, idempotency_key: str) -> bool:
        with self._lock:
            if idempotency_key in self._consumed:
                return False
            self._consumed.add(idempotency_key)
            return True


class InMemoryEventBus:
    def __init__(self) -> None:
        self._events: List[EventEnvelope] = []
        self._lock = Lock()

    def publish(self, event: EventEnvelope) -> None:
        with self._lock:
            self._events.append(event)

    def list_events(self) -> List[EventEnvelope]:
        with self._lock:
            return list(self._events)
