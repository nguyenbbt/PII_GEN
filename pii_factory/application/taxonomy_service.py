from __future__ import annotations

from pathlib import Path

from ..domain.models import (
    EventEnvelope,
    GenerationTask,
    GenerationTaxonomyContext,
    TaxonomySnapshot,
)
from ..ports import EventBus, RunRepository
from ..infrastructure.json_taxonomy import JsonTaxonomyParser
from .taxonomy_context import TaxonomyContextSelector


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_taxonomy_path(path: Path | str) -> Path:
    """Resolve a taxonomy from the caller's cwd, then the installed project."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    for base in (Path.cwd(), _PROJECT_ROOT):
        resolved = (base / candidate).resolve()
        if resolved.is_file():
            return resolved
    return (Path.cwd() / candidate).resolve()


class TaxonomyService:
    """Owns immutable taxonomy versions and supplies focused context to workers."""

    def __init__(self, repository: RunRepository, event_bus: EventBus) -> None:
        self.repository = repository
        self.event_bus = event_bus
        self._context_selector = TaxonomyContextSelector()

    def register(self, taxonomy: TaxonomySnapshot) -> TaxonomySnapshot:
        self.repository.add_taxonomy(taxonomy)
        self.event_bus.publish(EventEnvelope(
            event_type="taxonomy.version.created",
            idempotency_key=f"taxonomy:{taxonomy.version_id}:created",
            payload=taxonomy.dict(),
        ))
        return taxonomy

    def import_json(self, path: Path) -> TaxonomySnapshot:
        """Create one immutable taxonomy version from the canonical JSON file."""
        return self.register(
            JsonTaxonomyParser().parse_file(resolve_taxonomy_path(path))
        )

    def get(self, version_id: str) -> TaxonomySnapshot:
        return self.repository.get_taxonomy(version_id)

    def list_versions(self) -> list[TaxonomySnapshot]:
        return self.repository.list_taxonomies()

    def generation_context(
        self,
        version_id: str,
        task: GenerationTask,
    ) -> GenerationTaxonomyContext:
        return self._context_selector.select(self.get(version_id), task)
