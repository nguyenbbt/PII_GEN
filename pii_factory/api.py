from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse, Response

from .bootstrap import build_pipeline
from .application.verification import VerifierInfrastructureError
from .domain.models import CreateRunRequest, DataGenerationResult, Run, RunConfig, TaxonomySnapshot


def create_app(
    offline: bool = False,
    taxonomy_path: Path | str | None = None,
) -> FastAPI:
    pipeline, repository, event_bus = build_pipeline(offline=offline)
    default_taxonomy = pipeline.taxonomy_service.import_json(
        Path(
            taxonomy_path
            or os.getenv("PII_TAXONOMY_PATH", "pii_taxonomy_rules.json")
        )
    )
    app = FastAPI(title="PII Data Factory", version="0.4.0")
    app.state.default_taxonomy_version_id = default_taxonomy.version_id
    app.state.default_taxonomy_label_count = len(default_taxonomy.labels)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """Make opening the local server in a browser useful by default."""
        return RedirectResponse(url="/docs")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "offline" if offline else "azure"}

    @app.get("/favicon.ico", include_in_schema=False, status_code=204)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/v1/taxonomies", response_model=List[TaxonomySnapshot])
    def list_taxonomies() -> List[TaxonomySnapshot]:
        return pipeline.taxonomy_service.list_versions()

    @app.get("/api/v1/taxonomies/{version_id}", response_model=TaxonomySnapshot)
    def get_taxonomy(version_id: str) -> TaxonomySnapshot:
        try:
            return pipeline.taxonomy_service.get(version_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/v1/runs", response_model=Run, status_code=201)
    def create_run(config: RunConfig) -> Run:
        try:
            return pipeline.create_run(CreateRunRequest(
                run_name=config.run_name,
                taxonomy_version_id=default_taxonomy.version_id,
                config=config,
            ))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/taxonomies/{version_id}/runs", response_model=Run, status_code=201)
    def create_run_from_config(version_id: str, config: RunConfig) -> Run:
        try:
            pipeline.taxonomy_service.get(version_id)
            return pipeline.create_run(CreateRunRequest(
                run_name=config.run_name,
                taxonomy_version_id=version_id,
                config=config,
            ))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_id}", response_model=Run)
    def get_run(run_id: str) -> Run:
        try:
            return repository.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_id}/tasks")
    def list_tasks(run_id: str) -> List[dict]:
        try:
            repository.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return [task.dict() for task in repository.list_tasks(run_id)]

    @app.post("/api/v1/runs/{run_id}/generate", response_model=List[DataGenerationResult])
    def generate(run_id: str, limit: int = Query(1, ge=1, le=100)) -> List[DataGenerationResult]:
        try:
            repository.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            return pipeline.generate_pending(run_id, limit)
        except VerifierInfrastructureError as exc:
            raise HTTPException(
                status_code=503,
                detail="LLM Verifier is temporarily unavailable",
            ) from exc

    @app.get("/api/v1/runs/{run_id}/events")
    def events(run_id: str) -> List[dict]:
        return [event.dict() for event in event_bus.list_events() if event.correlation_id == run_id]

    return app
