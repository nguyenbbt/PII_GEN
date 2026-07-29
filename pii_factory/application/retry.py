from __future__ import annotations

import random
from typing import Collection, Literal, Mapping, Sequence

from ..domain.models import ContentSeeds, GenerationTask, SeedPack, TaxonomyLabel
from .seed_generation import ContextFrameSelector, SampleTypeRouter


RegenerationScope = Literal["TEXT", "SEEDS", "CONTEXT"]


class RegenerationRouter:
    """Applies retry scope without conflating seed, context, and text failures."""

    def __init__(self, seed_router: SampleTypeRouter, context_selector: ContextFrameSelector) -> None:
        self.seed_router = seed_router
        self.context_selector = context_selector

    def route(
        self,
        scope: RegenerationScope,
        *,
        task: GenerationTask,
        taxonomy: Sequence[TaxonomyLabel],
        rng: random.Random,
        current: SeedPack,
        excluded_values_by_label: Mapping[str, Collection[str]] | None = None,
    ) -> SeedPack:
        if scope == "TEXT":
            return current
        if scope == "SEEDS":
            return self.seed_router.build_seed_pack(
                task,
                taxonomy,
                rng,
                excluded_values_by_label=excluded_values_by_label,
                excluded_decoy_strategy_ids={
                    decoy.strategy_id
                    for decoy in current.decoys
                },
            )
        if scope == "CONTEXT":
            domain_sets = [
                set(decoy.realization_plan.compatible_domains)
                for decoy in current.decoys
                if (
                    decoy.realization_plan is not None
                    and decoy.realization_plan.compatible_domains
                )
            ]
            compatible_domains = (
                sorted(set.intersection(*domain_sets))
                if domain_sets
                else []
            )
            new_frame = self.context_selector.select(
                task.focus_labels,
                rng,
                decoy_semantic_types=[decoy.semantic_type for decoy in current.decoys],
                compatible_domains=compatible_domains,
                excluded_frame_ids=[current.context_frame.frame_id],
            )
            updates: dict[str, object] = {"context_frame": new_frame}
            if current.content_seeds is not None:
                updates["content_seeds"] = ContentSeeds(
                    domain=new_frame.domain,
                    document_type=new_frame.document_type,
                    tone=new_frame.tone,
                    generic_roles=current.content_seeds.generic_roles,
                    actions=current.content_seeds.actions,
                    objects=current.content_seeds.objects,
                )
            return current.copy(update=updates, deep=True)
        raise ValueError(f"unsupported regeneration scope: {scope}")
