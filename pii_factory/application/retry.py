from __future__ import annotations

import random
from typing import Literal, Sequence

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
    ) -> SeedPack:
        if scope == "TEXT":
            return current
        if scope == "SEEDS":
            return self.seed_router.build_seed_pack(task, taxonomy, rng)
        if scope == "CONTEXT":
            new_frame = self.context_selector.select(
                task.focus_labels,
                rng,
                decoy_semantic_types=[decoy.semantic_type for decoy in current.decoys],
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
