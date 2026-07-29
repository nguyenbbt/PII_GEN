from __future__ import annotations

import random
from dataclasses import dataclass
from math import floor
from typing import Collection, Sequence

from ..domain.models import (
    ContextFrame,
    DecoyBlueprint,
    DecoyRealizationPlan,
    DecoySeed,
    GenerationTask,
    HardNegativeConfig,
    PositiveEntitySeed,
    TaxonomyLabel,
)
from .context_catalog import compatible_context_frames


class MixedDecoyPlanningError(ValueError):
    scope = "SEEDS"


@dataclass(frozen=True)
class MixedDecoyPlan:
    context_frame: ContextFrame
    decoys: tuple[DecoySeed, ...]


class MixedContrastiveDecoyPlanner:
    """Select taxonomy-backed decoys together with a compatible context."""

    _FAMILY_WEIGHTS = {
        "semantic_ambiguity": 0.40,
        "business_reference": 0.30,
        "operational_code": 0.20,
        "technical_schema": 0.10,
    }
    _TARGET_TECHNICAL_RATIO = 0.10
    _DISCOURSE_STAGES = ("intake", "processing", "decision")

    def __init__(self, config: HardNegativeConfig) -> None:
        self.config = config

    def plan(
        self,
        *,
        task: GenerationTask,
        taxonomy: Sequence[TaxonomyLabel],
        positive_entities: Sequence[PositiveEntitySeed],
        rng: random.Random,
        count: int,
        excluded_blueprint_ids: Collection[str] = (),
    ) -> MixedDecoyPlan:
        if not positive_entities:
            raise MixedDecoyPlanningError(
                "mixed_contrastive requires at least one positive anchor"
            )

        labels_by_code = {label.code: label for label in taxonomy}
        supported = [
            labels_by_code[code]
            for code in task.focus_labels
            if code in labels_by_code
            and labels_by_code[code].decoy_blueprints
        ]
        if not supported:
            raise MixedDecoyPlanningError(
                "none of the focus labels has taxonomy decoy blueprints"
            )

        structure = task.sample_structure.type
        base_frames = compatible_context_frames(task.focus_labels)
        if not base_frames:
            raise MixedDecoyPlanningError(
                "no context frame supports all mixed_contrastive labels"
            )

        selected_frame: ContextFrame | None = None
        decoys: list[DecoySeed] = []
        used_values: set[str] = set()
        technical_available = self._technical_slot_is_available(task)
        technical_used = False

        for _ in range(count):
            options = self._options(
                supported,
                structure=structure,
                frames=base_frames,
                fixed_domain=(
                    selected_frame.domain
                    if selected_frame is not None
                    else None
                ),
                allow_technical=technical_available and not technical_used,
                excluded_blueprint_ids=excluded_blueprint_ids,
                used_values=used_values,
            )
            if not options:
                raise MixedDecoyPlanningError(
                    "no decoy blueprint is compatible with the task context"
                )
            if technical_available and not technical_used:
                technical_options = [
                    option
                    for option in options
                    if option[1].technical
                ]
                if technical_options:
                    options = technical_options

            label, blueprint, compatible_frames = self._choose_option(
                options,
                rng,
            )
            if selected_frame is None:
                selected_frame = self._choose_frame(
                    compatible_frames,
                    task.diversity_profile.context_frame_id,
                    rng,
                )

            value = self._unique_surface(
                blueprint,
                used_values,
                rng,
            )
            used_values.add(value.casefold())
            anchor = self._choose_anchor(
                label.code,
                positive_entities,
                rng,
            )
            relation = rng.choice(blueprint.integration_relations)
            stage = rng.choice(self._DISCOURSE_STAGES)
            allowed_domains = (
                list(blueprint.compatible_domains)
                if blueprint.compatible_domains
                else sorted(
                    {
                        frame.domain
                        for frame in compatible_frames
                    }
                )
            )
            decoys.append(
                DecoySeed(
                    strategy_id=blueprint.id,
                    target_label=label.code,
                    value=value,
                    family=blueprint.family,
                    semantic_type="taxonomy_few_shot_contrast",
                    negative_labels=[label.code],
                    required_context_cues=list(blueprint.evidence_cues),
                    forbidden_context_cues=list(blueprint.forbidden_cues),
                    realization_plan=DecoyRealizationPlan(
                        blueprint_id=blueprint.id,
                        family=blueprint.family,
                        contrast_principle=blueprint.contrast_principle,
                        anchor_label=anchor.label,
                        relation=relation,
                        discourse_stage=stage,
                        evidence_cues=list(blueprint.evidence_cues),
                        source_example_ids=list(
                            blueprint.source_example_ids
                        ),
                        compatible_domains=allowed_domains,
                    ),
                )
            )
            technical_used = technical_used or blueprint.technical

        assert selected_frame is not None
        return MixedDecoyPlan(
            context_frame=selected_frame.copy(deep=True),
            decoys=tuple(decoys),
        )

    def _technical_slot_is_available(
        self,
        task: GenerationTask,
    ) -> bool:
        ordinal = task.hard_negative_ordinal
        if ordinal is None:
            return False
        ratio = min(
            self._TARGET_TECHNICAL_RATIO,
            self.config.technical_decoy_max_ratio,
        )
        return floor(ordinal * ratio) > floor((ordinal - 1) * ratio)

    def _options(
        self,
        labels: Sequence[TaxonomyLabel],
        *,
        structure: str,
        frames: Sequence[ContextFrame],
        fixed_domain: str | None,
        allow_technical: bool,
        excluded_blueprint_ids: Collection[str],
        used_values: Collection[str],
    ) -> list[
        tuple[TaxonomyLabel, DecoyBlueprint, list[ContextFrame]]
    ]:
        options: list[
            tuple[TaxonomyLabel, DecoyBlueprint, list[ContextFrame]]
        ] = []
        excluded = set(excluded_blueprint_ids)
        used = {
            value.casefold()
            for value in used_values
        }
        for label in labels:
            for blueprint in label.decoy_blueprints:
                if blueprint.id in excluded:
                    continue
                if not any(
                    template.casefold() not in used
                    for template in blueprint.surface_templates
                ):
                    continue
                if structure not in blueprint.compatible_structures:
                    continue
                if blueprint.technical and not allow_technical:
                    continue
                compatible = [
                    frame
                    for frame in frames
                    if (
                        fixed_domain is None
                        or frame.domain == fixed_domain
                    )
                    and (
                        not self.config.require_context_compatible_decoy
                        or not blueprint.compatible_domains
                        or frame.domain in blueprint.compatible_domains
                    )
                ]
                if compatible:
                    options.append((label, blueprint, compatible))
        return options

    def _choose_option(
        self,
        options: Sequence[
            tuple[TaxonomyLabel, DecoyBlueprint, list[ContextFrame]]
        ],
        rng: random.Random,
    ) -> tuple[TaxonomyLabel, DecoyBlueprint, list[ContextFrame]]:
        options_by_family: dict[
            str,
            list[
                tuple[
                    TaxonomyLabel,
                    DecoyBlueprint,
                    list[ContextFrame],
                ]
            ],
        ] = {}
        for option in options:
            options_by_family.setdefault(
                option[1].family,
                [],
            ).append(option)
        families = list(options_by_family)
        family = rng.choices(
            families,
            weights=[
                self._FAMILY_WEIGHTS[name]
                for name in families
            ],
            k=1,
        )[0]
        return rng.choice(options_by_family[family])

    @staticmethod
    def _choose_frame(
        frames: Sequence[ContextFrame],
        preferred_frame_id: str | None,
        rng: random.Random,
    ) -> ContextFrame:
        if preferred_frame_id:
            preferred = next(
                (
                    frame
                    for frame in frames
                    if frame.frame_id == preferred_frame_id
                ),
                None,
            )
            if preferred is not None:
                return preferred.copy(deep=True)
        return rng.choice(list(frames)).copy(deep=True)

    @staticmethod
    def _choose_anchor(
        target_label: str,
        positive_entities: Sequence[PositiveEntitySeed],
        rng: random.Random,
    ) -> PositiveEntitySeed:
        matching = [
            entity
            for entity in positive_entities
            if entity.label == target_label
        ]
        return rng.choice(matching or list(positive_entities))

    @staticmethod
    def _unique_surface(
        blueprint: DecoyBlueprint,
        used_values: set[str],
        rng: random.Random,
    ) -> str:
        available = [
            template
            for template in blueprint.surface_templates
            if template.casefold() not in used_values
        ]
        if not available:
            raise MixedDecoyPlanningError(
                f"blueprint {blueprint.id} has no unused surface template"
            )
        return rng.choice(available)
