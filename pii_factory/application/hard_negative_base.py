from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Callable

from ..domain.models import DecoySeed


@dataclass(frozen=True)
class DecoyStrategy:
    strategy_id: str
    target_label: str
    family: str
    semantic_type: str
    required_context_cues: tuple[str, ...]
    forbidden_context_cues: tuple[str, ...]
    negative_labels: tuple[str, ...]
    possible_collision_labels: tuple[str, ...]
    generator: Callable[[random.Random], str]
    validator: Callable[[str], bool]

    def build(self, rng: random.Random) -> DecoySeed:
        return DecoySeed(
            strategy_id=self.strategy_id,
            target_label=self.target_label,
            value=self.generator(rng),
            family=self.family,
            semantic_type=self.semantic_type,
            negative_labels=list(self.negative_labels),
            possible_collision_labels=list(self.possible_collision_labels),
            required_context_cues=list(self.required_context_cues),
            forbidden_context_cues=list(self.forbidden_context_cues),
        )


def strategy(
    strategy_id: str,
    target_label: str,
    family: str,
    semantic_type: str,
    required_context_cues: tuple[str, ...],
    forbidden_context_cues: tuple[str, ...],
    generator: Callable[[random.Random], str],
    validator: Callable[[str], bool],
    *,
    negative_labels: tuple[str, ...] | None = None,
    possible_collision_labels: tuple[str, ...] = (),
) -> tuple[DecoyStrategy, ...]:
    return (DecoyStrategy(
        strategy_id=strategy_id,
        target_label=target_label,
        family=family,
        semantic_type=semantic_type,
        required_context_cues=required_context_cues,
        forbidden_context_cues=forbidden_context_cues,
        negative_labels=negative_labels or (target_label,),
        possible_collision_labels=possible_collision_labels,
        generator=generator,
        validator=validator,
    ),)


def digits(rng: random.Random, count: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(count))
