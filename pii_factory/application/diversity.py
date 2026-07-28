from __future__ import annotations

import random
from collections import Counter
from math import floor
from typing import Mapping, Sequence

from ..domain.models import (
    DiversityProfile,
    LengthTarget,
    SampleStructureConfig,
)
from .context_catalog import compatible_context_frames


class _QuotaAxis:
    def __init__(self, values: Sequence[str], rng: random.Random) -> None:
        self._values = list(values)
        self._rng = rng
        self._remaining: list[str] = []

    def next(self) -> str:
        if not self._remaining:
            self._remaining = list(self._values)
            self._rng.shuffle(self._remaining)
        return self._remaining.pop()


class DiversityPlanner:
    """Builds reproducible, quota-balanced realization profiles without an LLM."""

    def __init__(
        self,
        random_seed: int,
        length_distribution: Mapping[str, float] | None = None,
        total_samples: int | None = None,
        sample_structures: Sequence[SampleStructureConfig] | None = None,
    ) -> None:
        self._rng = random.Random(random_seed ^ 0x5EED_D1)
        self._context_usage: Counter[str] = Counter()
        self._speaker_roles = _QuotaAxis(
            ("customer", "staff_member", "system_operator", "third_party"), self._rng
        )
        self._intents = _QuotaAxis(
            ("request_action", "report_issue", "confirm_information", "provide_update", "ask_for_help"),
            self._rng,
        )
        self._structures = _QuotaAxis(
            ("single_sentence", "two_sentence_note", "short_dialogue", "form_like_record"), self._rng
        )
        self._registers = _QuotaAxis(
            ("formal", "neutral", "informal", "concise_technical"), self._rng
        )
        length_values = (
            self._weighted_values(
                length_distribution,
                total_samples,
                ("short", "medium", "long"),
            )
            if length_distribution is not None and total_samples is not None
            else ["short", "medium", "long"]
        )
        self._lengths = _QuotaAxis(length_values, self._rng)
        self._sample_structures = list(sample_structures or ())

    @staticmethod
    def _weighted_values(
        distribution: Mapping[str, float],
        total_samples: int,
        order: Sequence[str],
    ) -> list[str]:
        raw_counts = {
            name: probability * total_samples
            for name, probability in distribution.items()
        }
        counts = {
            name: floor(value)
            for name, value in raw_counts.items()
        }
        remaining = total_samples - sum(counts.values())
        largest_remainders = sorted(
            distribution,
            key=lambda name: (
                raw_counts[name] - counts[name],
                distribution[name],
                name,
            ),
            reverse=True,
        )
        for name in largest_remainders[:remaining]:
            counts[name] += 1
        return [
            name
            for name in order
            if name in counts
            for _ in range(counts[name])
        ]

    def select_sample_structure(self) -> SampleStructureConfig:
        """Select only from the configured pool using this planner's seeded RNG."""
        if not self._sample_structures:
            raise ValueError("sample_structures must contain at least one configured structure")
        return self._rng.choice(self._sample_structures)

    def plan(
        self,
        focus_labels: Sequence[str],
        sample_structure: SampleStructureConfig | None = None,
    ) -> DiversityProfile:
        candidates = compatible_context_frames(focus_labels)
        if not candidates:
            raise ValueError(f"no diversity context supports labels: {sorted(set(focus_labels))}")
        minimum_usage = min(self._context_usage[frame.frame_id] for frame in candidates)
        least_used = [frame for frame in candidates if self._context_usage[frame.frame_id] == minimum_usage]
        frame = self._rng.choice(least_used)
        self._context_usage[frame.frame_id] += 1
        document_structure = self._structures.next()
        language_register = self._registers.next()
        if sample_structure is not None:
            if sample_structure.type == "contract":
                document_structure = "contract"
                language_register = self._rng.choice(
                    ("formal", "neutral", "concise_technical")
                )
            elif sample_structure.type == "chat":
                document_structure = "chat"
                language_register = self._rng.choice(("informal", "neutral"))
            else:
                document_structure = "custom_format"

        return DiversityProfile(
            context_frame_id=frame.frame_id,
            speaker_role=self._speaker_roles.next(),
            intent=self._intents.next(),
            document_structure=document_structure,
            language_register=language_register,
            length_bucket=self._lengths.next(),
        )


_WORD_TARGETS = {
    "short": (80, 120),
    "medium": (150, 230),
    "long": (260, 400),
}
_CONTRACT_UNIT_TARGETS = {
    "short": (3, 5),
    "medium": (6, 9),
    "long": (10, 14),
}
_CHAT_UNIT_TARGETS = {
    "short": (6, 8),
    "medium": (10, 14),
    "long": (16, 22),
}


def resolve_length_target(
    sample_structure: SampleStructureConfig,
    bucket: str,
) -> LengthTarget:
    min_words, max_words = _WORD_TARGETS[bucket]
    if sample_structure.type == "chat":
        min_units, max_units = _CHAT_UNIT_TARGETS[bucket]
        unit = "turns"
    elif sample_structure.type == "contract":
        min_units, max_units = _CONTRACT_UNIT_TARGETS[bucket]
        unit = "content_units"
    else:
        min_units = max_units = 1
        unit = "words"
    return LengthTarget(
        bucket=bucket,
        min_words=min_words,
        max_words=max_words,
        unit=unit,
        min_units=min_units,
        max_units=max_units,
    )
