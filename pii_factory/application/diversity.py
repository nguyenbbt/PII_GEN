from __future__ import annotations

import random
from collections import Counter
from typing import Sequence

from ..domain.models import DiversityProfile, SampleStructureConfig
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

    def __init__(self, random_seed: int) -> None:
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
        self._contract_structures = _QuotaAxis(
            (
                "agreement_clause",
                "administrative_record",
                "company_notice",
                "handover_minutes",
            ),
            self._rng,
        )
        self._chat_structures = _QuotaAxis(
            ("friend_chat", "customer_support_chat"),
            self._rng,
        )
        self._registers = _QuotaAxis(
            ("formal", "neutral", "informal", "concise_technical"), self._rng
        )
        self._lengths = _QuotaAxis(("short", "medium", "long"), self._rng)

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
                document_structure = self._contract_structures.next()
                language_register = self._rng.choice(
                    ("formal", "neutral", "concise_technical")
                )
            elif sample_structure.type == "chat":
                document_structure = self._chat_structures.next()
                language_register = (
                    "informal"
                    if document_structure == "friend_chat"
                    else "neutral"
                )
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
