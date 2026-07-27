from __future__ import annotations

from difflib import SequenceMatcher
import re
import unicodedata
from typing import Sequence

from ..domain.models import FewShotExample, ValidationIssue


_TAGGED_SPAN = re.compile(
    r"<([A-Za-z][A-Za-z0-9_]*)>.*?</\1>",
    re.DOTALL,
)
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9_]*>")
_TOKEN = re.compile(r"\w+", re.UNICODE)


class FewShotImitationGuard:
    """Reject candidates that copy a supplied few-shot sentence template."""

    def __init__(
        self,
        *,
        sequence_threshold: float = 0.84,
        ngram_threshold: float = 0.72,
    ) -> None:
        self.sequence_threshold = sequence_threshold
        self.ngram_threshold = ngram_threshold

    def find_imitation(
        self,
        tagged_text: str,
        examples: Sequence[FewShotExample],
    ) -> ValidationIssue | None:
        candidate = self._normalize(tagged_text)
        closest: tuple[float, float, FewShotExample] | None = None

        for example in examples:
            reference = self._normalize(example.expected_tagged_text)
            sequence_similarity = SequenceMatcher(
                None,
                candidate,
                reference,
            ).ratio()
            ngram_similarity = self._ngram_similarity(
                candidate,
                reference,
            )
            similarity = max(sequence_similarity, ngram_similarity)
            closest_similarity = (
                max(closest[:2])
                if closest is not None
                else -1.0
            )
            if similarity > closest_similarity:
                closest = (
                    sequence_similarity,
                    ngram_similarity,
                    example,
                )

        if closest is None:
            return None
        sequence_similarity, ngram_similarity, example = closest
        if (
            sequence_similarity < self.sequence_threshold
            and ngram_similarity < self.ngram_threshold
        ):
            return None
        return ValidationIssue(
            type="few_shot_imitation",
            scope="TEXT",
            reason=(
                f"candidate is too similar to few-shot {example.id!r} "
                f"(sequence={sequence_similarity:.3f}, "
                f"ngram={ngram_similarity:.3f}); regenerate with a different "
                "context, action, sentence structure, and wording"
            ),
        )

    @staticmethod
    def _normalize(text: str) -> str:
        masked = _TAGGED_SPAN.sub(" ENTITY ", text)
        clean = _TAG.sub(" ", masked)
        normalized = unicodedata.normalize("NFKC", clean).casefold()
        return " ".join(_TOKEN.findall(normalized))

    @staticmethod
    def _ngram_similarity(candidate: str, reference: str) -> float:
        candidate_tokens = candidate.split()
        reference_tokens = reference.split()
        shortest = min(len(candidate_tokens), len(reference_tokens))
        width = 2 if shortest < 8 else 3
        candidate_ngrams = {
            tuple(candidate_tokens[index:index + width])
            for index in range(len(candidate_tokens) - width + 1)
        }
        reference_ngrams = {
            tuple(reference_tokens[index:index + width])
            for index in range(len(reference_tokens) - width + 1)
        }
        if not candidate_ngrams or not reference_ngrams:
            return 0.0
        return len(candidate_ngrams & reference_ngrams) / len(
            candidate_ngrams | reference_ngrams
        )
