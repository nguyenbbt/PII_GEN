from __future__ import annotations

import json
from pathlib import Path
from typing import List

from pydantic import Field, validator

from ..domain.models import (
    FewShotExample,
    Schema,
    TaxonomyExamples,
    TaxonomyLabel,
    TaxonomySnapshot,
)


class _JsonExampleGroups(Schema):
    positive: List[FewShotExample] = Field(
        ...,
        alias="POSITIVE",
        min_items=3,
    )
    pure_negative: List[FewShotExample] = Field(
        ...,
        alias="PURE_NEGATIVE",
        min_items=3,
    )
    hard_negative: List[FewShotExample] = Field(
        ...,
        alias="HARD_NEGATIVE",
        min_items=3,
    )

    @validator("positive", "pure_negative", "hard_negative")
    def example_ids_are_unique(
        cls,
        examples: List[FewShotExample],
    ) -> List[FewShotExample]:
        identifiers = [example.id for example in examples]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("few-shot ids must be unique within each group")
        return examples


class _JsonTaxonomyEntry(Schema):
    code: str = Field(..., alias="LABEL", min_length=1, max_length=100)
    definition: str = Field(..., alias="DEFINITION", min_length=1)
    rule: str = Field(..., alias="RULE", min_length=1)
    examples: _JsonExampleGroups = Field(..., alias="EXAMPLES")

    @validator("code")
    def normalize_code(cls, code: str) -> str:
        normalized = code.strip().upper()
        if normalized != code:
            raise ValueError("LABEL must already be uppercase and trimmed")
        return normalized

    def to_taxonomy_label(self) -> TaxonomyLabel:
        return TaxonomyLabel(
            code=self.code,
            definition=self.definition,
            rules=[self.rule],
            examples=TaxonomyExamples(
                positive=self.examples.positive,
                pure_negative=self.examples.pure_negative,
                hard_negative=self.examples.hard_negative,
            ),
        )


class JsonTaxonomyParser:
    """Parse the canonical JSON rules file into the domain taxonomy schema."""

    def parse_file(self, path: Path) -> TaxonomySnapshot:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, list):
            raise ValueError("taxonomy JSON root must be an array")
        entries = [_JsonTaxonomyEntry.parse_obj(item) for item in raw]
        return TaxonomySnapshot(
            labels=[entry.to_taxonomy_label() for entry in entries]
        )
