from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from math import isclose
import re
from typing import Any, Dict, List, Literal, Optional, Sequence
from uuid import uuid4

from pydantic import BaseModel, Field, root_validator, validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Schema(BaseModel):
    """Base Pydantic v1 schema used by all message and persistence contracts."""

    class Config:
        extra = "forbid"
        validate_assignment = True
        anystr_strip_whitespace = True
        use_enum_values = True


class RunStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskStatus(str, Enum):
    CREATED = "CREATED"
    GENERATING = "GENERATING"
    GENERATED = "GENERATED"
    VALIDATING = "VALIDATING"
    VERIFYING = "VERIFYING"
    REPAIRING = "REPAIRING"
    FORMATTING = "FORMATTING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class SampleType(str, Enum):
    POSITIVE = "positive"
    PURE_NEGATIVE = "pure_negative"
    HARD_NEGATIVE = "hard_negative"


class FewShotExample(Schema):
    id: str = Field(..., min_length=1, max_length=160)
    expected_tagged_text: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)


class TaxonomyExamples(Schema):
    positive: List[FewShotExample] = Field(default_factory=list)
    pure_negative: List[FewShotExample] = Field(default_factory=list)
    hard_negative: List[FewShotExample] = Field(default_factory=list)


class TaxonomyLabel(Schema):
    code: str = Field(..., min_length=1, max_length=100)
    definition: str = Field(..., min_length=1)
    rules: List[str] = Field(default_factory=list)
    examples: TaxonomyExamples = Field(default_factory=TaxonomyExamples)


class TaxonomySnapshot(Schema):
    version_id: str = Field(default_factory=lambda: str(uuid4()))
    labels: List[TaxonomyLabel] = Field(..., min_items=1)
    created_at: datetime = Field(default_factory=utc_now)

    @validator("labels")
    def label_codes_are_unique(cls, labels: List[TaxonomyLabel]) -> List[TaxonomyLabel]:
        codes = [label.code for label in labels]
        if len(codes) != len(set(codes)):
            raise ValueError("taxonomy label codes must be unique")
        return labels


class LabelGenerationContext(Schema):
    label: str = Field(..., min_length=1, max_length=100)
    definition: str = Field(..., min_length=1)
    rule: str = ""
    examples: List[FewShotExample] = Field(default_factory=list)


class GenerationTaxonomyContext(Schema):
    taxonomy_version_id: str = Field(..., min_length=1)
    sample_type: SampleType
    focus_label: LabelGenerationContext
    robin_labels: List[LabelGenerationContext] = Field(default_factory=list)
    available_labels: List[LabelGenerationContext] = Field(default_factory=list)


class ValueBankConfig(Schema):
    path: str = Field(default="PII_Value_Bank", min_length=1)
    language_files: Dict[str, str] = Field(default_factory=lambda: {
        "vi": "vi_pii_value_pools.json",
        "en": "en_pii_value_pools.json",
        "de": "de_pii_value_pools.json",
    })
    max_seed_pack_attempts: int = Field(default=5, ge=1, le=20)
    allow_additional_unseeded_pii: bool = False
    partition_index: int = Field(default=0, ge=0, le=99_999)
    partition_count: int = Field(default=1, ge=1, le=100_000)

    @root_validator
    def partition_index_is_in_range(
        cls,
        values: Dict[str, Any],
    ) -> Dict[str, Any]:
        if values.get("partition_index", 0) >= values.get(
            "partition_count",
            1,
        ):
            raise ValueError(
                "value_bank.partition_index must be smaller than partition_count"
            )
        return values

    @validator("language_files")
    def validate_language_files(
        cls,
        value: Dict[str, str],
    ) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        for raw_language, raw_path in value.items():
            language = str(raw_language).strip().casefold()
            file_path = str(raw_path).strip()
            if not re.fullmatch(r"[a-z]{2}", language):
                raise ValueError(
                    "value_bank.language_files keys must be two-letter "
                    "language codes"
                )
            if not file_path:
                raise ValueError(
                    "value_bank.language_files paths cannot be empty"
                )
            normalized[language] = file_path
        return normalized


class HardNegativeConfig(Schema):
    mode: Literal["decoy_only", "mixed_contrastive"] = "decoy_only"
    min_decoys: int = Field(default=1, ge=1, le=3)
    max_decoys: int = Field(default=1, ge=1, le=3)
    max_focus_labels: int = Field(default=1, ge=1, le=10)
    unsupported_label_policy: Literal["rebuild_task"] = "rebuild_task"

    @root_validator
    def min_does_not_exceed_max(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("min_decoys", 1) > values.get("max_decoys", 1):
            raise ValueError("hard_negative.min_decoys cannot exceed max_decoys")
        return values


class RobinSelectionConfig(Schema):
    min_per_sample: int = Field(default=0, ge=0, le=9)
    max_per_sample: int = Field(default=2, ge=0, le=9)
    minimum_per_label: int = Field(default=0, ge=0)

    @root_validator
    def min_does_not_exceed_max(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("min_per_sample", 0) > values.get("max_per_sample", 2):
            raise ValueError("robin_selection.min_per_sample cannot exceed max_per_sample")
        return values


class ComplexityLimits(Schema):
    positive: int = Field(default=4, ge=1, le=20)
    pure_negative: int = Field(default=2, ge=0, le=20)
    hard_negative: int = Field(default=4, ge=2, le=20)


class SampleStructureConfig(Schema):
    type: Literal["contract", "chat", "custom"] = "contract"
    custom_instruction: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=1_000,
    )

    @root_validator
    def instruction_matches_structure_type(
        cls,
        values: Dict[str, Any],
    ) -> Dict[str, Any]:
        structure_type = values.get("type")
        custom_instruction = values.get("custom_instruction")
        if structure_type == "custom" and not custom_instruction:
            raise ValueError("custom_instruction is required when type is custom")
        if structure_type != "custom" and custom_instruction is not None:
            raise ValueError("custom_instruction is only allowed when type is custom")
        return values


class ValidationConfig(Schema):
    quality_checks_enabled: bool = False
    accept_last_candidate_on_exhaustion: bool = False
    local_context_window: int = Field(default=80, ge=20, le=500)
    reject_mixed_locale: bool = True
    pure_negative_structured_scan: bool = True
    hard_negative_structured_scan: bool = True
    novelty_mode: Literal["off", "audit", "enforce"] = "audit"
    near_duplicate_threshold: float = Field(default=0.9, ge=0.5, le=1.0)
    novelty_recent_window: int = Field(default=100, ge=1, le=10_000)
    max_novelty_feedback: int = Field(default=3, ge=0, le=10)


class NoveltyConfig(Schema):
    enabled: bool = False
    mode: Literal["off", "audit", "enforce"] = "audit"
    near_duplicate_threshold: float = Field(default=0.9, ge=0.5, le=1.0)
    recent_window: int = Field(default=100, ge=1, le=10_000)
    max_feedback: int = Field(default=3, ge=0, le=10)


class VerifierConfig(Schema):
    enabled: bool = False
    max_repairs_per_candidate: int = Field(default=1, ge=0, le=2)


class ParallelGenerationConfig(Schema):
    workers: int = Field(default=1, ge=1, le=16)
    shard_size: int = Field(default=10, ge=1, le=100)
    max_shard_retries: int = Field(default=1, ge=0, le=5)


class RunConfig(Schema):
    run_name: str = Field(default="pii-run", min_length=1, max_length=120)
    num_samples: int = Field(..., gt=0, le=100_000)
    language: str = Field(default="vi", min_length=2, max_length=20)
    minimum_per_label: int = Field(default=0, ge=0)
    batch_size: int = Field(default=5, gt=0, le=10_000)
    focus_labels: Optional[List[str]] = None
    focus_label: Optional[str] = None
    robin_labels: List[str] = Field(default_factory=list)
    robin_selection: RobinSelectionConfig = Field(default_factory=RobinSelectionConfig)
    difficulty_distribution: Dict[str, float] = Field(default_factory=lambda: {"easy": 0.3, "medium": 0.4, "hard": 0.3})
    sample_type_distribution: Dict[str, float] = Field(default_factory=lambda: {"positive": 0.8, "pure_negative": 0.05, "hard_negative": 0.15})
    sample_length_distribution: Dict[str, float] = Field(
        default_factory=lambda: {"short": 1 / 3, "medium": 1 / 3, "long": 1 / 3}
    )
    # Deprecated compatibility field. New configs should define
    # ``sample_structures``; the legacy single value is copied into that pool.
    sample_structure: SampleStructureConfig = Field(default_factory=SampleStructureConfig)
    sample_structures: List[SampleStructureConfig] = Field(default_factory=list)
    optional_constraint_distribution: Dict[str, float] = Field(default_factory=dict)
    max_entities: Dict[str, int] = Field(default_factory=lambda: {"easy": 2, "medium": 4, "hard": 6})
    max_regenerate_attempts: int = Field(default=2, ge=0, le=4)
    max_task_replacements: int = Field(default=2, ge=0, le=10)
    random_seed: int = 42
    value_bank: ValueBankConfig = Field(default_factory=ValueBankConfig)
    hard_negative: HardNegativeConfig = Field(default_factory=HardNegativeConfig)
    complexity_limits: ComplexityLimits = Field(default_factory=ComplexityLimits)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    novelty: NoveltyConfig = Field(default_factory=NoveltyConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    parallel_generation: ParallelGenerationConfig = Field(
        default_factory=ParallelGenerationConfig
    )

    @root_validator(pre=True)
    def support_legacy_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values = dict(values)
        if "sample_structures" in values and not values["sample_structures"]:
            raise ValueError("sample_structures cannot be empty")
        if "sample_structures" not in values:
            values["sample_structures"] = [
                values.get("sample_structure") or {"type": "contract"}
            ]
        if "target_samples" in values and "num_samples" not in values:
            values["num_samples"] = values.pop("target_samples")
        if "difficulty" in values and "difficulty_distribution" not in values:
            raw_difficulty = values.pop("difficulty")
            difficulty = str(getattr(raw_difficulty, "value", raw_difficulty))
            values["difficulty_distribution"] = {name: float(name == difficulty) for name in ("easy", "medium", "hard")}
        if "sample_type" in values and "sample_type_distribution" not in values:
            raw_sample_type = values.pop("sample_type")
            sample_type = str(getattr(raw_sample_type, "value", raw_sample_type))
            values["sample_type_distribution"] = {name: float(name == sample_type) for name in ("positive", "pure_negative", "hard_negative")}
        if "optional_constraints" in values and "optional_constraint_distribution" not in values:
            values["optional_constraint_distribution"] = {str(name): 1.0 for name in values.pop("optional_constraints")}
        if isinstance(values.get("max_entities"), int):
            count = values["max_entities"]
            values["max_entities"] = {name: count for name in ("easy", "medium", "hard")}
        if "max_attempts" in values and "max_regenerate_attempts" not in values:
            values["max_regenerate_attempts"] = max(0, int(values.pop("max_attempts")) - 1)
        legacy_seed_config = None
        if "faker" in values and "value_bank" not in values:
            legacy_seed_config = values.pop("faker")
        elif "seed_generation" in values and "value_bank" not in values:
            legacy_seed_config = values.pop("seed_generation")
        if legacy_seed_config is not None:
            if isinstance(legacy_seed_config, BaseModel):
                legacy_seed_config = legacy_seed_config.dict()
            if isinstance(legacy_seed_config, dict):
                legacy_seed_config = {
                    key: value
                    for key, value in legacy_seed_config.items()
                    if key != "locale"
                }
            values["value_bank"] = legacy_seed_config
        hard_negative = values.get("hard_negative")
        if isinstance(hard_negative, dict) and hard_negative.get("unsupported_label_policy") == "auxiliary_date":
            # Legacy configs used DATE as a fallback for unsupported decoys. All
            # taxonomy labels now have their own strategy, so the safe equivalent
            # is to rebuild the task without injecting an unrelated label.
            values["hard_negative"] = {
                **hard_negative,
                "unsupported_label_policy": "rebuild_task",
            }
        validation = values.get("validation")
        if isinstance(validation, dict) and "quality_checks_enabled" in validation:
            if "novelty" not in values:
                values["novelty"] = {
                    "enabled": bool(validation["quality_checks_enabled"]),
                    "mode": validation.get("novelty_mode", "audit"),
                    "near_duplicate_threshold": validation.get(
                        "near_duplicate_threshold",
                        0.9,
                    ),
                    "recent_window": validation.get(
                        "novelty_recent_window",
                        100,
                    ),
                    "max_feedback": validation.get(
                        "max_novelty_feedback",
                        3,
                    ),
                }
            verifier = dict(values.get("verifier") or {})
            verifier.setdefault(
                "enabled",
                bool(validation["quality_checks_enabled"]),
            )
            values["verifier"] = verifier
        return values

    @validator("language")
    def normalise_language(cls, language: str) -> str:
        aliases = {
            "vietnamese": "vi",
            "vi-vn": "vi",
            "vi_vn": "vi",
            "english": "en",
            "en-us": "en",
            "en_us": "en",
            "german": "de",
            "de-de": "de",
            "de_de": "de",
        }
        return aliases.get(language.casefold(), language)

    @validator("focus_labels")
    def normalise_focus_labels(cls, labels: Optional[List[str]]) -> Optional[List[str]]:
        if labels is None:
            return None
        normalised = list(dict.fromkeys(
            label.strip().upper()
            for label in labels
            if label.strip()
        ))
        if not normalised:
            raise ValueError("focus_labels cannot be empty; omit it to use the full taxonomy")
        return normalised

    @validator("focus_label")
    def normalise_focus_label(cls, label: Optional[str]) -> Optional[str]:
        return label.strip().upper() if label else None

    @validator("robin_labels")
    def normalise_robin_labels(cls, labels: List[str]) -> List[str]:
        return list(dict.fromkeys(label.strip().upper() for label in labels if label.strip()))

    @validator("difficulty_distribution")
    def validate_difficulty_distribution(cls, distribution: Dict[str, float]) -> Dict[str, float]:
        return cls._validate_distribution(distribution, {"easy", "medium", "hard"}, "difficulty_distribution")

    @validator("sample_type_distribution")
    def validate_sample_type_distribution(cls, distribution: Dict[str, float]) -> Dict[str, float]:
        return cls._validate_distribution(distribution, {"positive", "pure_negative", "hard_negative"}, "sample_type_distribution")

    @validator("sample_length_distribution")
    def validate_sample_length_distribution(
        cls,
        distribution: Dict[str, float],
    ) -> Dict[str, float]:
        return cls._validate_distribution(
            distribution,
            {"short", "medium", "long"},
            "sample_length_distribution",
        )

    @validator("optional_constraint_distribution")
    def validate_optional_distribution(cls, distribution: Dict[str, float]) -> Dict[str, float]:
        invalid = {name: probability for name, probability in distribution.items() if not 0 <= probability <= 1}
        if invalid:
            raise ValueError(f"optional constraint probabilities must be between 0 and 1: {invalid}")
        return distribution

    @validator("max_entities")
    def validate_max_entities(cls, limits: Dict[str, int]) -> Dict[str, int]:
        if set(limits) != {"easy", "medium", "hard"}:
            raise ValueError("max_entities must define easy, medium, and hard")
        if any(not 1 <= limit <= 10 for limit in limits.values()):
            raise ValueError("each max_entities value must be between 1 and 10")
        return limits

    @root_validator
    def validate_focus_and_robin_mode(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        focus_label = values.get("focus_label")
        robin_labels = values.get("robin_labels") or []
        if not focus_label:
            if robin_labels:
                raise ValueError("robin_labels requires focus_label")
            return values
        if values.get("minimum_per_label", 0) > 0:
            raise ValueError(
                "minimum_per_label is only supported with focus_labels; "
                "use robin_selection.minimum_per_label in anchor mode"
            )
        if values.get("focus_labels") is not None:
            raise ValueError("use either focus_label/robin_labels or legacy focus_labels, not both")
        if focus_label in robin_labels:
            raise ValueError("focus_label cannot also appear in robin_labels")

        selection: RobinSelectionConfig = values.get("robin_selection") or RobinSelectionConfig()
        if selection.min_per_sample > len(robin_labels):
            raise ValueError("robin_selection.min_per_sample exceeds the robin_labels pool")
        num_samples = values.get("num_samples") or 0
        if selection.minimum_per_label > num_samples:
            raise ValueError(
                "robin_selection.minimum_per_label cannot exceed num_samples"
            )
        required_robin_occurrences = (
            selection.minimum_per_label * len(robin_labels)
        )
        available_robin_occurrences = num_samples * selection.max_per_sample
        if required_robin_occurrences > available_robin_occurrences:
            raise ValueError(
                "robin_selection.minimum_per_label is not feasible for "
                "num_samples and max_per_sample"
            )

        sample_types = values.get("sample_type_distribution") or {}
        if sample_types.get("pure_negative", 0) > 0:
            raise ValueError("sample_type_distribution.pure_negative must be 0 when focus_label is enabled")
        hard_negative: HardNegativeConfig = values.get("hard_negative") or HardNegativeConfig()
        if sample_types.get("hard_negative", 0) > 0 and hard_negative.mode != "mixed_contrastive":
            raise ValueError("focus_label hard-negative samples require hard_negative.mode=mixed_contrastive")

        max_entities = values.get("max_entities") or {}
        complexity: ComplexityLimits = values.get("complexity_limits") or ComplexityLimits()
        minimum_capacity = 1 + selection.min_per_sample
        capacities: List[int] = []
        for difficulty, probability in (values.get("difficulty_distribution") or {}).items():
            if probability <= 0:
                continue
            entity_limit = max_entities.get(difficulty, 1)
            if sample_types.get("positive", 0) > 0:
                capacities.append(min(entity_limit, complexity.positive))
            if sample_types.get("hard_negative", 0) > 0:
                capacities.append(min(
                    entity_limit,
                    hard_negative.max_focus_labels,
                    max(1, complexity.hard_negative - hard_negative.min_decoys),
                ))
        if capacities and min(capacities) < minimum_capacity:
            raise ValueError(
                "focus_label plus robin_selection.min_per_sample exceeds an active task capacity"
            )
        maximum_capacity = 1 + selection.max_per_sample
        if capacities and min(capacities) < maximum_capacity:
            raise ValueError(
                "focus_label plus robin_selection.max_per_sample exceeds an active task capacity"
            )
        return values

    @staticmethod
    def _validate_distribution(distribution: Dict[str, float], expected: set[str], field_name: str) -> Dict[str, float]:
        if set(distribution) != expected:
            raise ValueError(f"{field_name} must define exactly {sorted(expected)}")
        if any(probability < 0 for probability in distribution.values()):
            raise ValueError(f"{field_name} cannot contain negative probabilities")
        if not isclose(sum(distribution.values()), 1.0, abs_tol=1e-9):
            raise ValueError(f"{field_name} probabilities must sum to 1.0")
        return distribution

    @property
    def target_samples(self) -> int:
        return self.num_samples

    @property
    def max_attempts(self) -> int:
        return self.max_regenerate_attempts + 1

    @property
    def label_pool(self) -> Optional[List[str]]:
        if self.focus_label:
            return [self.focus_label, *self.robin_labels]
        return self.focus_labels


class CreateRunRequest(Schema):
    run_name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    config: RunConfig
    taxonomy_version_id: Optional[str] = None
    taxonomy: Optional[TaxonomySnapshot] = None

    @root_validator
    def use_exactly_one_taxonomy_source(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        has_version = bool(values.get("taxonomy_version_id"))
        has_inline = values.get("taxonomy") is not None
        if has_version == has_inline:
            raise ValueError("provide exactly one of taxonomy_version_id or taxonomy")
        return values


class Run(Schema):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    status: RunStatus = RunStatus.CREATED
    config: RunConfig
    taxonomy_version_id: str
    output_path: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class DiversityProfile(Schema):
    context_frame_id: Optional[str] = None
    speaker_role: str = "participant"
    intent: str = "provide_information"
    document_structure: str = "single_paragraph"
    language_register: str = "neutral"
    length_bucket: Literal["short", "medium", "long"] = "medium"
    entity_format_variants: Dict[str, str] = Field(default_factory=dict)


class LengthTarget(Schema):
    bucket: Literal["short", "medium", "long"]
    min_words: int = Field(..., ge=1)
    max_words: int = Field(..., ge=1)
    unit: Literal["content_units", "turns", "words"]
    min_units: int = Field(..., ge=1)
    max_units: int = Field(..., ge=1)

    @root_validator
    def minimums_do_not_exceed_maximums(
        cls,
        values: Dict[str, Any],
    ) -> Dict[str, Any]:
        if values.get("min_words", 1) > values.get("max_words", 1):
            raise ValueError("length target min_words cannot exceed max_words")
        if values.get("min_units", 1) > values.get("max_units", 1):
            raise ValueError("length target min_units cannot exceed max_units")
        return values


class GenerationTask(Schema):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    sequence_no: int = Field(..., gt=0)
    language: str
    focus_labels: List[str] = Field(..., min_items=1)
    annotation_labels: List[str] = Field(default_factory=list)
    focus_label: Optional[str] = None
    robin_labels: List[str] = Field(default_factory=list)
    difficulty: Difficulty
    sample_type: SampleType
    sample_structure: SampleStructureConfig = Field(
        default_factory=SampleStructureConfig
    )
    optional_constraints: List[str] = Field(default_factory=list)
    max_entities: int = Field(..., gt=0)
    max_attempts: int = Field(..., gt=0)
    random_seed: int
    diversity_profile: DiversityProfile = Field(default_factory=DiversityProfile)
    length_target: LengthTarget = Field(
        default_factory=lambda: LengthTarget(
            bucket="medium",
            min_words=150,
            max_words=230,
            unit="content_units",
            min_units=6,
            max_units=9,
        )
    )
    current_attempt: int = 0
    slot_no: Optional[int] = Field(default=None, gt=0)
    replacement_no: int = Field(default=0, ge=0)
    parent_task_id: Optional[str] = None
    status: TaskStatus = TaskStatus.CREATED


class PositiveEntitySeed(Schema):
    label: str = Field(..., min_length=1, max_length=100)
    value: str = Field(..., min_length=1)
    semantic_role: str = Field(..., min_length=1)
    format_variant: str = "default"


class DecoySeed(Schema):
    strategy_id: str = Field(..., min_length=1)
    target_label: str = Field(..., min_length=1, max_length=100)
    value: str = Field(..., min_length=1)
    family: str = Field(..., min_length=1)
    semantic_type: str = Field(..., min_length=1)
    negative_labels: List[str] = Field(..., min_items=1)
    possible_collision_labels: List[str] = Field(default_factory=list)
    required_context_cues: List[str] = Field(..., min_items=1)
    forbidden_context_cues: List[str] = Field(default_factory=list)
    must_remain_untagged: bool = True


class ContentSeeds(Schema):
    domain: str
    document_type: str
    tone: str
    generic_roles: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    objects: List[str] = Field(default_factory=list)


class ContextFrame(Schema):
    frame_id: str
    domain: str
    document_type: str
    tone: str
    max_sentences: int = Field(..., ge=1, le=8)
    supported_labels: List[str]


class SeedPack(Schema):
    seed_pack_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    sample_type: SampleType
    hard_negative_mode: Optional[Literal["decoy_only", "mixed_contrastive"]] = None
    positive_entities: List[PositiveEntitySeed] = Field(default_factory=list)
    decoys: List[DecoySeed] = Field(default_factory=list)
    content_seeds: Optional[ContentSeeds] = None
    context_frame: ContextFrame


class ValidationIssue(Schema):
    type: str
    scope: Literal["TEXT", "SEEDS", "CONTEXT"]
    reason: str
    label: Optional[str] = None
    value: Optional[str] = None


class DeterministicValidationResult(Schema):
    valid: bool
    issues: List[ValidationIssue] = Field(default_factory=list)


class NoveltyAssessment(Schema):
    valid: bool = True
    sentence_skeleton: str = ""
    max_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    nearest_skeletons: List[str] = Field(default_factory=list)
    issues: List[ValidationIssue] = Field(default_factory=list)


class ReflectionContext(Schema):
    previous_tagged_text: Optional[str] = None
    must_not_repeat: List[Dict[str, str]] = Field(default_factory=list)
    mandatory_repairs: List[str] = Field(default_factory=list)
    few_shot_examples: List[str] = Field(default_factory=list)


class DataGenerationRequest(Schema):
    task: GenerationTask
    seed_pack: SeedPack
    seed_validation: DeterministicValidationResult
    taxonomy_context: GenerationTaxonomyContext
    attempt_no: int = Field(..., gt=0)
    reflection: Optional[ReflectionContext] = None
    dynamic_prompt_additions: List[str] = Field(default_factory=list)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None

    @validator("attempt_no")
    def attempt_is_allowed(cls, attempt_no: int, values: Dict[str, Any]) -> int:
        task: Optional[GenerationTask] = values.get("task")
        if task and attempt_no > task.max_attempts:
            raise ValueError("attempt_no exceeds task.max_attempts")
        return attempt_no


class TokenUsage(Schema):
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    money_cost: Decimal = Field(..., ge=0)

    @classmethod
    def zero(cls) -> "TokenUsage":
        return cls(input_tokens=0, output_tokens=0, total_tokens=0, money_cost=Decimal("0"))

    @classmethod
    def combine(cls, calls: Sequence["TokenUsage"]) -> "TokenUsage":
        return cls(
            input_tokens=sum(call.input_tokens for call in calls),
            output_tokens=sum(call.output_tokens for call in calls),
            total_tokens=sum(call.total_tokens for call in calls),
            money_cost=sum((call.money_cost for call in calls), Decimal("0")),
        )


class GenerationQuery(Schema):
    language: str
    focus_labels: List[str]
    focus_label: Optional[str] = None
    robin_labels: List[str] = Field(default_factory=list)
    constraints: List[str]
    difficulty: Difficulty
    sample_type: SampleType
    sample_structure: SampleStructureConfig = Field(
        default_factory=SampleStructureConfig
    )
    length_target: LengthTarget = Field(
        default_factory=lambda: LengthTarget(
            bucket="medium",
            min_words=150,
            max_words=230,
            unit="content_units",
            min_units=6,
            max_units=9,
        )
    )
    max_entities: int


class GeneratedEntity(Schema):
    """One code-inserted PII value present in ``tagged_text``."""

    label: str = Field(..., min_length=1, max_length=100)
    value: str = Field(..., min_length=1)


class GenerationCandidate(Schema):
    task_id: str
    attempt_no: int = Field(..., gt=0)
    seed_pack_id: str
    context_frame_id: str
    generation_query: GenerationQuery
    entities: List[GeneratedEntity]
    tagged_text: str = Field(..., min_length=1)
    token_usage: TokenUsage
    latency_ms: int = Field(..., ge=0)
    model: str
    prompt_version: str
    output_hash: str
    seed_validation: DeterministicValidationResult
    diversity_profile: DiversityProfile = Field(default_factory=DiversityProfile)
    taxonomy_context_used: Optional[GenerationTaxonomyContext] = None
    created_at: datetime = Field(default_factory=utc_now)


class VerificationIssue(Schema):
    type: str = Field(..., min_length=1, max_length=100)
    severity: Literal["low", "medium", "high", "critical"]
    field: str = Field(..., min_length=1, max_length=100)
    reason: str = Field(..., min_length=1)
    suggested_fix: str = Field(
        default="Apply the routing decision using the issue reason.",
        min_length=1,
    )

    @validator("severity", pre=True)
    def normalize_severity_case(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @validator("suggested_fix", pre=True, always=True)
    def normalize_missing_suggested_fix(cls, value: Any, values: Dict[str, Any]) -> str:
        if isinstance(value, str) and value.strip():
            return value
        severity = str(values.get("severity") or "").strip().lower()
        if severity == "low":
            return "Apply the smallest safe local correction described by the issue reason."
        if severity == "critical":
            return "Reject the sample and do not preserve the unsafe content."
        return "Regenerate the sample so the reported issue no longer occurs."


class VerificationEditSegment(Schema):
    label: str = Field(..., min_length=1, max_length=100)
    value: str = Field(..., min_length=1)


class VerificationEdit(Schema):
    action: Literal[
        "add_tag",
        "split_tag",
        "adjust_tag_boundary",
        "remove_tag",
        "replace_template_artifact",
        "sync_entities",
    ]
    reason: str = Field(..., min_length=1)
    label: Optional[str] = Field(default=None, min_length=1, max_length=100)
    value: Optional[str] = Field(default=None, min_length=1)
    occurrence: Optional[int] = Field(default=None, ge=1)
    source_label: Optional[str] = Field(default=None, min_length=1, max_length=100)
    source_value: Optional[str] = Field(default=None, min_length=1)
    replacement: Optional[str] = None
    segments: List[VerificationEditSegment] = Field(default_factory=list)

    @validator("action", pre=True)
    def normalize_action_case(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @validator(
        "label",
        "value",
        "source_label",
        "source_value",
        pre=True,
    )
    def normalize_blank_optional_field(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @validator("occurrence", pre=True)
    def normalize_zero_based_occurrence(cls, value: Any) -> Any:
        """Treat the common LLM index ``0`` as the first occurrence."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if isinstance(value, bool):
            return value
        if value == 0 or (isinstance(value, str) and value.strip() == "0"):
            return 1
        return value

    @validator("segments", pre=True)
    def normalize_null_segments(cls, value: Any) -> Any:
        return [] if value is None else value


class VerifierDecision(Schema):
    status: Literal["PASS", "FIXABLE", "REGENERATE", "REJECTED"]
    score: int = Field(..., ge=0, le=100)
    issues: List[VerificationIssue] = Field(default_factory=list, max_items=20)
    edits: List[VerificationEdit] = Field(default_factory=list, max_items=20)
    token_usage: TokenUsage
    latency_ms: int = Field(..., ge=0)
    model: str
    prompt_version: str

    @validator("status", pre=True)
    def normalize_status_case(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @validator("issues", "edits", pre=True)
    def normalize_null_collections(cls, value: Any) -> Any:
        return [] if value is None else value

    @root_validator(skip_on_failure=True)
    def status_matches_issues(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        status = values.get("status")
        issues: List[VerificationIssue] = values.get("issues") or []
        edits: List[VerificationEdit] = values.get("edits") or []
        if status == "PASS" and issues:
            raise ValueError("PASS verifier decisions cannot contain issues")
        if status == "PASS" and edits:
            raise ValueError("PASS verifier decisions cannot contain edits")
        if status != "PASS" and not issues:
            raise ValueError(f"{status} verifier decisions require at least one issue")
        if status == "FIXABLE" and any(issue.severity != "low" for issue in issues):
            raise ValueError("FIXABLE verifier decisions may contain only low severity issues")
        if status == "REJECTED" and not any(issue.severity == "critical" for issue in issues):
            raise ValueError("REJECTED verifier decisions require a critical issue")
        return values


class RepairResult(Schema):
    tagged_text: str = Field(..., min_length=1)
    entities: List[GeneratedEntity]
    token_usage: TokenUsage
    latency_ms: int = Field(..., ge=0)
    model: str
    prompt_version: str


class VerificationTrace(Schema):
    initial_judge: VerifierDecision
    repair: Optional[RepairResult] = None
    final_judge: Optional[VerifierDecision] = None
    outcome: Literal["PASS", "FIXED", "REGENERATE", "REJECTED"]


class FormattedEntity(Schema):
    label: str = Field(..., min_length=1, max_length=100)
    start: int = Field(..., ge=0)
    end: int = Field(..., gt=0)
    text: str = Field(..., min_length=1)

    @root_validator
    def end_is_after_start(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("end", 0) <= values.get("start", 0):
            raise ValueError("formatted entity end must be greater than start")
        return values


class FormattedRoleTokenUsage(Schema):
    """Input/output token counts for one LLM role."""

    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)


class FormattedTokenUsage(Schema):
    """Token counts exposed with one accepted sample in the final dataset."""

    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    generator: FormattedRoleTokenUsage = Field(
        default_factory=lambda: FormattedRoleTokenUsage(
            input_tokens=0,
            output_tokens=0,
        )
    )
    verifier: FormattedRoleTokenUsage = Field(
        default_factory=lambda: FormattedRoleTokenUsage(
            input_tokens=0,
            output_tokens=0,
        )
    )


class FormattedSample(Schema):
    entities: List[FormattedEntity] = Field(default_factory=list)
    text: str = Field(..., min_length=1)
    token_usage: Optional[FormattedTokenUsage] = None

    @root_validator
    def spans_match_text(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        text = values.get("text", "")
        entities: List[FormattedEntity] = values.get("entities") or []
        previous_end = 0
        for entity in sorted(entities, key=lambda item: item.start):
            if entity.start < previous_end:
                raise ValueError("formatted entities cannot overlap")
            if text[entity.start:entity.end] != entity.text:
                raise ValueError("formatted entity offsets do not match sample text")
            previous_end = entity.end
        return values


class PipelineTokenUsage(Schema):
    generator: TokenUsage = Field(default_factory=TokenUsage.zero)
    verifier_judge: TokenUsage = Field(default_factory=TokenUsage.zero)
    verifier_repair: TokenUsage = Field(default_factory=TokenUsage.zero)
    verifier_rejudge: TokenUsage = Field(default_factory=TokenUsage.zero)
    total: TokenUsage = Field(default_factory=TokenUsage.zero)

    def verifier_total(self) -> TokenUsage:
        """Combine Judge, Repair, and re-Judge calls for the verifier model."""

        return TokenUsage.combine(
            (
                self.verifier_judge,
                self.verifier_repair,
                self.verifier_rejudge,
            )
        )

    @classmethod
    def from_calls(
        cls,
        *,
        generator: Sequence[TokenUsage] = (),
        verifier_judge: Sequence[TokenUsage] = (),
        verifier_repair: Sequence[TokenUsage] = (),
        verifier_rejudge: Sequence[TokenUsage] = (),
    ) -> "PipelineTokenUsage":
        generator_total = TokenUsage.combine(generator)
        judge_total = TokenUsage.combine(verifier_judge)
        repair_total = TokenUsage.combine(verifier_repair)
        rejudge_total = TokenUsage.combine(verifier_rejudge)
        return cls(
            generator=generator_total,
            verifier_judge=judge_total,
            verifier_repair=repair_total,
            verifier_rejudge=rejudge_total,
            total=TokenUsage.combine(
                (generator_total, judge_total, repair_total, rejudge_total)
            ),
        )


class DataGenerationResult(Schema):
    task_id: str
    attempt_no: int
    seed_pack_id: str
    context_frame_id: str
    generation_query: GenerationQuery
    entities: List[GeneratedEntity]
    tagged_text: str = Field(..., min_length=1)
    token_usage: TokenUsage
    latency_ms: int = Field(..., ge=0)
    model: str
    prompt_version: str
    output_hash: str
    seed_validation: DeterministicValidationResult
    output_validation: DeterministicValidationResult
    diversity_profile: DiversityProfile = Field(default_factory=DiversityProfile)
    sentence_skeleton: str = ""
    novelty_assessment: NoveltyAssessment = Field(default_factory=NoveltyAssessment)
    taxonomy_context_used: Optional[GenerationTaxonomyContext] = None
    regeneration_scope: Optional[Literal["TEXT", "SEEDS", "CONTEXT"]] = None
    verification_trace: Optional[VerificationTrace] = None
    formatted_sample: Optional[FormattedSample] = None
    pipeline_token_usage: Optional[PipelineTokenUsage] = None
    created_at: datetime = Field(default_factory=utc_now)


class EventEnvelope(Schema):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    event_version: int = 1
    occurred_at: datetime = Field(default_factory=utc_now)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    idempotency_key: str
    payload: Dict[str, Any]


class RunSummary(Schema):
    run: Run
    tasks_created: int
    tasks_generated: int
    total_cost: Decimal
