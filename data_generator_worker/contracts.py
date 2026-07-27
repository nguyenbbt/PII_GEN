from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class GenerationTask:
    task_id: str
    language: str
    focus_labels: list[str]
    difficulty: str
    sample_type: str
    max_entities: int
    run_id: str | None = None
    focus_label: str | None = None
    robin_labels: list[str] = field(default_factory=list)
    optional_constraints: list[str] = field(default_factory=list)
    max_attempts: int = 3
    diversity_profile: dict[str, Any] = field(default_factory=dict)
    length_target: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationTask":
        required = ("task_id", "focus_labels", "difficulty", "sample_type", "max_entities")
        if any(key not in value for key in required):
            raise ValueError("task is missing a required field")
        if not value["focus_labels"] or value["difficulty"] not in {"easy", "medium", "hard"}:
            raise ValueError("task has invalid focus_labels or difficulty")
        if value["sample_type"] not in {"positive", "pure_negative", "hard_negative"} or int(value["max_entities"]) < 1:
            raise ValueError("task has invalid sample_type or max_entities")
        return cls(
            task_id=str(value["task_id"]), language=str(value.get("language", "vi")),
            focus_labels=[str(v) for v in value["focus_labels"]], difficulty=str(value["difficulty"]),
            sample_type=str(value["sample_type"]), max_entities=int(value["max_entities"]), run_id=value.get("run_id"),
            focus_label=str(value["focus_label"]) if value.get("focus_label") else None,
            robin_labels=[str(v) for v in value.get("robin_labels", [])],
            optional_constraints=[str(v) for v in value.get("optional_constraints", [])], max_attempts=int(value.get("max_attempts", 3)),
            diversity_profile=dict(value.get("diversity_profile", {})),
            length_target=dict(value.get("length_target", {})),
        )


@dataclass(frozen=True)
class PositiveEntitySeed:
    label: str
    value: str
    semantic_role: str
    format_variant: str = "default"


@dataclass(frozen=True)
class DecoySeed:
    strategy_id: str
    target_label: str
    value: str
    family: str
    semantic_type: str
    negative_labels: list[str]
    required_context_cues: list[str]
    possible_collision_labels: list[str] = field(default_factory=list)
    forbidden_context_cues: list[str] = field(default_factory=list)
    must_remain_untagged: bool = True


@dataclass(frozen=True)
class ContentSeeds:
    domain: str
    document_type: str
    tone: str
    generic_roles: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContextFrame:
    frame_id: str
    domain: str
    document_type: str
    tone: str
    max_sentences: int
    supported_labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SeedPack:
    seed_pack_id: str
    task_id: str
    sample_type: str
    context_frame: ContextFrame
    hard_negative_mode: str | None = None
    positive_entities: list[PositiveEntitySeed] = field(default_factory=list)
    decoys: list[DecoySeed] = field(default_factory=list)
    content_seeds: ContentSeeds | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SeedPack":
        required = ("seed_pack_id", "task_id", "sample_type", "context_frame")
        if any(key not in value for key in required):
            raise ValueError("seed_pack is missing a required field")
        content = value.get("content_seeds")
        return cls(
            seed_pack_id=str(value["seed_pack_id"]), task_id=str(value["task_id"]),
            sample_type=str(value["sample_type"]), context_frame=ContextFrame(**value["context_frame"]),
            hard_negative_mode=value.get("hard_negative_mode"),
            positive_entities=[PositiveEntitySeed(**item) for item in value.get("positive_entities", [])],
            decoys=[DecoySeed(**item) for item in value.get("decoys", [])],
            content_seeds=ContentSeeds(**content) if content else None,
        )


@dataclass(frozen=True)
class ReflectionContext:
    previous_tagged_text: str | None = None
    must_not_repeat: list[dict[str, str]] = field(default_factory=list)
    mandatory_repairs: list[str] = field(default_factory=list)
    few_shot_examples: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DataGenerationRequest:
    task: GenerationTask
    seed_pack: SeedPack
    attempt_no: int
    taxonomy_context: dict[str, Any] = field(default_factory=dict)
    dynamic_prompt_additions: list[str] = field(default_factory=list)
    reflection: ReflectionContext | None = None
    event_id: str | None = None
    correlation_id: str | None = None
    causation_id: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DataGenerationRequest":
        task = GenerationTask.from_dict(value["task"])
        attempt_no = int(value["attempt_no"])
        if attempt_no < 1 or attempt_no > task.max_attempts:
            raise ValueError("attempt_no must be between 1 and task.max_attempts")
        reflection = value.get("reflection")
        return cls(
            task=task, seed_pack=SeedPack.from_dict(value["seed_pack"]),
            attempt_no=attempt_no, taxonomy_context=value.get("taxonomy_context", {}),
            dynamic_prompt_additions=value.get("dynamic_prompt_additions", []),
            reflection=ReflectionContext(**reflection) if reflection else None, event_id=value.get("event_id"),
            correlation_id=value.get("correlation_id"), causation_id=value.get("causation_id"),
        )


@dataclass(frozen=True)
class GenerationQuery:
    language: str
    constraints: list[str]
    focus_labels: list[str]
    difficulty: str
    sample_type: str
    max_entities: int
    length_target: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMTokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    money_cost: str


@dataclass(frozen=True)
class GeneratedEntity:
    label: str
    value: str


@dataclass(frozen=True)
class DataGenerationResult:
    task_id: str
    attempt_no: int
    seed_pack_id: str
    context_frame_id: str
    generation_query: GenerationQuery
    entities: list[GeneratedEntity]
    tagged_text: str
    token_usage: LLMTokenUsage
    latency_ms: int
    model: str
    prompt_version: str
    output_hash: str
    taxonomy_context_used: dict[str, Any] = field(default_factory=dict)
    diversity_profile: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EventEnvelope:
    idempotency_key: str
    payload: DataGenerationResult
    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: str = "data.generated"
    event_version: int = 1
    occurred_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: str | None = None
    causation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
