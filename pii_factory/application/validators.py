from __future__ import annotations

import ipaddress
import re
from datetime import datetime
from typing import Iterable, Sequence
from urllib.parse import urlparse

from data_generator_worker.validation import extract_occurrence_contexts, validate_generated_output

from ..domain.models import (
    DeterministicValidationResult,
    GeneratedEntity,
    LengthTarget,
    HardNegativeConfig,
    SampleType,
    SeedPack,
    TaxonomyLabel,
    ValidationConfig,
    ValidationIssue,
)
from .entity_variants import VIETNAMESE_ADMINISTRATIVE_AREAS
from .seed_generation import HARD_NEGATIVE_STRATEGIES


_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w-])")
_PHONE = re.compile(r"(?<!\d)(?:\+?84|0)[ .-]?(?:3|5|7|8|9)(?:[ .-]?\d){8}(?!\d)")
_URL = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)
_DATE = re.compile(r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[/.-](?:0?[1-9]|1[0-2])[/.-](?:19|20)\d{2}(?!\d)")
_TIME = re.compile(r"(?<!\d)(?:(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?|(?:0?[1-9]|1[0-2]):[0-5]\d\s?(?:AM|PM))(?!\d)", re.IGNORECASE)
_IP = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_IDENTIFIER = re.compile(r"\b(?:ACC|USR|EMP|INC|BH)-[A-Z0-9]{5,}\b", re.IGNORECASE)
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9_]*>")
_MIXED_LOCALE = re.compile(r"JaneHuyện|JohnQuận|SmithPhường|\b(?:County|Street|Avenue|undefined|null|N/A|xxx)\b", re.IGNORECASE)


def extract_local_context(text: str, value: str, window: int = 80) -> str:
    clean = _TAG.sub("", text)
    start = clean.find(value)
    if start < 0:
        return ""
    return clean[max(0, start - window): min(len(clean), start + len(value) + window)]


def extract_containing_sentence(text: str, value: str) -> str:
    clean = _TAG.sub("", text)
    start = clean.find(value)
    if start < 0:
        return ""
    left = max((clean.rfind(mark, 0, start) for mark in (".", "!", "?", ";", "\n")), default=-1)
    ends = [clean.find(mark, start + len(value)) for mark in (".", "!", "?", ";", "\n")]
    right_candidates = [position for position in ends if position >= 0]
    right = min(right_candidates) if right_candidates else len(clean)
    return clean[left + 1:right + 1].strip()


class SeedPackValidator:
    def __init__(self, hard_negative: HardNegativeConfig, config: ValidationConfig) -> None:
        self.hard_negative = hard_negative
        self.config = config

    def validate(
        self, pack: SeedPack, task_focus_labels: Sequence[str], taxonomy: Sequence[TaxonomyLabel]
    ) -> DeterministicValidationResult:
        issues: list[ValidationIssue] = []
        taxonomy_labels = {label.code for label in taxonomy}
        sample_type = SampleType(pack.sample_type)
        positives = pack.positive_entities

        if sample_type == SampleType.PURE_NEGATIVE:
            if positives:
                issues.append(self._issue("invalid_seed", "pure_negative contains positive entities"))
            if pack.decoys:
                issues.append(self._issue("invalid_seed", "pure_negative contains decoys"))
            if pack.content_seeds is None:
                issues.append(self._issue("invalid_seed", "pure_negative requires content_seeds"))
            elif self._content_contains_structured_pii(pack.content_seeds.json()):
                issues.append(self._issue("malformed_seed", "pure-negative content seeds contain PII-like values"))
            return DeterministicValidationResult(valid=not issues, issues=issues)

        hard_decoy_only = (
            sample_type == SampleType.HARD_NEGATIVE and pack.hard_negative_mode == "decoy_only"
        )
        present_labels = {seed.label for seed in positives}
        if hard_decoy_only:
            if positives:
                issues.append(self._issue("invalid_seed", "decoy_only hard_negative cannot contain positive entities"))
        else:
            for missing in set(task_focus_labels) - present_labels:
                issues.append(self._issue("missing_positive_seed", f"missing positive seed for {missing}", missing))
            for extra in present_labels - set(task_focus_labels):
                issues.append(self._issue("invalid_seed", f"positive seed label is outside focus labels: {extra}", extra))

        seen: set[str] = set()
        for seed in positives:
            key = seed.value.strip().casefold()
            if not key:
                issues.append(self._issue("malformed_seed", "positive seed value is empty", seed.label))
            elif key in seen:
                issues.append(self._issue("invalid_seed", "positive seed values must be unique", seed.label, seed.value))
            seen.add(key)
            if seed.label not in taxonomy_labels:
                issues.append(self._issue("invalid_seed", "seed label is absent from taxonomy", seed.label))
            if self.config.reject_mixed_locale and _MIXED_LOCALE.search(seed.value):
                issues.append(self._issue("mixed_locale", "seed contains a mixed-locale or placeholder token", seed.label, seed.value))
            if not self._valid_format(seed.label, seed.value):
                issues.append(self._issue("malformed_seed", "seed does not satisfy label format", seed.label, seed.value))

        if sample_type == SampleType.HARD_NEGATIVE:
            if len(pack.decoys) < self.hard_negative.min_decoys:
                issues.append(self._issue("unsupported_decoy", "hard_negative has too few decoys"))
            if len(pack.decoys) > self.hard_negative.max_decoys:
                issues.append(self._issue("unsupported_decoy", "hard_negative exceeds max_decoys"))
            for decoy in pack.decoys:
                if decoy.value.strip().casefold() in seen:
                    issues.append(self._issue("invalid_seed", "decoy duplicates a positive seed", decoy.target_label, decoy.value))
                strategies = HARD_NEGATIVE_STRATEGIES.get(decoy.target_label, ())
                strategy = next((item for item in strategies if item.strategy_id == decoy.strategy_id), None)
                if strategy is None:
                    issues.append(self._issue("unsupported_decoy", "decoy strategy is not registered", decoy.target_label, decoy.value))
                else:
                    try:
                        strategy_matches = strategy.validator(decoy.value)
                    except (IndexError, TypeError, ValueError):
                        strategy_matches = False
                    if not strategy_matches:
                        issues.append(self._issue("malformed_seed", "decoy value does not match its strategy", decoy.target_label, decoy.value))
                if not decoy.required_context_cues:
                    issues.append(self._issue("unsupported_decoy", "decoy requires contextual cues", decoy.target_label, decoy.value))
                if re.fullmatch(r"[a-z_]+-\d+(?:-beta)?", decoy.value, re.IGNORECASE):
                    issues.append(self._issue("unsupported_decoy", "label-plus-random-number decoys are forbidden", decoy.target_label, decoy.value))
                if decoy.target_label not in task_focus_labels:
                    issues.append(self._issue("unsupported_decoy", "decoy target must be a taxonomy focus label", decoy.target_label, decoy.value))
                if decoy.target_label not in decoy.negative_labels:
                    issues.append(self._issue("invalid_seed", "decoy target must be declared negative", decoy.target_label, decoy.value))
                unknown_metadata = (
                    set(decoy.negative_labels) | set(decoy.possible_collision_labels)
                ) - taxonomy_labels
                if unknown_metadata:
                    issues.append(self._issue(
                        "invalid_seed", f"decoy metadata contains labels absent from taxonomy: {sorted(unknown_metadata)}",
                        decoy.target_label, decoy.value,
                    ))
                collisions = self._structured_labels_for_value(decoy.value) - set(decoy.negative_labels)
                if collisions:
                    issues.append(self._issue(
                        "taxonomy_collision",
                        f"decoy also satisfies structured taxonomy labels: {sorted(collisions)}",
                        decoy.target_label, decoy.value,
                    ))
                declared_collisions = set(decoy.possible_collision_labels) & self._structured_labels_for_value(decoy.value)
                if declared_collisions:
                    issues.append(self._issue(
                        "taxonomy_collision",
                        f"decoy has unresolved possible taxonomy collisions: {sorted(declared_collisions)}",
                        decoy.target_label, decoy.value,
                    ))
            if hard_decoy_only:
                covered = {decoy.target_label for decoy in pack.decoys}
                missing_targets = set(task_focus_labels) - covered
                if missing_targets:
                    issues.append(self._issue(
                        "unsupported_decoy", f"hard-negative targets lack decoys: {sorted(missing_targets)}"
                    ))
        elif pack.decoys:
            issues.append(self._issue("invalid_seed", "positive samples cannot contain decoys"))

        return DeterministicValidationResult(valid=not issues, issues=issues)

    @staticmethod
    def _issue(issue_type: str, reason: str, label: str | None = None, value: str | None = None) -> ValidationIssue:
        return ValidationIssue(type=issue_type, scope="SEEDS", reason=reason, label=label, value=value)

    @staticmethod
    def _content_contains_structured_pii(value: str) -> bool:
        return any(pattern.search(value) for pattern in (_EMAIL, _PHONE, _URL, _DATE, _TIME, _IP, _IDENTIFIER))

    @classmethod
    def _valid_format(cls, label: str, value: str) -> bool:
        if label == "EMAIL":
            return _EMAIL.fullmatch(value) is not None
        if label in {"DATE", "BIRTHDATE"}:
            return any(cls._matches_datetime(value, date_format) for date_format in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"))
        if label == "TIME":
            return any(cls._matches_datetime(value, time_format) for time_format in ("%H:%M", "%H:%M:%S", "%I:%M %p"))
        if label == "PHONE":
            return _PHONE.fullmatch(value) is not None
        if label == "URL":
            parsed = urlparse(value)
            return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
        if label == "IP":
            try:
                ipaddress.ip_address(value)
                return True
            except ValueError:
                return False
        if label == "ADDRESS":
            normalized = value.casefold()
            administrative_markers = (
                "phường", "quận", "huyện", "tỉnh",
                *(city.casefold() for city in VIETNAMESE_ADMINISTRATIVE_AREAS),
            )
            has_street_detail = bool(re.search(
                r"\b\d+\s+đường\s+\S+",
                value,
                re.IGNORECASE,
            ))
            has_premise_detail = bool(re.search(
                r"\b(?:căn hộ|phòng)\s+[A-Z0-9-]+"
                r"|\btòa(?:\s+nhà)?\s+[\w-]+",
                value,
                re.IGNORECASE,
            ))
            return (
                (has_street_detail or has_premise_detail)
                and not any(marker in normalized for marker in administrative_markers)
            )
        if label in {"CARD_NUMBER", "NATIONAL_ID", "BANK_ACCOUNT", "TIN", "ZIP_CODE", "CVV", "PIN"}:
            return value.isdigit()
        return bool(value.strip())

    @staticmethod
    def _matches_datetime(value: str, value_format: str) -> bool:
        try:
            datetime.strptime(value, value_format)
            return True
        except ValueError:
            return False

    @classmethod
    def _structured_labels_for_value(cls, value: str) -> set[str]:
        labels: set[str] = set()
        for label in ("EMAIL", "PHONE", "URL", "IP", "DATE", "BIRTHDATE", "TIME", "ADDRESS"):
            if cls._valid_format(label, value):
                labels.add(label)
        if re.fullmatch(r"\d{5,6}", value):
            labels.add("ZIP_CODE")
        if re.fullmatch(r"\d{13,19}", value):
            labels.add("CARD_NUMBER")
        if re.fullmatch(r"\d{3,4}", value):
            labels.update({"CVV", "PIN"})
        if re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", value):
            labels.add("IBAN")
        if re.fullmatch(r"[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?", value):
            labels.add("SWIFT")
        if re.fullmatch(r"0x[0-9a-fA-F]{40}", value):
            labels.add("WALLET")
        return labels


class DeterministicOutputValidator:
    def __init__(self, config: ValidationConfig) -> None:
        self.config = config

    def validate(
        self,
        *,
        tagged_text: str,
        entities: Sequence[GeneratedEntity | dict[str, str]],
        seed_pack: SeedPack,
        focus_labels: Sequence[str],
        max_entities: int,
        length_target: LengthTarget | None = None,
    ) -> DeterministicValidationResult:
        issues: list[ValidationIssue] = []
        raw_entities = [entity.dict() if isinstance(entity, GeneratedEntity) else entity for entity in entities]
        hard_decoy_only = (
            SampleType(seed_pack.sample_type) == SampleType.HARD_NEGATIVE
            and seed_pack.hard_negative_mode == "decoy_only"
        )
        try:
            validate_generated_output(
                tagged_text=tagged_text,
                entities=raw_entities,
                allowed_labels=focus_labels,
                required_labels=focus_labels,
                sample_type="pure_negative" if hard_decoy_only else str(seed_pack.sample_type),
                max_entities=max_entities,
            )
        except ValueError as exc:
            issues.append(ValidationIssue(type="invalid_output", scope="TEXT", reason=str(exc)))

        clean_text = _TAG.sub("", tagged_text).strip()
        if length_target is not None:
            word_count = len(re.findall(r"\S+", clean_text))
            if not length_target.min_words <= word_count <= length_target.max_words:
                issues.append(ValidationIssue(
                    type="length_out_of_range",
                    scope="TEXT",
                    reason=(
                        f"clean text has {word_count} words; expected "
                        f"{length_target.min_words}-{length_target.max_words}"
                    ),
                ))

        if SampleType(seed_pack.sample_type) == SampleType.PURE_NEGATIVE:
            if self.config.pure_negative_structured_scan:
                clean = clean_text
                for label, pattern in self._structured_detectors():
                    match = pattern.search(clean)
                    if match:
                        issues.append(ValidationIssue(
                            type="pure_negative_contains_pii", scope="TEXT",
                            reason=f"pure_negative contains a structured {label} candidate",
                            label=label, value=match.group(0),
                        ))
            return DeterministicValidationResult(valid=not issues, issues=self._unique_issues(issues))

        listed_pairs = {(str(item.get("label")), str(item.get("value"))) for item in raw_entities}
        for seed in seed_pack.positive_entities:
            expected = f"<{seed.label}>{seed.value}</{seed.label}>"
            if tagged_text.count(expected) != 1:
                issues.append(ValidationIssue(
                    type="missing_positive_seed", scope="TEXT",
                    reason="positive seed must appear exactly once with its exact tag", label=seed.label, value=seed.value,
                ))
            if (seed.label, seed.value) not in listed_pairs:
                issues.append(ValidationIssue(
                    type="missing_entity_metadata", scope="TEXT",
                    reason="positive seed is missing from entities", label=seed.label, value=seed.value,
                ))
            remainder = tagged_text.replace(expected, "", 1)
            if seed.value in _TAG.sub("", remainder):
                issues.append(ValidationIssue(
                    type="duplicate_positive_seed", scope="TEXT",
                    reason="positive seed also appears outside its required tag", label=seed.label, value=seed.value,
                ))

        entity_values = {str(item.get("value", "")) for item in raw_entities}
        for decoy in seed_pack.decoys:
            occurrence_count = tagged_text.count(decoy.value)
            max_occurrences = 2 if hard_decoy_only else 1
            if occurrence_count < 1 or occurrence_count > max_occurrences:
                issues.append(self._decoy_issue(
                    "decoy_occurrence",
                    f"decoy must appear between 1 and {max_occurrences} times",
                    decoy,
                ))
            if re.search(rf"<[A-Za-z][A-Za-z0-9_]*>{re.escape(decoy.value)}</[A-Za-z][A-Za-z0-9_]*>", tagged_text):
                issues.append(self._decoy_issue("decoy_tagged", "decoy must remain untagged", decoy))
            if decoy.value in entity_values:
                issues.append(self._decoy_issue("decoy_in_entities", "decoy cannot appear in entities", decoy))
            contexts = extract_occurrence_contexts(tagged_text, decoy.value)
            if len(contexts) != occurrence_count or any(
                not any(cue.casefold() in context.casefold() for cue in decoy.required_context_cues)
                for context in contexts
            ):
                issues.append(self._decoy_issue(
                    "decoy_context_unclear",
                    "every decoy occurrence must have a required context cue in the same sentence",
                    decoy,
                ))
            clean = _TAG.sub("", tagged_text)
            positions: list[int] = []
            search_from = 0
            while decoy.value:
                position = clean.find(decoy.value, search_from)
                if position < 0:
                    break
                positions.append(position)
                search_from = position + len(decoy.value)
            if any(
                cue.casefold() in clean[max(0, position - 30):position].casefold()
                for position in positions
                for cue in decoy.forbidden_context_cues
            ):
                issues.append(self._decoy_issue("decoy_used_as_pii", "forbidden PII cue directly introduces decoy", decoy))

        if hard_decoy_only and self.config.hard_negative_structured_scan:
            clean_without_decoys = _TAG.sub("", tagged_text)
            for decoy in seed_pack.decoys:
                clean_without_decoys = clean_without_decoys.replace(decoy.value, " ")
            for label, pattern in self._structured_detectors():
                match = pattern.search(clean_without_decoys)
                if match:
                    issues.append(ValidationIssue(
                        type="hard_negative_contains_extra_pii", scope="TEXT",
                        reason=f"hard_negative contains an extra structured {label} candidate",
                        label=label, value=match.group(0),
                    ))

        return DeterministicValidationResult(valid=not issues, issues=self._unique_issues(issues))

    @staticmethod
    def _structured_detectors() -> Iterable[tuple[str, re.Pattern[str]]]:
        return (("EMAIL", _EMAIL), ("PHONE", _PHONE), ("URL", _URL), ("DATE", _DATE),
                ("TIME", _TIME), ("IP", _IP), ("ACCOUNT_ID", _IDENTIFIER))

    @staticmethod
    def _decoy_issue(issue_type: str, reason: str, decoy: object) -> ValidationIssue:
        return ValidationIssue(
            type=issue_type, scope="TEXT", reason=reason,
            label=getattr(decoy, "target_label"), value=getattr(decoy, "value"),
        )

    @staticmethod
    def _unique_issues(issues: Sequence[ValidationIssue]) -> list[ValidationIssue]:
        unique: list[ValidationIssue] = []
        seen: set[tuple[str, str, str | None, str | None]] = set()
        for issue in issues:
            key = (issue.type, issue.reason, issue.label, issue.value)
            if key not in seen:
                seen.add(key)
                unique.append(issue)
        return unique
