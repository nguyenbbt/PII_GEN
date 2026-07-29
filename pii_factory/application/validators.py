from __future__ import annotations

import ipaddress
import re
from collections import Counter
from datetime import datetime
from typing import Iterable, Sequence
from urllib.parse import urlparse

from data_generator_worker.validation import (
    decoy_contexts_are_valid,
    extract_occurrence_contexts,
    find_template_artifacts,
    seed_realization_status,
    tagged_text_to_clean_and_spans,
    validate_generated_output,
)

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
from .seed_generation import HARD_NEGATIVE_STRATEGIES


_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w-])")
_PHONE = re.compile(r"(?<!\d)(?:\+?84|0)[ .-]?(?:3|5|7|8|9)(?:[ .-]?\d){8}(?!\d)")
_URL = re.compile(
    r"https?://[^\s<>]*[^\s<>.,;:!?\)\]\}]",
    re.IGNORECASE,
)
_DATE = re.compile(
    r"(?:(?<!\d)(?:0?[1-9]|[12]\d|3[01])\s*-\s*"
    r"(?:0?[1-9]|[12]\d|3[01])[/.-](?:0?[1-9]|1[0-2])[/.-](?:19|20)\d{2}(?!\d)|"
    r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[/.-](?:0?[1-9]|1[0-2])[/.-](?:19|20)\d{2}(?!\d)|"
    r"(?<![A-Za-z0-9-])(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    r"(?![A-Za-z0-9-])|"
    r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])\s+tháng\s+(?:0?[1-9]|1[0-2])"
    r"(?:\s+năm\s+(?:(?:19|20)\d{2}|nay))?(?![\w]))",
    re.IGNORECASE,
)
_TIME = re.compile(
    r"(?<!\d)(?:"
    r"(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?"
    r"|(?:[01]?\d|2[0-3])\s*h(?:\s*[0-5]?\d)?"
    r"|(?:[01]?\d|2[0-3])\s*giờ(?:\s*[0-5]?\d(?:\s*phút)?)?"
    r")"
    r"(?:\s*(?:AM|PM|sáng|trưa|chiều|tối))?"
    r"(?![\w])",
    re.IGNORECASE,
)
_IP = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_IDENTIFIER = re.compile(r"\b(?:ACC|USR|EMP|INC|BH)-[A-Z0-9]{5,}\b", re.IGNORECASE)
_PLATE_WITH_CONTEXT = re.compile(
    r"(?i)(?:biển(?:\s+số)?(?:\s+xe)?|biển\s+kiểm\s+soát)"
    r"\s*(?:là|số|:)?\s*"
    r"(?P<value>\d{2}[A-Z]{1,2}[-\s]?\d{3}(?:[.\s]?\d{2})?)"
)
_TICKET_WITH_CONTEXT = re.compile(
    r"(?i)(?:mã\s+(?:sự\s+cố|phiếu|yêu\s+cầu|ticket)|ticket)"
    r"\s*(?:là|số|:)?\s*"
    r"(?P<value>[A-Z]{2,10}-[A-Z0-9]+(?:-[A-Z0-9]+)+)"
)
_HARD_NEGATIVE_META = re.compile(
    r"(?i)(?:đây|giá\s+trị\s+này|chuỗi\s+này|dữ\s+liệu\s+này)?\s*"
    r"(?:không\s+phải\s+(?:là\s+)?(?:email|số\s+điện\s+thoại|PII|thông\s+tin\s+cá\s+nhân)"
    r"|chỉ\s+là\s+(?:dữ\s+liệu|chuỗi|giá\s+trị)\s+giả)"
)
_ADDRESS = re.compile(r"(?=.*\d)(?=.*[^\W\d_]).*\s+.*", re.UNICODE)
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9_]*>")
_ENTITY_PAIR = re.compile(r"<([A-Z][A-Z0-9_]*)>(.*?)</\1>", re.DOTALL)
_MIXED_LOCALE = re.compile(r"JaneHuyện|JohnQuận|SmithPhường|\b(?:County|Street|Avenue|undefined|null|N/A|xxx)\b", re.IGNORECASE)
_BIRTHDATE_CONTEXT = re.compile(
    r"(?:ngày\s+sinh|sinh\s+ngày|chào\s+đời)(?:\s+(?:là|vào))?\s*$",
    re.IGNORECASE,
)
_TIME_CONTEXT = re.compile(
    r"(?:lúc|vào\s+lúc|bắt\s+đầu(?:\s+lúc)?|kết\s+thúc(?:\s+lúc)?|"
    r"từ|đến|thời\s+điểm|khoảng)\s*$",
    re.IGNORECASE,
)
_NON_TIME_CONTEXT = re.compile(
    r"(?:thời\s+lượng|kéo\s+dài|trong\s+vòng|sau|tỷ\s+lệ|tỉ\s+lệ)\s*$",
    re.IGNORECASE,
)
_TIME_DAYPART = re.compile(r"(?:AM|PM|sáng|trưa|chiều|tối)\s*$", re.IGNORECASE)


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
            key = seed.value
            comes_from_value_bank = seed.format_variant == "value_bank"
            if not key.strip():
                issues.append(self._issue("malformed_seed", "positive seed value is empty", seed.label))
            elif key in seen:
                issues.append(self._issue("invalid_seed", "positive seed values must be unique", seed.label, seed.value))
            seen.add(key)
            if seed.label not in taxonomy_labels:
                issues.append(self._issue("invalid_seed", "seed label is absent from taxonomy", seed.label))
            if (
                not comes_from_value_bank
                and self.config.reject_mixed_locale
                and _MIXED_LOCALE.search(seed.value)
            ):
                issues.append(self._issue("mixed_locale", "seed contains a mixed-locale or placeholder token", seed.label, seed.value))
            if not comes_from_value_bank and not self._valid_format(seed.label, seed.value):
                issues.append(self._issue("malformed_seed", "seed does not satisfy label format", seed.label, seed.value))

        if sample_type == SampleType.HARD_NEGATIVE:
            if len(pack.decoys) < self.hard_negative.min_decoys:
                issues.append(self._issue("unsupported_decoy", "hard_negative has too few decoys"))
            if len(pack.decoys) > self.hard_negative.max_decoys:
                issues.append(self._issue("unsupported_decoy", "hard_negative exceeds max_decoys"))
            for decoy in pack.decoys:
                if decoy.value in seen:
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
            return _ADDRESS.fullmatch(value) is not None
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
        allowed_labels: Sequence[str] | None = None,
        length_target: LengthTarget | None = None,
    ) -> DeterministicValidationResult:
        issues: list[ValidationIssue] = []
        raw_entities = [entity.dict() if isinstance(entity, GeneratedEntity) else entity for entity in entities]
        seeded_entity_counts = Counter(
            (seed.label, seed.value)
            for seed in seed_pack.positive_entities
        )
        observed_seed_counts: Counter[tuple[str, str]] = Counter()
        additional_temporal_entities = 0
        for entity in raw_entities:
            pair = (
                str(entity.get("label", "")).strip(),
                str(entity.get("value", "")).strip(),
            )
            if observed_seed_counts[pair] < seeded_entity_counts[pair]:
                observed_seed_counts[pair] += 1
            elif pair[0] in {"DATE", "TIME"}:
                additional_temporal_entities += 1
        hard_decoy_only = (
            SampleType(seed_pack.sample_type) == SampleType.HARD_NEGATIVE
            and seed_pack.hard_negative_mode == "decoy_only"
        )
        try:
            validate_generated_output(
                tagged_text=tagged_text,
                entities=raw_entities,
                allowed_labels=(allowed_labels or focus_labels),
                required_labels=focus_labels,
                sample_type="pure_negative" if hard_decoy_only else str(seed_pack.sample_type),
                # DATE/TIME values naturally introduced as event context must be
                # annotated, even though they were not part of the planned
                # Value Bank seed budget.
                max_entities=max_entities + additional_temporal_entities,
            )
        except ValueError as exc:
            reason = str(exc)
            if reason == "entities must correspond exactly to tagged spans":
                issues.extend(self._metadata_sync_issues(tagged_text, raw_entities))
            else:
                issues.append(ValidationIssue(type="invalid_output", scope="TEXT", reason=reason))

        clean_text = _TAG.sub("", tagged_text).strip()
        for _, _, artifact in find_template_artifacts(tagged_text):
            issues.append(ValidationIssue(
                type="template_artifact",
                scope="TEXT",
                reason=(
                    f"unresolved human-facing template field {artifact!r}; "
                    "replace it with finished generic prose, not invented PII"
                ),
                value=artifact,
            ))
        if length_target is not None:
            word_count = len(re.findall(r"\S+", clean_text))
            if word_count < length_target.min_words:
                issues.append(ValidationIssue(
                    type="length_below_minimum",
                    scope="TEXT",
                    reason=(
                        f"clean text has {word_count} words; expected at least "
                        f"{length_target.min_words}. The preferred upper target "
                        f"{length_target.max_words} is guidance only."
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

        for seed in seed_pack.positive_entities:
            status = seed_realization_status(
                tagged_text,
                label=seed.label,
                value=seed.value,
            )
            if status == "missing":
                issues.append(ValidationIssue(
                    type="missing_positive_seed", scope="TEXT",
                    reason="positive seed surface is absent from clean text",
                    label=seed.label, value=seed.value,
                ))
            elif status == "boundary_whitespace":
                issues.append(ValidationIssue(
                    type="entity_boundary_whitespace", scope="TEXT",
                    reason="entity tag includes whitespace outside the exact positive-seed boundary",
                    label=seed.label, value=seed.value,
                ))
            elif status not in {"exact", "partitioned"}:
                issues.append(ValidationIssue(
                    type="positive_seed_annotation_mismatch", scope="TEXT",
                    reason=(
                        "positive seed surface is present but its occurrences are "
                        "untagged, inconsistently tagged, or use unsafe taxonomy boundaries"
                    ),
                    label=seed.label, value=seed.value,
                ))

        if not hard_decoy_only:
            issues.extend(self._missing_repeated_annotations(tagged_text))
            issues.extend(self._missing_structured_annotations(
                tagged_text=tagged_text,
                allowed_labels=(allowed_labels or focus_labels),
                excluded_values={decoy.value for decoy in seed_pack.decoys},
            ))
            issues.extend(self._temporal_boundary_issues(tagged_text))

        entity_values = {str(item.get("value", "")) for item in raw_entities}
        for decoy in seed_pack.decoys:
            occurrence_count = tagged_text.count(decoy.value)
            max_occurrences = (
                3
                if SampleType(seed_pack.sample_type)
                == SampleType.HARD_NEGATIVE
                else 1
            )
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
            if (
                len(contexts) != occurrence_count
                or not decoy_contexts_are_valid(
                    tagged_text,
                    decoy.value,
                    decoy.required_context_cues,
                )
            ):
                cues = ", ".join(repr(cue) for cue in decoy.required_context_cues)
                observed = " | ".join(contexts) if contexts else "<not found>"
                issues.append(self._decoy_issue(
                    "decoy_context_unclear",
                    (
                        "the first decoy occurrence must have one of these exact "
                        "cues; later occurrences may omit it in the same "
                        "paragraph. A later schema/data-field paragraph may use "
                        "the established short reference 'field'; otherwise "
                        f"repeat an exact cue: {cues}; observed context: {observed}"
                    ),
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

        if SampleType(seed_pack.sample_type) == SampleType.HARD_NEGATIVE:
            meta_match = _HARD_NEGATIVE_META.search(clean_text)
            if meta_match:
                issues.append(ValidationIssue(
                    type="hard_negative_meta_explanation",
                    scope="TEXT",
                    reason=(
                        "hard-negative contrast is explained as annotation policy "
                        "instead of being demonstrated by natural operational context"
                    ),
                    value=meta_match.group(0),
                ))

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
    def _missing_repeated_annotations(
        tagged_text: str,
    ) -> list[ValidationIssue]:
        """Report exact, case-sensitive repeats left outside same-label spans."""
        clean, spans = tagged_text_to_clean_and_spans(tagged_text)
        issues: list[ValidationIssue] = []
        for label, value in dict.fromkeys(
            (span.label, span.value)
            for span in spans
            if span.value
        ):
            search_from = 0
            occurrence = 0
            while True:
                start = clean.find(value, search_from)
                if start < 0:
                    break
                occurrence += 1
                end = start + len(value)
                search_from = end
                covered_by_same_label = any(
                    span.label == label
                    and span.start <= start
                    and span.end >= end
                    for span in spans
                )
                if covered_by_same_label:
                    continue
                issues.append(ValidationIssue(
                    type="missing_annotation_candidate",
                    scope="TEXT",
                    reason=(
                        f"exact repeated {label} value {value!r} at occurrence "
                        f"{occurrence} is untagged; every case-sensitive standalone "
                        "occurrence must be annotated"
                    ),
                    label=label,
                    value=value,
                ))
        return issues

    @staticmethod
    def _missing_structured_annotations(
        *,
        tagged_text: str,
        allowed_labels: Sequence[str],
        excluded_values: set[str],
    ) -> list[ValidationIssue]:
        clean, spans = tagged_text_to_clean_and_spans(tagged_text)
        allowed = set(allowed_labels)
        candidates: list[tuple[str, str, int, int]] = []
        for label, pattern in DeterministicOutputValidator._structured_detectors():
            if label not in allowed:
                continue
            for match in pattern.finditer(clean):
                if (
                    label == "DATE"
                    and _BIRTHDATE_CONTEXT.search(clean[max(0, match.start() - 35):match.start()])
                ):
                    continue
                if (
                    label == "TIME"
                    and not DeterministicOutputValidator._is_specific_time_context(
                        clean,
                        match.start(),
                        match.end(),
                    )
                ):
                    continue
                candidates.append((label, match.group(0), match.start(), match.end()))
        for label, pattern in (
            ("PLATE", _PLATE_WITH_CONTEXT),
            ("TICKET_ID", _TICKET_WITH_CONTEXT),
        ):
            if label not in allowed:
                continue
            for match in pattern.finditer(clean):
                candidates.append((
                    label,
                    match.group("value"),
                    match.start("value"),
                    match.end("value"),
                ))

        issues: list[ValidationIssue] = []
        for label, value, start, end in candidates:
            if value in excluded_values:
                continue
            if any(span.start <= start and span.end >= end for span in spans):
                continue
            context = clean[max(0, start - 45):min(len(clean), end + 45)].strip()
            issues.append(ValidationIssue(
                type="missing_annotation_candidate",
                scope="TEXT",
                reason=(
                    f"clear contextual {label} candidate {value!r} is untagged; "
                    f"local context: {context!r}"
                ),
                label=label,
                value=value,
            ))
        return issues

    @staticmethod
    def _is_specific_time_context(text: str, start: int, end: int) -> bool:
        value = text[start:end]
        if _TIME_DAYPART.search(value):
            return True
        preceding = text[max(0, start - 35):start]
        if _NON_TIME_CONTEXT.search(preceding):
            return False
        return bool(_TIME_CONTEXT.search(preceding))

    @staticmethod
    def _temporal_boundary_issues(
        tagged_text: str,
    ) -> list[ValidationIssue]:
        """Require DATE/TIME tags to contain only the temporal value surface."""
        _, spans = tagged_text_to_clean_and_spans(tagged_text)
        patterns = {"DATE": _DATE, "TIME": _TIME}
        issues: list[ValidationIssue] = []
        for span in spans:
            pattern = patterns.get(span.label)
            if pattern is None:
                continue
            matches = list(pattern.finditer(span.value))
            if not matches:
                continue
            exact_single_match = (
                len(matches) == 1
                and matches[0].start() == 0
                and matches[0].end() == len(span.value)
            )
            if exact_single_match:
                continue
            expected = [match.group(0) for match in matches]
            issues.append(ValidationIssue(
                type="temporal_boundary",
                scope="TEXT",
                reason=(
                    f"{span.label} span {span.value!r} has an invalid boundary; "
                    f"tag only the temporal value(s) {expected!r}. Keep leading "
                    "cue words and UTC/GMT/timezone text outside the tag, and "
                    "use separate tags when multiple temporal values appear."
                ),
                label=span.label,
                value=span.value,
            ))
        return issues

    @staticmethod
    def _metadata_sync_issues(
        tagged_text: str,
        entities: Sequence[dict[str, str]],
    ) -> list[ValidationIssue]:
        tag_counts = Counter(_ENTITY_PAIR.findall(tagged_text))
        entity_counts = Counter(
            (
                str(entity.get("label", "")).strip(),
                str(entity.get("value", "")).strip(),
            )
            for entity in entities
        )
        normalized_tag_counts = Counter(
            (label, value.strip())
            for label, value in _ENTITY_PAIR.findall(tagged_text)
        )
        if normalized_tag_counts == entity_counts:
            return [
                ValidationIssue(
                    type="entity_boundary_whitespace",
                    scope="TEXT",
                    reason=(
                        "entity tag includes whitespace outside the exact "
                        "positive-seed boundary"
                    ),
                    label=label,
                    value=value.strip(),
                )
                for label, value in _ENTITY_PAIR.findall(tagged_text)
                if value != value.strip()
            ]
        issues: list[ValidationIssue] = []
        for (label, value), count in (tag_counts - entity_counts).items():
            issues.append(ValidationIssue(
                type="missing_entity_metadata",
                scope="TEXT",
                reason=(
                    f"{count} tagged {label} occurrence(s) are missing matching "
                    "entities metadata"
                ),
                label=label,
                value=value,
            ))
        for (label, value), count in (entity_counts - tag_counts).items():
            issues.append(ValidationIssue(
                type="extra_entity_metadata",
                scope="TEXT",
                reason=(
                    f"{count} {label} entities metadata occurrence(s) have no "
                    "matching tagged span"
                ),
                label=label,
                value=value,
            ))
        return issues or [
            ValidationIssue(
                type="invalid_output",
                scope="TEXT",
                reason="entities must correspond exactly to tagged spans",
            )
        ]

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
