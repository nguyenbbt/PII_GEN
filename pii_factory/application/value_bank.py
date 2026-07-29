from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence


_CLASS_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_LANGUAGE = re.compile(r"^[a-z]{2}$")
_SUPPORTED_VERSION = 1
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_value_bank_path(
    value_bank_path: str | Path,
    *,
    config_directory: str | Path | None = None,
) -> Path:
    """Resolve a bank path consistently for CLI, API, and installed entry points."""
    path = Path(value_bank_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates: list[Path] = []
    if config_directory is not None:
        candidates.append(Path(config_directory).expanduser() / path)
    candidates.extend((Path.cwd() / path, _PROJECT_ROOT / path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


class ValueBankError(ValueError):
    """Base error for Value Bank configuration and data problems."""


class ValueBankLanguageError(ValueBankError):
    """Raised when the requested language has no Value Bank file."""


class ValueBankClassError(ValueBankError):
    """Raised when a language bank does not define the requested entity class."""


class ValueBankEmptyClassError(ValueBankError):
    """Raised when an entity class has no selectable values."""


class InvalidValueBankError(ValueBankError):
    """Raised when a Value Bank file does not satisfy the versioned JSON contract."""


@dataclass(frozen=True)
class GeneratedEntityValue:
    value: str
    format_variant: str = "value_bank"


class ValueBankEntityProvider:
    """Load and deterministically sample entity values from language JSON banks.

    Relative paths are resolved from the process working directory with a
    project-root fallback. Files are loaded lazily and cached after validation.
    Every source entry is retained, including case variants and intentional
    duplicates. ADDRESS entries whose comma-delimited suffix is also present in
    LOCATION are excluded at selection time so taxonomy boundaries cannot be
    collapsed; the source file itself remains unchanged.
    """

    def __init__(
        self,
        value_bank_path: str | Path = "PII_Value_Bank",
        language_files: Mapping[str, str | Path] | None = None,
        *,
        partition_index: int = 0,
        partition_count: int = 1,
    ) -> None:
        if partition_count < 1 or not 0 <= partition_index < partition_count:
            raise ValueError("invalid Value Bank partition")
        self.path = resolve_value_bank_path(value_bank_path)
        self.partition_index = partition_index
        self.partition_count = partition_count
        self.language_files = {
            self._normalise_language(language): Path(file_path).expanduser()
            for language, file_path in (language_files or {}).items()
        }
        self._banks: Dict[str, Dict[str, tuple[str, ...]]] = {}

    def validate(
        self,
        *,
        language: str,
        required_labels: Iterable[str],
    ) -> None:
        """Eagerly validate the selected language and required taxonomy classes."""
        for label in dict.fromkeys(str(item).strip().upper() for item in required_labels):
            self.values_for(language, label)

    def generate(
        self,
        label: str,
        language: str,
        rng: random.Random,
        *,
        excluded_values: Iterable[str] = (),
    ) -> str:
        return self.generate_with_variant(
            label,
            language,
            rng,
            excluded_values=excluded_values,
        ).value

    def generate_with_variant(
        self,
        label: str,
        language: str,
        rng: random.Random,
        *,
        excluded_values: Iterable[str] = (),
    ) -> GeneratedEntityValue:
        values = self.values_for(language, label)
        excluded = {str(value) for value in excluded_values}
        available = [value for value in values if value not in excluded]
        if not available:
            raise ValueBankEmptyClassError(
                f"Value Bank class {label!r} for language {language!r} "
                "has no unused value available"
            )
        return GeneratedEntityValue(rng.choice(available))

    def values_for(self, language: str, label: str) -> Sequence[str]:
        normalised_language = self._normalise_language(language)
        normalised_label = str(label).strip().upper()
        bank = self._load_language(normalised_language)
        if normalised_label not in bank:
            raise ValueBankClassError(
                f"Value Bank language {normalised_language!r} "
                f"does not define class {normalised_label!r}"
            )
        source_values = bank[normalised_label]
        if normalised_label == "ADDRESS":
            source_values = tuple(
                value
                for value in source_values
                if not self._contains_location_suffix(
                    value,
                    bank.get("LOCATION", ()),
                )
            )
        values = tuple(
            value
            for value in source_values
            if self._belongs_to_partition(value)
        )
        if not values:
            raise ValueBankEmptyClassError(
                f"Value Bank class {normalised_label!r} for language "
                f"{normalised_language!r} has no values"
            )
        return values

    @staticmethod
    def _contains_location_suffix(
        address: str,
        locations: Sequence[str],
    ) -> bool:
        normalized_address = " ".join(
            address.casefold().split()
        )
        normalized_locations = {
            " ".join(location.casefold().split())
            for location in locations
        }
        return any(
            normalized_address.endswith(f", {location}")
            for location in normalized_locations
            if location
        )

    def _belongs_to_partition(self, value: str) -> bool:
        if self.partition_count == 1:
            return True
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % self.partition_count
        return bucket == self.partition_index

    def _load_language(self, language: str) -> Dict[str, tuple[str, ...]]:
        cached = self._banks.get(language)
        if cached is not None:
            return cached
        configured_file = self.language_files.get(language)
        if self.language_files and configured_file is None:
            raise ValueBankLanguageError(
                f"Value Bank has no configured file for language {language!r}"
            )
        configured_file = configured_file or Path(
            f"{language}_pii_value_pools.json"
        )
        if configured_file.is_absolute():
            file_path = configured_file
        else:
            if not self.path.is_dir():
                raise ValueBankLanguageError(
                    "Value Bank directory does not exist or is not a "
                    f"directory: {self.path}"
                )
            file_path = self.path / configured_file
        if not file_path.is_file():
            raise ValueBankLanguageError(
                f"Value Bank has no file for language {language!r}: {file_path}"
            )
        bank = self._read_and_validate(file_path, language)
        self._banks[language] = bank
        return bank

    @classmethod
    def _read_and_validate(
        cls,
        file_path: Path,
        language: str,
    ) -> Dict[str, tuple[str, ...]]:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InvalidValueBankError(
                f"cannot read valid JSON from Value Bank file {file_path}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise InvalidValueBankError(
                f"Value Bank file {file_path} must contain a JSON object"
            )
        if payload.get("version") != _SUPPORTED_VERSION:
            raise InvalidValueBankError(
                f"Value Bank file {file_path} has unsupported version "
                f"{payload.get('version')!r}; expected {_SUPPORTED_VERSION}"
            )
        raw_classes = payload.get("entity_values")
        if not isinstance(raw_classes, Mapping):
            raise InvalidValueBankError(
                f"Value Bank file {file_path} requires an entity_values object"
            )

        bank: Dict[str, tuple[str, ...]] = {}
        for raw_label, raw_items in raw_classes.items():
            if not isinstance(raw_label, str) or not _CLASS_NAME.fullmatch(raw_label):
                raise InvalidValueBankError(
                    f"Value Bank file {file_path} has invalid class name {raw_label!r}"
                )
            if not isinstance(raw_items, list):
                raise InvalidValueBankError(
                    f"Value Bank class {raw_label!r} in {file_path} must be an array"
                )
            if not raw_items:
                raise ValueBankEmptyClassError(
                    f"Value Bank class {raw_label!r} for language "
                    f"{language!r} has no values"
                )

            values: list[str] = []
            for index, item in enumerate(raw_items):
                if not isinstance(item, Mapping):
                    raise InvalidValueBankError(
                        f"Value Bank item {raw_label}[{index}] in {file_path} "
                        "must be an object"
                    )
                value = item.get("value")
                locale = item.get("locale")
                if not isinstance(value, str) or not value.strip():
                    raise InvalidValueBankError(
                        f"Value Bank item {raw_label}[{index}] in {file_path} "
                        "requires a non-empty string value"
                    )
                if locale != language:
                    raise InvalidValueBankError(
                        f"Value Bank item {raw_label}[{index}] in {file_path} "
                        f"has locale {locale!r}; expected {language!r}"
                    )
                values.append(value)
            if not values:
                raise ValueBankEmptyClassError(
                    f"Value Bank class {raw_label!r} for language "
                    f"{language!r} has no usable values"
                )
            bank[raw_label] = tuple(values)
        return bank

    @staticmethod
    def _normalise_language(language: str) -> str:
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
        normalised = aliases.get(str(language).strip().casefold(), str(language).strip().casefold())
        if not _LANGUAGE.fullmatch(normalised):
            raise ValueBankLanguageError(
                f"unsupported Value Bank language code: {language!r}"
            )
        return normalised
