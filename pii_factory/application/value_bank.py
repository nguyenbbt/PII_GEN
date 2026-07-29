from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence


_CLASS_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_LANGUAGE = re.compile(r"^[a-z]{2}$")
_SUPPORTED_VERSION = 1


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

    Relative paths are resolved against the process working directory. Files are
    loaded lazily and cached after validation. Every source entry is retained,
    including case variants and intentional duplicates, so the configured bank
    remains the source of truth for sampling probabilities.
    """

    def __init__(
        self,
        value_bank_path: str | Path = "PII_Value_Bank",
        language_files: Mapping[str, str | Path] | None = None,
    ) -> None:
        path = Path(value_bank_path).expanduser()
        self.path = path if path.is_absolute() else Path.cwd() / path
        self.language_files = {
            self._normalise_language(language): Path(file_path).expanduser()
            for language, file_path in (language_files or {}).items()
        }
        self._banks: Dict[str, Dict[str, tuple[str, ...]]] = {}

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
        values = bank[normalised_label]
        if not values:
            raise ValueBankEmptyClassError(
                f"Value Bank class {normalised_label!r} for language "
                f"{normalised_language!r} has no values"
            )
        return values

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
