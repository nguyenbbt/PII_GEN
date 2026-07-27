from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Sequence

from ..domain.models import FormattedEntity, FormattedSample, GeneratedEntity


_ENTITY_TAG = re.compile(r"<([A-Z][A-Z0-9_]*)>([^<>]+)</\1>", re.DOTALL)
_ANY_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9_]*>")
_UNSAFE_FILENAME = re.compile(r"[^\w.-]+", re.UNICODE)


class OutputFormatter:
    """Convert validated XML-like annotations to end-exclusive Unicode spans."""

    def format(
        self,
        *,
        tagged_text: str,
        entities: Sequence[GeneratedEntity | dict[str, str]],
        allowed_labels: Sequence[str],
    ) -> FormattedSample:
        if not tagged_text.strip():
            raise ValueError("tagged_text must be non-empty")
        metadata = [
            entity.dict() if isinstance(entity, GeneratedEntity) else entity
            for entity in entities
        ]
        allowed = set(allowed_labels)
        clean_parts: list[str] = []
        formatted_entities: list[FormattedEntity] = []
        tagged_pairs: list[tuple[str, str]] = []
        clean_length = 0
        cursor = 0

        for match in _ENTITY_TAG.finditer(tagged_text):
            prefix = tagged_text[cursor:match.start()]
            if _ANY_TAG.search(prefix):
                raise ValueError("tagged_text contains nested, unknown, or malformed tags")
            clean_parts.append(prefix)
            clean_length += len(prefix)

            label, value = match.group(1), match.group(2)
            if label not in allowed:
                raise ValueError(f"formatter encountered an unknown label: {label}")
            if not value:
                raise ValueError("formatted entity text cannot be empty")
            start = clean_length
            end = start + len(value)
            formatted_entities.append(
                FormattedEntity(label=label, start=start, end=end, text=value)
            )
            tagged_pairs.append((label, value))
            clean_parts.append(value)
            clean_length = end
            cursor = match.end()

        tail = tagged_text[cursor:]
        if _ANY_TAG.search(tail):
            raise ValueError("tagged_text contains nested, unknown, or malformed tags")
        clean_parts.append(tail)
        source_text = "".join(clean_parts)

        metadata_pairs = Counter(
            (str(item.get("label", "")), str(item.get("value", "")))
            for item in metadata
        )
        if metadata_pairs != Counter(tagged_pairs):
            raise ValueError("entity metadata must correspond exactly to tagged spans")

        sample = FormattedSample(
            entities=sorted(formatted_entities, key=lambda item: item.start),
            text=source_text,
        )
        for entity in sample.entities:
            if sample.text[entity.start:entity.end] != entity.text:
                raise ValueError("formatter produced an invalid entity offset")
        return sample


class JsonDatasetWriter:
    """Atomically maintain a partial dataset and publish it when a run completes."""

    def __init__(self, base_directory: Path | str = Path("gen_data")) -> None:
        self.base_directory = Path(base_directory)

    def write_partial(
        self,
        *,
        run_name: str,
        run_id: str,
        samples: Sequence[FormattedSample],
    ) -> Path:
        path = self._path(run_name, run_id, partial=True)
        self._atomic_write(path, samples)
        return path

    def finalize(
        self,
        *,
        run_name: str,
        run_id: str,
        samples: Sequence[FormattedSample],
    ) -> Path:
        partial_path = self._path(run_name, run_id, partial=True)
        final_path = self._path(run_name, run_id, partial=False)
        self._atomic_write(partial_path, samples)
        partial_path.replace(final_path)
        return final_path

    def _path(self, run_name: str, run_id: str, *, partial: bool) -> Path:
        safe_name = _UNSAFE_FILENAME.sub("_", run_name).strip("._") or "pii-run"
        suffix = ".partial.json" if partial else ".json"
        return self.base_directory / f"{safe_name}-{run_id}{suffix}"

    def _atomic_write(self, path: Path, samples: Sequence[FormattedSample]) -> None:
        self.base_directory.mkdir(parents=True, exist_ok=True)
        payload = [sample.dict() for sample in samples]
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.base_directory,
                prefix=".pii-dataset-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                json.dump(payload, temporary, ensure_ascii=False, indent=2)
                temporary.write("\n")
                temporary_path = Path(temporary.name)
            temporary_path.replace(path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
