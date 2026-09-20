"""Versioned scenario taxonomy loading and strict label expansion."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import orjson

TAXONOMY_FIELDS = (
    "id",
    "split",
    "domain_l1_en",
    "domain_l1_zh",
    "domain_l2_en",
    "domain_l2_zh",
    "domain_path_en",
    "domain_path_zh",
)
DEFAULT_TAXONOMY_PATH = Path(__file__).with_name("taxonomy.json")


class TaxonomyError(ValueError):
    """Raised when a taxonomy or a model-selected label is invalid."""


@dataclass(frozen=True, slots=True)
class ScenarioTaxonomy:
    """An immutable, content-addressed scenario taxonomy."""

    labels: tuple[dict[str, str | int], ...]
    sha256: str
    _by_id: dict[int, dict[str, str | int]]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_TAXONOMY_PATH) -> ScenarioTaxonomy:
        source = Path(path).expanduser()
        try:
            payload = source.read_bytes()
        except OSError as error:
            raise TaxonomyError(f"could not read taxonomy: {source}") from error
        try:
            value = orjson.loads(payload)
        except orjson.JSONDecodeError as error:
            raise TaxonomyError("taxonomy must be valid JSON") from error
        if type(value) is not list or not value:
            raise TaxonomyError("taxonomy must be a non-empty JSON array")

        labels: list[dict[str, str | int]] = []
        by_id: dict[int, dict[str, str | int]] = {}
        required = set(TAXONOMY_FIELDS)
        for index, raw in enumerate(value):
            if type(raw) is not dict or set(raw) != required:
                raise TaxonomyError(
                    f"taxonomy item {index} must contain exactly the supported fields"
                )
            identifier = raw["id"]
            if type(identifier) is not int or identifier < 0:
                raise TaxonomyError(f"taxonomy item {index} has an invalid id")
            if identifier in by_id:
                raise TaxonomyError(f"taxonomy contains duplicate id {identifier}")
            split = raw["split"]
            if split not in {"tob", "toc", "toe"}:
                raise TaxonomyError(f"taxonomy item {index} has an invalid split")
            for field in TAXONOMY_FIELDS[2:]:
                if type(raw[field]) is not str or not raw[field]:
                    raise TaxonomyError(f"taxonomy item {index} has an invalid {field}")
            label = {field: raw[field] for field in TAXONOMY_FIELDS}
            labels.append(label)
            by_id[identifier] = label

        canonical = orjson.dumps(labels, option=orjson.OPT_SORT_KEYS)
        return cls(
            labels=tuple(labels),
            sha256=hashlib.sha256(canonical).hexdigest(),
            _by_id=by_id,
        )

    def expand(self, identifiers: list[int]) -> list[dict[str, str | int]]:
        """Expand unique IDs in model order into canonical taxonomy objects."""

        if not identifiers:
            raise TaxonomyError("scenario_label_ids must contain at least one id")
        expanded: list[dict[str, str | int]] = []
        seen: set[int] = set()
        for identifier in identifiers:
            if type(identifier) is not int:
                raise TaxonomyError("scenario label ids must be integers")
            if identifier in seen:
                raise TaxonomyError("scenario label ids must not contain duplicates")
            seen.add(identifier)
            try:
                label = self._by_id[identifier]
            except KeyError as error:
                raise TaxonomyError(
                    f"unknown scenario taxonomy id: {identifier}"
                ) from error
            expanded.append(dict(label))
        return expanded

    def prompt_catalog(self) -> str:
        """Return a compact catalog containing only information needed to choose IDs."""

        compact = [
            {
                "id": label["id"],
                "split": label["split"],
                "path_en": label["domain_path_en"],
                "path_zh": label["domain_path_zh"],
            }
            for label in self.labels
        ]
        return orjson.dumps(compact).decode("utf-8")


__all__ = [
    "DEFAULT_TAXONOMY_PATH",
    "TAXONOMY_FIELDS",
    "ScenarioTaxonomy",
    "TaxonomyError",
]
