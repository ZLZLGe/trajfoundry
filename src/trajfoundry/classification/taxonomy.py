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
# The published classifier selects one first-level scenario label.  ``key`` is
# deliberately a readable composite rather than an integer derived from the
# order of the source taxonomy: it remains stable when L2 rows are reordered.
L1_TAXONOMY_FIELDS = (
    "key",
    "split",
    "domain_l1_en",
    "domain_l1_zh",
)
L1_KEY_SEPARATOR = "|"
DEFAULT_TAXONOMY_PATH = Path(__file__).with_name("taxonomy.json")


class TaxonomyError(ValueError):
    """Raised when a taxonomy or a model-selected label is invalid."""


@dataclass(frozen=True, slots=True)
class ScenarioTaxonomy:
    """An immutable, content-addressed scenario taxonomy."""

    labels: tuple[dict[str, str | int], ...]
    sha256: str
    _by_id: dict[int, dict[str, str | int]]
    l1_labels: tuple[dict[str, str], ...]
    l1_sha256: str
    _l1_by_key: dict[str, dict[str, str]]

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

        # Derive the L1 catalog from the validated L2 rows. Sort the unique
        # keys rather than relying on source row order, so reordering L2 rows
        # does not change the selectable L1 catalog or its digest.
        l1_values: dict[str, dict[str, str]] = {}
        for label in labels:
            split = str(label["split"])
            domain_l1_en = str(label["domain_l1_en"])
            domain_l1_zh = str(label["domain_l1_zh"])
            key = make_l1_key(split, domain_l1_en, domain_l1_zh)
            candidate = {
                "key": key,
                "split": split,
                "domain_l1_en": domain_l1_en,
                "domain_l1_zh": domain_l1_zh,
            }
            previous = l1_values.get(key)
            if previous is not None and previous != candidate:
                # This should be unreachable because the key contains every
                # field used by the candidate, but keeping the invariant here
                # makes future key-format changes fail closed.
                raise TaxonomyError(f"taxonomy has conflicting L1 key: {key}")
            l1_values[key] = candidate
        l1_labels = tuple(l1_values[key] for key in sorted(l1_values))
        l1_by_key = {label["key"]: label for label in l1_labels}

        canonical = orjson.dumps(labels, option=orjson.OPT_SORT_KEYS)
        l1_canonical = orjson.dumps(l1_labels, option=orjson.OPT_SORT_KEYS)
        return cls(
            labels=tuple(labels),
            sha256=hashlib.sha256(canonical).hexdigest(),
            _by_id=by_id,
            l1_labels=l1_labels,
            l1_sha256=hashlib.sha256(l1_canonical).hexdigest(),
            _l1_by_key=l1_by_key,
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
        """Return the L1 catalog used by the classifier prompt."""

        return self.l1_prompt_catalog()

    def expand_l1(self, key: str) -> dict[str, str]:
        """Return the canonical L1 object selected by its stable key."""

        if type(key) is not str or not key:
            raise TaxonomyError("scenario_label_key must be a non-empty string")
        try:
            label = self._l1_by_key[key]
        except KeyError as error:
            raise TaxonomyError(f"unknown scenario L1 taxonomy key: {key}") from error
        return dict(label)

    # Explicit aliases make the intent clear to callers that need to validate
    # a model-selected key and keep the lookup name discoverable.
    expand_l1_key = expand_l1

    def l1_prompt_catalog(self) -> str:
        """Return a compact, deterministic catalog of selectable L1 labels."""

        return orjson.dumps(self.l1_labels).decode("utf-8")


def make_l1_key(split: str, domain_l1_en: str, domain_l1_zh: str) -> str:
    """Build the canonical stable key for one L1 catalog entry.

    The taxonomy currently contains no separator characters in these fields;
    reject them explicitly so the key remains unambiguous if source data grows
    later.
    """

    values = (split, domain_l1_en, domain_l1_zh)
    if any(type(value) is not str or not value for value in values):
        raise TaxonomyError("L1 key fields must be non-empty strings")
    if any(L1_KEY_SEPARATOR in value for value in values):
        raise TaxonomyError(f"L1 key fields must not contain {L1_KEY_SEPARATOR!r}")
    return L1_KEY_SEPARATOR.join(values)


__all__ = [
    "DEFAULT_TAXONOMY_PATH",
    "L1_KEY_SEPARATOR",
    "L1_TAXONOMY_FIELDS",
    "TAXONOMY_FIELDS",
    "ScenarioTaxonomy",
    "TaxonomyError",
    "make_l1_key",
]
