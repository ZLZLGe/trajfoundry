"""Load the closed scenario, capability, and subject catalogs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

DEFAULT_TAXONOMY_PATH = Path(__file__).with_name("taxonomy.json")
TAXONOMY_SCHEMA_VERSION = "trajfoundry-taxonomy-v2"
GENERIC_SUBJECT_LABEL = "通用（非学科）"


class TaxonomyError(ValueError):
    """A taxonomy or a selected label is invalid."""


def _catalog(value: object, *, scenario: bool = False) -> tuple[dict[str, Any], ...]:
    if type(value) is not dict or not value:
        raise TaxonomyError("each taxonomy catalog must be a non-empty object")
    result = []
    for name, raw in sorted(value.items()):
        if type(name) is not str or not name or name.strip() != name:
            raise TaxonomyError("label names must be non-empty, trimmed strings")
        required = {"描述", "典型任务"} | ({"分组"} if scenario else set())
        if type(raw) is not dict or set(raw) != required:
            raise TaxonomyError(f"label {name!r} has unsupported fields")
        description = raw["描述"]
        tasks = raw["典型任务"]
        if type(description) is not str or not description.strip():
            raise TaxonomyError(f"label {name!r} needs a description")
        if (
            type(tasks) is not list
            or not tasks
            or any(type(task) is not str or not task.strip() for task in tasks)
        ):
            raise TaxonomyError(f"label {name!r} needs non-empty typical tasks")
        item = {"name": name, "description": description, "typical_tasks": tasks}
        if scenario:
            group = raw["分组"]
            if type(group) is not str or group not in {"toc", "tob", "toe"}:
                raise TaxonomyError(f"label {name!r} has an invalid group")
            item["group"] = group
        result.append(item)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ScenarioTaxonomy:
    """Content-addressed catalogs used together for one classification."""

    scenario_catalog: tuple[dict[str, Any], ...]
    capability_catalog: tuple[dict[str, Any], ...]
    subject_catalog: tuple[dict[str, Any], ...]
    sha256: str

    @classmethod
    def load(cls, path: str | Path = DEFAULT_TAXONOMY_PATH) -> ScenarioTaxonomy:
        try:
            value = orjson.loads(Path(path).expanduser().read_bytes())
        except (OSError, orjson.JSONDecodeError) as error:
            raise TaxonomyError(
                "could not read a valid taxonomy JSON document"
            ) from error
        fields = {"schema_version", "scenarios", "capabilities", "subjects"}
        if type(value) is not dict or set(value) != fields:
            raise TaxonomyError("taxonomy must contain the three v2 catalogs")
        if value["schema_version"] != TAXONOMY_SCHEMA_VERSION:
            raise TaxonomyError("unsupported taxonomy schema_version")
        scenarios = _catalog(value["scenarios"], scenario=True)
        capabilities = _catalog(value["capabilities"])
        subjects = _catalog(value["subjects"])
        if GENERIC_SUBJECT_LABEL not in {item["name"] for item in subjects}:
            raise TaxonomyError(f"subjects must include {GENERIC_SUBJECT_LABEL!r}")
        return cls(
            scenario_catalog=scenarios,
            capability_catalog=capabilities,
            subject_catalog=subjects,
            sha256=hashlib.sha256(
                orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
            ).hexdigest(),
        )

    @property
    def scenario_names(self) -> tuple[str, ...]:
        return tuple(item["name"] for item in self.scenario_catalog)

    @property
    def capability_names(self) -> tuple[str, ...]:
        return tuple(item["name"] for item in self.capability_catalog)

    @property
    def subject_names(self) -> tuple[str, ...]:
        return tuple(item["name"] for item in self.subject_catalog)

    def prompt_catalog(self) -> str:
        return orjson.dumps(
            {
                "scenarios": self.scenario_catalog,
                "capabilities": self.capability_catalog,
                "subjects": self.subject_catalog,
            },
            option=orjson.OPT_SORT_KEYS,
        ).decode("utf-8")


__all__ = [
    "DEFAULT_TAXONOMY_PATH",
    "GENERIC_SUBJECT_LABEL",
    "TAXONOMY_SCHEMA_VERSION",
    "ScenarioTaxonomy",
    "TaxonomyError",
]
