"""Trajectory-level scenario and capability classification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

import orjson

from .client import (
    ClassificationConfigurationError,
    ClassificationRequestError,
    ClassificationRetryExhausted,
)
from .taxonomy import ScenarioTaxonomy, TaxonomyError

CAPABILITY_LABELS = (
    "Task Understanding",
    "Information Gathering",
    "Planning & Decision Making",
    "State Management",
    "Tool Use",
    "Code & Programmatic Operations",
    "Data Analysis",
    "Office & Document Handling",
    "Interactive Collaboration",
    "Reliability & Safety",
)
# Bump when the classification contract or publication behavior changes so
# scheduler assertions and the persistent cache cannot silently mix runs.
CLASSIFIER_REVISION = "2026-09-22.1"
PROMPT_VERSION = "v001"
DEFAULT_MAX_CONTEXT_CHARS = 500_000


class CompletionClient(Protocol):
    model: str

    def complete(self, messages: list[dict[str, str]]) -> str: ...


class ModelDecisionError(ValueError):
    """The model returned syntactically or semantically invalid labels."""


@dataclass(frozen=True, slots=True)
class ClassificationAttempt:
    classification: dict[str, Any]
    cacheable: bool


def _display_value(value: object) -> str:
    return value if isinstance(value, str) and value else "unknown"


def _model_context(
    trajectory: dict[str, Any], max_context_chars: int
) -> tuple[str, bool]:
    payload = orjson.dumps(trajectory, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    if len(payload) <= max_context_chars:
        return payload, False
    marker = '\n"[...TRAJECTORY_CONTEXT_TRUNCATED...]"\n'
    available = max_context_chars - len(marker)
    if available <= 0:
        raise ValueError("max_context_chars is too small")
    leading = available // 2
    trailing = available - leading
    return f"{payload[:leading]}{marker}{payload[-trailing:]}", True


class TrajectoryClassifier:
    """Classify one complete root trajectory using an external model."""

    def __init__(
        self,
        *,
        client: CompletionClient,
        taxonomy: ScenarioTaxonomy,
        input_manifest_sha256: str,
        classifier_revision: str = CLASSIFIER_REVISION,
        prompt_version: str = PROMPT_VERSION,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> None:
        if len(input_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in input_manifest_sha256
        ):
            raise ValueError("input_manifest_sha256 must be a lowercase SHA-256")
        if not classifier_revision or not prompt_version:
            raise ValueError("classifier and prompt versions must not be empty")
        if max_context_chars < 1024:
            raise ValueError("max_context_chars must be at least 1024")
        self.client = client
        self.taxonomy = taxonomy
        self.input_manifest_sha256 = input_manifest_sha256
        self.classifier_revision = classifier_revision
        self.prompt_version = prompt_version
        self.max_context_chars = max_context_chars
        self._system_prompt = self._build_system_prompt()

    @property
    def config_hash(self) -> str:
        payload = {
            "classifier_revision": self.classifier_revision,
            "prompt_version": self.prompt_version,
            "taxonomy_sha256": self.taxonomy.sha256,
            "model": self.client.model,
            "api_url": getattr(self.client, "api_url", "injected-client"),
            "max_context_chars": self.max_context_chars,
        }
        return hashlib.sha256(
            orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
        ).hexdigest()

    def classify(self, trajectory: dict[str, Any]) -> ClassificationAttempt:
        if type(trajectory) is not dict:
            raise TypeError("trajectory must be an object")
        context, truncated = _model_context(trajectory, self.max_context_chars)
        base = {
            "model_label": _display_value(trajectory.get("model")),
            "harness_label": _display_value(trajectory.get("harness")),
            "classifier_revision": self.classifier_revision,
            "prompt_version": self.prompt_version,
            "taxonomy_sha256": self.taxonomy.sha256,
            "input_manifest_sha256": self.input_manifest_sha256,
            "context_truncated": truncated,
        }
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": (
                    "Classify this complete root trajectory. Embedded sub-agent "
                    "trajectories are context for the same classification.\n\n"
                    f"TRAJECTORY:\n{context}"
                ),
            },
        ]
        try:
            response = self.client.complete(messages)
            decision = self._parse_decision(response)
        except ClassificationConfigurationError:
            raise
        except ClassificationRetryExhausted as error:
            return ClassificationAttempt(
                classification={"status": "failed", "reason": error.reason, **base},
                cacheable=False,
            )
        except ClassificationRequestError as error:
            return ClassificationAttempt(
                classification={"status": "failed", "reason": error.reason, **base},
                cacheable=False,
            )
        except (ModelDecisionError, TaxonomyError, orjson.JSONDecodeError):
            return ClassificationAttempt(
                classification={
                    "status": "failed",
                    "reason": "invalid_model_output",
                    **base,
                },
                cacheable=False,
            )

        return ClassificationAttempt(
            classification={
                "status": "accepted",
                "scenario_labels": self.taxonomy.expand(decision["scenario_label_ids"]),
                "capability_labels": decision["capability_labels"],
                **base,
            },
            cacheable=True,
        )

    def _parse_decision(self, payload: str) -> dict[str, list[Any]]:
        try:
            value = orjson.loads(payload)
        except orjson.JSONDecodeError as error:
            raise ModelDecisionError("model output is not JSON") from error
        if type(value) is not dict or set(value) != {
            "scenario_label_ids",
            "capability_labels",
        }:
            raise ModelDecisionError("model output has unsupported fields")
        identifiers = value["scenario_label_ids"]
        capabilities = value["capability_labels"]
        if type(identifiers) is not list or not identifiers:
            raise ModelDecisionError("scenario_label_ids must be non-empty")
        if any(type(identifier) is not int for identifier in identifiers):
            raise ModelDecisionError("scenario_label_ids must contain integers")
        if len(set(identifiers)) != len(identifiers):
            raise ModelDecisionError("scenario_label_ids must be unique")
        self.taxonomy.expand(identifiers)
        if type(capabilities) is not list or not capabilities:
            raise ModelDecisionError("capability_labels must be non-empty")
        if any(type(label) is not str for label in capabilities):
            raise ModelDecisionError("capability_labels must contain strings")
        if len(set(capabilities)) != len(capabilities):
            raise ModelDecisionError("capability_labels must be unique")
        if any(label not in CAPABILITY_LABELS for label in capabilities):
            raise ModelDecisionError("capability_labels contains an unknown label")
        return {
            "scenario_label_ids": identifiers,
            "capability_labels": capabilities,
        }

    def _build_system_prompt(self) -> str:
        capabilities = orjson.dumps(CAPABILITY_LABELS).decode("utf-8")
        return (
            "You classify complete agent trajectories. Return exactly one JSON "
            "object with two fields: scenario_label_ids (a non-empty array of "
            "unique integer IDs from the catalog) and capability_labels (a "
            "non-empty array of unique strings from the allowed capability list). "
            "Choose only labels supported by the trajectory. Do not return prose, "
            "markdown, model names, or harness names.\n\n"
            f"ALLOWED_CAPABILITIES={capabilities}\n\n"
            f"SCENARIO_CATALOG={self.taxonomy.prompt_catalog()}"
        )


__all__ = [
    "CAPABILITY_LABELS",
    "CLASSIFIER_REVISION",
    "DEFAULT_MAX_CONTEXT_CHARS",
    "PROMPT_VERSION",
    "ClassificationAttempt",
    "CompletionClient",
    "ModelDecisionError",
    "TrajectoryClassifier",
]
