"""Trajectory-level scenario and capability classification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

import orjson

from .client import (
    DEFAULT_MAX_OUTPUT_TOKENS,
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
# Bump when the classification contract, model budget, or publication
# behavior changes so metadata and the persistent cache cannot silently mix
# runs.
CLASSIFIER_REVISION = "2026-09-29.1"
PROMPT_VERSION = "v001"
# The model documentation advertises a 256K-token context window.  The
# trajectory cap below remains a character budget because no Atria tokenizer
# is available in this runtime; it is a conservative envelope for current
# data, not an exact tokenizer-level guarantee.
CLASSIFIER_CONTEXT_WINDOW_TOKENS = 256_000
TRUNCATION_STRATEGY = "head_tail_char_v1"
# This is the serialized trajectory budget in characters, not model tokens.
DEFAULT_MAX_CONTEXT_CHARS = 600_000

# Keep label-policy changes visible to the persistent cache through an
# independent fingerprint. The policy is intentionally data rather than a
# version string so its digest changes automatically when a contract limit is
# edited.
CLASSIFICATION_POLICY = {
    "scenario_selection": "exactly_one_l1",
    "scenario_key_field": "scenario_label_key",
    "scenario_output_fields": (
        "key",
        "split",
        "domain_l1_en",
        "domain_l1_zh",
    ),
    "capability_min": 1,
    "capability_max": 2,
}
CLASSIFICATION_POLICY_SHA256 = hashlib.sha256(
    orjson.dumps(CLASSIFICATION_POLICY, option=orjson.OPT_SORT_KEYS)
).hexdigest()
# Short aliases are useful to callers building diagnostics without depending
# on the internal spelling of the policy constant.
POLICY_SHA256 = CLASSIFICATION_POLICY_SHA256


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
            "catalog_sha256": self.catalog_sha256,
            "policy_sha256": self.policy_sha256,
            "prompt_sha256": self.prompt_sha256,
            "strategy_fingerprint": self.strategy_fingerprint,
            "model": self.client.model,
            "api_url": getattr(self.client, "api_url", "injected-client"),
            "max_context_chars": self.max_context_chars,
            "context_window_tokens": CLASSIFIER_CONTEXT_WINDOW_TOKENS,
            "max_output_tokens": getattr(
                self.client, "max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS
            ),
            "truncation_strategy": TRUNCATION_STRATEGY,
            "temperature": 0,
            "response_format": "json_object",
        }
        return hashlib.sha256(
            orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
        ).hexdigest()

    @property
    def catalog_sha256(self) -> str:
        """Digest of the selectable, derived L1 catalog."""

        return self.taxonomy.l1_sha256

    @property
    def policy_sha256(self) -> str:
        """Digest of the output and selection policy."""

        return CLASSIFICATION_POLICY_SHA256

    @property
    def prompt_sha256(self) -> str:
        """Digest of the exact system prompt sent to the model."""

        return hashlib.sha256(self._system_prompt.encode("utf-8")).hexdigest()

    @property
    def strategy_fingerprint(self) -> str:
        """Digest tying policy, catalog, and prompt behavior together."""

        payload = {
            "policy_sha256": self.policy_sha256,
            "catalog_sha256": self.catalog_sha256,
            "prompt_sha256": self.prompt_sha256,
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
            "catalog_sha256": self.catalog_sha256,
            "policy_sha256": self.policy_sha256,
            "prompt_sha256": self.prompt_sha256,
            "strategy_fingerprint": self.strategy_fingerprint,
            "config_hash": self.config_hash,
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
                "scenario_labels": [
                    self.taxonomy.expand_l1(decision["scenario_label_key"])
                ],
                "capability_labels": decision["capability_labels"],
                **base,
            },
            cacheable=True,
        )

    def _parse_decision(self, payload: str) -> dict[str, Any]:
        try:
            value = orjson.loads(payload)
        except orjson.JSONDecodeError as error:
            raise ModelDecisionError("model output is not JSON") from error
        if type(value) is not dict or set(value) != {
            "scenario_label_key",
            "capability_labels",
        }:
            raise ModelDecisionError("model output has unsupported fields")
        key = value["scenario_label_key"]
        capabilities = value["capability_labels"]
        if type(key) is not str or not key:
            raise ModelDecisionError("scenario_label_key must be a non-empty string")
        self.taxonomy.expand_l1(key)
        if type(capabilities) is not list or not capabilities:
            raise ModelDecisionError("capability_labels must be non-empty")
        if len(capabilities) > 2:
            raise ModelDecisionError(
                "capability_labels must contain at most two labels"
            )
        if any(type(label) is not str for label in capabilities):
            raise ModelDecisionError("capability_labels must contain strings")
        if len(set(capabilities)) != len(capabilities):
            raise ModelDecisionError("capability_labels must be unique")
        if any(label not in CAPABILITY_LABELS for label in capabilities):
            raise ModelDecisionError("capability_labels contains an unknown label")
        return {
            "scenario_label_key": key,
            "capability_labels": capabilities,
        }

    def _build_system_prompt(self) -> str:
        capabilities = orjson.dumps(CAPABILITY_LABELS).decode("utf-8")
        return (
            "You classify complete agent trajectories. Return exactly one JSON "
            "object with two fields: scenario_label_key (exactly one string key "
            "from the L1 catalog) and capability_labels (an array of one or two "
            "unique strings from the allowed capability list). "
            "Choose the single scenario that best matches the final task. "
            "Choose the smallest sufficient capability set supported by direct "
            "evidence; do not infer capabilities from the domain alone. Do not "
            "return prose, markdown, model names, or harness names.\n\n"
            f"ALLOWED_CAPABILITIES={capabilities}\n\n"
            f"SCENARIO_CATALOG={self.taxonomy.prompt_catalog()}"
        )


__all__ = [
    "CAPABILITY_LABELS",
    "CLASSIFICATION_POLICY",
    "CLASSIFICATION_POLICY_SHA256",
    "CLASSIFIER_CONTEXT_WINDOW_TOKENS",
    "CLASSIFIER_REVISION",
    "DEFAULT_MAX_CONTEXT_CHARS",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "POLICY_SHA256",
    "PROMPT_VERSION",
    "TRUNCATION_STRATEGY",
    "ClassificationAttempt",
    "CompletionClient",
    "ModelDecisionError",
    "TrajectoryClassifier",
]
