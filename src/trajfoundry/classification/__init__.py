"""Independent trajectory classification primitives."""

from .classifier import (
    CLASSIFICATION_POLICY,
    CLASSIFIER_REVISION,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_SEMANTIC_RETRIES,
    MAX_CONTEXT_REDUCTIONS,
    PROMPT_VERSION,
    TrajectoryClassifier,
    has_tool_calls,
)
from .client import ChatCompletionsClient
from .input_projection import (
    PROJECTION_NAME,
    TRUNCATION_MARKER,
    UserInputProjection,
    project_user_input,
    serialize_for_model,
)
from .input_projection import TRUNCATION_STRATEGY as USER_INPUT_TRUNCATION_STRATEGY
from .input_projection import VERSION as PROJECTION_VERSION
from .taxonomy import GENERIC_SUBJECT_LABEL, ScenarioTaxonomy

__all__ = [
    "CLASSIFICATION_POLICY",
    "CLASSIFIER_REVISION",
    "DEFAULT_MAX_CONTEXT_CHARS",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_SEMANTIC_RETRIES",
    "GENERIC_SUBJECT_LABEL",
    "MAX_CONTEXT_REDUCTIONS",
    "PROJECTION_NAME",
    "PROJECTION_VERSION",
    "PROMPT_VERSION",
    "TRUNCATION_MARKER",
    "USER_INPUT_TRUNCATION_STRATEGY",
    "ChatCompletionsClient",
    "ScenarioTaxonomy",
    "TrajectoryClassifier",
    "UserInputProjection",
    "has_tool_calls",
    "project_user_input",
    "serialize_for_model",
]
