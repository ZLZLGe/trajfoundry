"""Independent trajectory classification primitives."""

from .classifier import (
    CAPABILITY_LABELS,
    CLASSIFICATION_POLICY,
    CLASSIFICATION_POLICY_SHA256,
    CLASSIFIER_CONTEXT_WINDOW_TOKENS,
    CLASSIFIER_REVISION,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_SEMANTIC_RETRIES,
    MAX_CONTEXT_REDUCTIONS,
    POLICY_SHA256,
    PROMPT_VERSION,
    TrajectoryClassifier,
)
from .client import ChatCompletionsClient
from .input_projection import (
    PROJECTION_NAME,
    TRUNCATION_MARKER,
    UserInputProjection,
    project_user_input,
    serialize_for_model,
)
from .input_projection import (
    TRUNCATION_STRATEGY as USER_INPUT_TRUNCATION_STRATEGY,
)
from .input_projection import (
    VERSION as PROJECTION_VERSION,
)
from .taxonomy import (
    L1_KEY_SEPARATOR,
    L1_TAXONOMY_FIELDS,
    ScenarioTaxonomy,
    make_l1_key,
)

__all__ = [
    "CAPABILITY_LABELS",
    "CLASSIFICATION_POLICY",
    "CLASSIFICATION_POLICY_SHA256",
    "CLASSIFIER_CONTEXT_WINDOW_TOKENS",
    "CLASSIFIER_REVISION",
    "DEFAULT_MAX_CONTEXT_CHARS",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_SEMANTIC_RETRIES",
    "L1_KEY_SEPARATOR",
    "L1_TAXONOMY_FIELDS",
    "MAX_CONTEXT_REDUCTIONS",
    "POLICY_SHA256",
    "PROJECTION_NAME",
    "PROJECTION_VERSION",
    "PROMPT_VERSION",
    "TRUNCATION_MARKER",
    "USER_INPUT_TRUNCATION_STRATEGY",
    "ChatCompletionsClient",
    "ScenarioTaxonomy",
    "TrajectoryClassifier",
    "UserInputProjection",
    "make_l1_key",
    "project_user_input",
    "serialize_for_model",
]
