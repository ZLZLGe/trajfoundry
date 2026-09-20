from __future__ import annotations

import io
import urllib.error

import orjson
import pytest

from trajfoundry.classification.classifier import TrajectoryClassifier
from trajfoundry.classification.client import (
    ChatCompletionsClient,
    ClassificationConfigurationError,
    ClassificationRetryExhausted,
)
from trajfoundry.classification.taxonomy import ScenarioTaxonomy


class _Completion:
    model = "classifier-test"

    def __init__(self, response: object) -> None:
        self.response = response
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.messages.append(messages)
        return orjson.dumps(self.response).decode()


def _taxonomy(tmp_path) -> ScenarioTaxonomy:
    path = tmp_path / "taxonomy.json"
    path.write_bytes(
        orjson.dumps(
            [
                {
                    "id": 10026,
                    "split": "toc",
                    "domain_l1_en": "Shopping",
                    "domain_l1_zh": "购物",
                    "domain_l2_en": "General E-commerce",
                    "domain_l2_zh": "综合电商",
                    "domain_path_en": "Shopping > General E-commerce",
                    "domain_path_zh": "购物->综合电商",
                }
            ]
        )
    )
    return ScenarioTaxonomy.load(path)


def test_classifier_expands_taxonomy_and_copies_deterministic_labels(tmp_path) -> None:
    client = _Completion(
        {
            "scenario_label_ids": [10026],
            "capability_labels": ["Tool Use", "Information Gathering"],
        }
    )
    classifier = TrajectoryClassifier(
        client=client,
        taxonomy=_taxonomy(tmp_path),
        input_manifest_sha256="a" * 64,
    )

    attempt = classifier.classify(
        {"messages": [], "model": "gpt-test", "harness": "codex"}
    )

    assert attempt.cacheable
    assert attempt.classification["status"] == "accepted"
    assert attempt.classification["scenario_labels"][0]["id"] == 10026
    assert attempt.classification["capability_labels"] == [
        "Tool Use",
        "Information Gathering",
    ]
    assert attempt.classification["model_label"] == "gpt-test"
    assert attempt.classification["harness_label"] == "codex"
    assert "10026" in client.messages[0][0]["content"]


def test_classifier_marks_unknown_model_labels_as_failed(tmp_path) -> None:
    classifier = TrajectoryClassifier(
        client=_Completion(
            {
                "scenario_label_ids": [99999],
                "capability_labels": ["Tool Use"],
            }
        ),
        taxonomy=_taxonomy(tmp_path),
        input_manifest_sha256="a" * 64,
    )

    attempt = classifier.classify({"messages": []})

    assert not attempt.cacheable
    assert attempt.classification["status"] == "failed"
    assert attempt.classification["reason"] == "invalid_model_output"
    assert attempt.classification["model_label"] == "unknown"
    assert attempt.classification["harness_label"] == "unknown"


def test_chat_client_retries_429_without_exposing_key() -> None:
    attempts = 0
    sleeps: list[float] = []

    def opener(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(
            "https://classifier.invalid/v1/chat/completions",
            429,
            "limited",
            {},
            io.BytesIO(b'{"secret":"must not surface"}'),
        )

    client = ChatCompletionsClient(
        api_url="https://classifier.invalid/v1/chat/completions",
        model="model",
        api_key="sk-do-not-log-this",
        max_retries=1,
        opener=opener,
        sleeper=sleeps.append,
        jitter=lambda: 0.0,
    )

    with pytest.raises(ClassificationRetryExhausted) as caught:
        client.complete([{"role": "user", "content": "hello"}])

    assert caught.value.reason == "rate_limit_exhausted"
    assert attempts == 2
    assert sleeps == [0.5]
    assert "sk-do-not-log-this" not in repr(client)
    assert "must not surface" not in str(caught.value)


def test_chat_client_treats_404_as_configuration_error() -> None:
    def opener(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "https://classifier.invalid/v1/chat/completions",
            404,
            "missing",
            {},
            io.BytesIO(b"not found"),
        )

    client = ChatCompletionsClient(
        api_url="https://classifier.invalid/v1/chat/completions",
        model="model",
        api_key="secret",
        opener=opener,
    )

    with pytest.raises(ClassificationConfigurationError, match="HTTP 404"):
        client.complete([{"role": "user", "content": "hello"}])


def test_chat_client_retries_all_5xx_statuses() -> None:
    attempts = 0

    def opener(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(
            "https://classifier.invalid/v1/chat/completions",
            599,
            "upstream failure",
            {},
            io.BytesIO(b"upstream failure"),
        )

    client = ChatCompletionsClient(
        api_url="https://classifier.invalid/v1/chat/completions",
        model="model",
        api_key="secret",
        max_retries=1,
        opener=opener,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(ClassificationRetryExhausted) as caught:
        client.complete([{"role": "user", "content": "hello"}])

    assert caught.value.reason == "service_retry_exhausted"
    assert attempts == 2
