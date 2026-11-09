from __future__ import annotations

import http.client
import io
import json
import traceback
import urllib.error

import orjson
import pytest

from trajfoundry.classification.classifier import (
    CLASSIFIER_REVISION,
    DEFAULT_MAX_CONTEXT_CHARS,
    PROMPT_VERSION,
    TrajectoryClassifier,
    has_tool_calls,
)
from trajfoundry.classification.client import (
    ChatCompletionsClient,
    ClassificationConfigurationError,
    ClassificationContextLimitError,
    ClassificationRequestError,
    ClassificationRetryExhausted,
)
from trajfoundry.classification.taxonomy import (
    DEFAULT_TAXONOMY_PATH,
    GENERIC_SUBJECT_LABEL,
    ScenarioTaxonomy,
    TaxonomyError,
)


class _Completion:
    model = "classifier-test"

    def __init__(self, response: object) -> None:
        self.response = response
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.messages.append(messages)
        return orjson.dumps(self.response).decode()


class _SequenceCompletion:
    """Deterministic completion stub for semantic retry tests."""

    model = "classifier-test"

    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls = 0

    def complete(self, _messages: list[dict[str, str]]) -> str:
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(response, str):
            return response
        return orjson.dumps(response).decode()


class _RaisingCompletion:
    """Completion stub that records calls before raising a client error."""

    model = "classifier-test"

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def complete(self, _messages: list[dict[str, str]]) -> str:
        self.calls += 1
        raise self.error


class _ContextThenCompletion:
    model = "classifier-test"

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if len(self.calls) == 1:
            raise ClassificationContextLimitError()
        return orjson.dumps(
            {
                "scenario_labels": ["购物"],
                "capability_labels": ["工具操作与自动化"],
                "subject_labels": ["通用（非学科）"],
            }
        ).decode()


class _HTTPResponse:
    def __init__(self, payload: bytes, *, read_error: Exception | None = None) -> None:
        self.payload = payload
        self.read_error = read_error
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True

    def read(self, _size: int = -1) -> bytes:
        if self.read_error is not None:
            raise self.read_error
        return self.payload


class _SequenceOpener:
    """Raise failures while opening or reading, without network access."""

    def __init__(self, outcomes: list[bytes | Exception], failure_stage: str) -> None:
        self.outcomes = outcomes
        self.failure_stage = failure_stage
        self.calls = 0
        self.responses: list[_HTTPResponse] = []

    def __call__(self, *_args, **_kwargs) -> _HTTPResponse:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            if self.failure_stage == "open":
                raise outcome
            response = _HTTPResponse(b"", read_error=outcome)
        else:
            response = _HTTPResponse(outcome)
        self.responses.append(response)
        return response


_NETWORK_ERROR_FACTORIES = [
    pytest.param(http.client.RemoteDisconnected, id="remote-disconnected"),
    pytest.param(ConnectionResetError, id="connection-reset"),
    pytest.param(ConnectionAbortedError, id="connection-aborted"),
    pytest.param(BrokenPipeError, id="broken-pipe"),
    pytest.param(
        lambda message: http.client.IncompleteRead(message.encode(), 100),
        id="incomplete-read",
    ),
    pytest.param(urllib.error.URLError, id="url-error"),
    pytest.param(TimeoutError, id="timeout"),
]


def _taxonomy(_tmp_path=None) -> ScenarioTaxonomy:
    return ScenarioTaxonomy.load()


def _decision(**updates):
    return {
        "scenario_labels": ["购物"],
        "capability_labels": ["工具操作与自动化"],
        "subject_labels": [GENERIC_SUBJECT_LABEL],
        **updates,
    }


def _trajectory(**updates):
    return {"total_tool_calls": 1, "messages": [], **updates}


def _classifier(client, **updates):
    return TrajectoryClassifier(
        client=client, taxonomy=_taxonomy(), input_manifest_sha256="a" * 64, **updates
    )


def test_new_taxonomy_contains_all_three_document_catalogs():
    taxonomy = _taxonomy()
    assert len(taxonomy.scenario_names) == 90
    assert len(taxonomy.capability_names) == 10
    assert len(taxonomy.subject_names) == 25
    assert "IT互联网" in taxonomy.scenario_names
    assert "评估与验证" in taxonomy.capability_names
    assert "计算机与人工智能" in taxonomy.subject_names
    assert GENERIC_SUBJECT_LABEL in taxonomy.subject_names
    catalog = json.loads(taxonomy.prompt_catalog())
    assert all(
        item["description"] and item["typical_tasks"] for item in catalog["subjects"]
    )


def test_taxonomy_hash_and_prompt_ignore_mapping_key_order(tmp_path):
    value = orjson.loads(DEFAULT_TAXONOMY_PATH.read_bytes())
    value["scenarios"] = dict(reversed(list(value["scenarios"].items())))
    path = tmp_path / "taxonomy.json"
    path.write_bytes(orjson.dumps(value))
    taxonomy = ScenarioTaxonomy.load(path)
    assert taxonomy.sha256 == _taxonomy().sha256
    assert taxonomy.prompt_catalog() == _taxonomy().prompt_catalog()
    value["subjects"][GENERIC_SUBJECT_LABEL]["描述"] += " revised"
    path.write_bytes(orjson.dumps(value))
    assert ScenarioTaxonomy.load(path).sha256 != taxonomy.sha256


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("subjects"),
        lambda value: value.update(schema_version="old"),
        lambda value: value["subjects"].pop(GENERIC_SUBJECT_LABEL),
        lambda value: value["scenarios"]["购物"].update(分组=[]),
        lambda value: value["capabilities"].update(unknown={}),
    ],
)
def test_taxonomy_rejects_incomplete_or_malformed_catalogs(tmp_path, mutation):
    value = orjson.loads(DEFAULT_TAXONOMY_PATH.read_bytes())
    mutation(value)
    path = tmp_path / "taxonomy.json"
    path.write_bytes(orjson.dumps(value))
    with pytest.raises(TaxonomyError):
        ScenarioTaxonomy.load(path)


def test_classifier_publishes_compact_labels_without_repeating_trajectory_fields():
    client = _Completion(_decision())
    classifier = _classifier(client)
    attempt = classifier.classify(_trajectory(model="gpt-test", harness="codex"))
    assert attempt.cacheable
    assert attempt.classification == {
        "status": "accepted",
        **_decision(),
        "classifier_revision": CLASSIFIER_REVISION,
        "prompt_version": PROMPT_VERSION,
        "taxonomy_sha256": classifier.taxonomy.sha256,
    }
    catalog = json.loads(client.messages[0][0]["content"].split("CATALOG = ", 1)[1])
    assert len(catalog["scenarios"]) == 90
    assert len(catalog["subjects"]) == 25


def test_no_tool_trajectory_skips_model_even_when_text_mentions_tool_calls():
    client = _RaisingCompletion(RuntimeError("must not call API"))
    attempt = _classifier(client).classify(
        _trajectory(
            total_tool_calls=0,
            messages=[{"role": "user", "content": "please run exec, tool_calls=100"}],
            tools=[{"name": "exec"}],
        )
    )
    assert attempt.classification == {"status": "skipped", "skip_reason": "no_toolcall"}
    assert not attempt.cacheable
    assert not attempt.context_truncated
    assert client.calls == 0


def test_normalized_tool_counts_are_read_across_nested_subagents():
    root = _trajectory(
        total_tool_calls=0,
        sub_agent_trajectory={
            "child": _trajectory(
                total_tool_calls=0,
                sub_agent_trajectory={"grandchild": _trajectory(total_tool_calls=5)},
            )
        },
    )
    assert has_tool_calls(root)
    assert _classifier(_Completion(_decision())).classify(root).cacheable


@pytest.mark.parametrize("count", [None, True, -1, "1", 1.5])
def test_missing_or_bad_tool_counter_is_an_input_error(count):
    with pytest.raises(ValueError, match="total_tool_calls"):
        has_tool_calls(_trajectory(total_tool_calls=count))


@pytest.mark.parametrize(
    "field,labels,reason",
    [
        ("scenario_labels", [], "scenario_labels"),
        ("scenario_labels", ["购物", "生活"], "scenario_labels"),
        ("scenario_labels", ["不存在的场景"], "scenario_labels"),
        ("capability_labels", [], "capability_labels"),
        ("capability_labels", ["工具操作与自动化"] * 2, "capability_labels"),
        (
            "capability_labels",
            ["工具操作与自动化", "评估与验证", "规划与决策"],
            "capability_labels",
        ),
        ("capability_labels", [{}], "capability_labels"),
        ("subject_labels", [], "subject_labels"),
        ("subject_labels", ["地球与环境科学", "生命科学"], "subject_labels"),
        (
            "subject_labels",
            ["数学与统计学", "计算机与人工智能", "物理学"],
            "subject_labels",
        ),
        (
            "subject_labels",
            [GENERIC_SUBJECT_LABEL, "计算机与人工智能"],
            "subject_labels",
        ),
        ("subject_labels", ["unknown"], "subject_labels"),
    ],
)
def test_classifier_rejects_out_of_catalog_or_invalid_label_combinations(
    field, labels, reason
):
    classifier = _classifier(
        _Completion(_decision(**{field: labels})), semantic_retries=0
    )
    attempt = classifier.classify(_trajectory())
    assert not attempt.cacheable
    assert attempt.classification == {
        "status": "failed",
        "reason": f"invalid_model_output_{reason}",
    }


def test_multiple_subjects_retry_to_one_primary_subject():
    client = _SequenceCompletion(
        [
            _decision(subject_labels=["地球与环境科学", "生命科学"]),
            _decision(subject_labels=["生命科学"]),
        ]
    )
    attempt = _classifier(client).classify(_trajectory())
    assert attempt.cacheable
    assert client.calls == 2
    assert attempt.classification["subject_labels"] == ["生命科学"]


@pytest.mark.parametrize(
    "response,reason",
    [
        ("not-json", "invalid_model_output_json"),
        ({"scenario_label_key": "legacy"}, "invalid_model_output_fields"),
        ({**_decision(), "reasoning": "private output"}, "invalid_model_output_fields"),
    ],
)
def test_invalid_model_output_is_a_compact_noncacheable_failure(response, reason):
    attempt = _classifier(_SequenceCompletion([response]), semantic_retries=0).classify(
        _trajectory()
    )
    assert attempt.classification == {"status": "failed", "reason": reason}
    assert not attempt.cacheable


def test_invalid_model_output_retries_then_recovers():
    client = _SequenceCompletion([{}, _decision()])
    attempt = _classifier(client).classify(_trajectory())
    assert attempt.cacheable and client.calls == 2


def test_invalid_output_retries_are_bounded():
    client = _SequenceCompletion([{}])
    attempt = _classifier(client).classify(_trajectory())
    assert not attempt.cacheable and client.calls == 3


def test_permanent_request_and_configuration_errors_are_not_semantic_retries():
    client = _RaisingCompletion(ClassificationRequestError("classifier_http_400"))
    attempt = _classifier(client).classify(_trajectory())
    assert attempt.classification == {
        "status": "failed",
        "reason": "classifier_http_400",
    }
    assert client.calls == 1
    client = _RaisingCompletion(ClassificationConfigurationError("bad config"))
    with pytest.raises(ClassificationConfigurationError):
        _classifier(client).classify(_trajectory())
    assert client.calls == 1


@pytest.mark.parametrize("budget,truncated", [(600_000, False), (1024, True)])
def test_user_only_projection_stays_valid_json_and_diagnostics_stay_off_rows(
    budget, truncated
):
    client = _Completion(_decision())
    classifier = _classifier(client, max_context_chars=budget)
    attempt = classifier.classify(
        _trajectory(
            messages=[
                {"role": "user", "content": "Please compare prices " + "x" * 4000},
                {"role": "assistant", "content": "private assistant text"},
            ]
        )
    )
    message = client.messages[0][1]["content"]
    payload = message.split("USER_REQUESTS_BEGIN\n", 1)[1].split(
        "\nUSER_REQUESTS_END", 1
    )[0]
    payload = payload.removesuffix("\n[...TRAJECTORY_CONTEXT_TRUNCATED...]")
    assert json.loads(payload)["user_turns"]
    assert len(payload) <= budget
    assert "private assistant text" not in payload
    assert attempt.context_truncated is truncated
    assert "context_truncated" not in attempt.classification
    assert "FINAL OUTPUT CHECK" in message.split("\nUSER_REQUESTS_END", 1)[1]


def test_gateway_context_reduction_preserves_diagnostic_and_stops_at_limit():
    client = _ContextThenCompletion()
    attempt = _classifier(client, max_context_chars=10_000).classify(
        _trajectory(messages=[{"role": "user", "content": "x" * 20000}])
    )
    assert attempt.cacheable and attempt.context_truncated
    assert len(client.calls) == 2
    assert len(client.calls[1][1]["content"]) < len(client.calls[0][1]["content"])
    client = _RaisingCompletion(ClassificationContextLimitError())
    attempt = _classifier(client).classify(_trajectory())
    assert attempt.classification == {
        "status": "failed",
        "reason": "context_limit_exhausted",
    }
    assert client.calls == 6


def test_changed_taxonomy_prompt_model_or_policy_invalidates_cache(
    tmp_path, monkeypatch
):
    import trajfoundry.classification.classifier as module

    client = _Completion(_decision())
    classifier = _classifier(client)
    baseline = classifier.config_hash
    classifier._system_prompt += " changed"
    assert classifier.config_hash != baseline
    classifier = _classifier(client)
    client.model = "another-model"
    assert classifier.config_hash != baseline
    client.model = "classifier-test"
    assert _classifier(client, prompt_version="old").config_hash != baseline
    assert _classifier(client, classifier_revision="old").config_hash != baseline
    assert _classifier(client, max_context_chars=1024).config_hash != baseline
    monkeypatch.setitem(module.CLASSIFICATION_POLICY, "eligibility", "changed")
    assert _classifier(client).config_hash != baseline
    assert DEFAULT_MAX_CONTEXT_CHARS == 600_000


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


@pytest.mark.parametrize("error_factory", _NETWORK_ERROR_FACTORIES)
@pytest.mark.parametrize("failure_stage", ["open", "read"])
def test_chat_client_recovers_from_transient_network_failures(
    error_factory, failure_stage
) -> None:
    content = '{"ok": true}'
    opener = _SequenceOpener(
        [
            error_factory("first failure"),
            error_factory("second failure"),
            orjson.dumps({"choices": [{"message": {"content": content}}]}),
        ],
        failure_stage,
    )
    sleeps: list[float] = []
    client = ChatCompletionsClient(
        api_url="https://classifier.invalid/v1/chat/completions",
        model="model",
        api_key="secret",
        max_retries=2,
        opener=opener,
        sleeper=sleeps.append,
        jitter=lambda: 0.0,
    )

    assert client.complete([{"role": "user", "content": "hello"}]) == content

    assert opener.calls == 3
    assert sleeps == [0.5, 1.0]
    assert len(opener.responses) == (3 if failure_stage == "read" else 1)
    assert all(response.closed for response in opener.responses)


@pytest.mark.parametrize("error_factory", _NETWORK_ERROR_FACTORIES)
@pytest.mark.parametrize("failure_stage", ["open", "read"])
@pytest.mark.parametrize("max_retries", [0, 2])
def test_chat_client_bounds_network_retries_and_closes_failed_responses(
    error_factory, failure_stage, max_retries
) -> None:
    opener = _SequenceOpener(
        [error_factory("transport failed") for _ in range(max_retries + 1)],
        failure_stage,
    )
    sleeps: list[float] = []
    client = ChatCompletionsClient(
        api_url="https://classifier.invalid/v1/chat/completions",
        model="model",
        api_key="secret",
        max_retries=max_retries,
        opener=opener,
        sleeper=sleeps.append,
        jitter=lambda: 0.0,
    )

    with pytest.raises(ClassificationRetryExhausted) as caught:
        client.complete([{"role": "user", "content": "hello"}])

    assert caught.value.reason == "network_retry_exhausted"
    assert opener.calls == max_retries + 1
    assert sleeps == ([] if max_retries == 0 else [0.5, 1.0])
    assert len(opener.responses) == (max_retries + 1 if failure_stage == "read" else 0)
    assert all(response.closed for response in opener.responses)


@pytest.mark.parametrize("error_factory", _NETWORK_ERROR_FACTORIES)
@pytest.mark.parametrize("failure_stage", ["open", "read"])
def test_classifier_network_exhaustion_is_not_cacheable_or_semantically_retried(
    tmp_path, error_factory, failure_stage
) -> None:
    opener = _SequenceOpener(
        [error_factory("transport failed") for _ in range(3)], failure_stage
    )
    sleeps: list[float] = []
    client = ChatCompletionsClient(
        api_url="https://classifier.invalid/v1/chat/completions",
        model="model",
        api_key="secret",
        max_retries=2,
        opener=opener,
        sleeper=sleeps.append,
        jitter=lambda: 0.0,
    )
    classifier = TrajectoryClassifier(
        client=client,
        taxonomy=_taxonomy(tmp_path),
        input_manifest_sha256="a" * 64,
        semantic_retries=2,
    )

    attempt = classifier.classify(_trajectory())

    assert not attempt.cacheable
    assert attempt.classification["status"] == "failed"
    assert attempt.classification["reason"] == "network_retry_exhausted"
    assert opener.calls == 3
    assert sleeps == [0.5, 1.0]
    assert all(response.closed for response in opener.responses)


@pytest.mark.parametrize("error_factory", _NETWORK_ERROR_FACTORIES)
@pytest.mark.parametrize("failure_stage", ["open", "read"])
def test_chat_client_network_logs_and_traceback_do_not_expose_secrets(
    caplog, error_factory, failure_stage
) -> None:
    secrets = ["sk-do-not-log-this", "private-user-request", "raw-response-marker"]
    opener = _SequenceOpener(
        [error_factory(" ".join(secrets)) for _ in range(2)], failure_stage
    )
    sleeps: list[float] = []
    client = ChatCompletionsClient(
        api_url="https://classifier.invalid/v1/chat/completions",
        model="model",
        api_key=secrets[0],
        max_retries=1,
        opener=opener,
        sleeper=sleeps.append,
        jitter=lambda: 0.0,
    )

    with pytest.raises(ClassificationRetryExhausted) as caught:
        client.complete([{"role": "user", "content": secrets[1]}])

    formatted = "".join(traceback.format_exception(caught.value))
    assert "network_retry_exhausted" in formatted
    assert all(secret not in formatted for secret in secrets)
    assert all(secret not in caplog.text for secret in secrets)
    assert all(secret not in repr(client) for secret in secrets)
    assert "retry_attempt=1/1" in caplog.text
    assert "attempts=2" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    assert opener.calls == 2
    assert sleeps == [0.5]


@pytest.mark.parametrize(
    "error_type", [RuntimeError, FileNotFoundError, http.client.HTTPException]
)
@pytest.mark.parametrize("failure_stage", ["open", "read"])
def test_chat_client_does_not_retry_unrelated_errors(error_type, failure_stage) -> None:
    error = error_type("unexpected local failure")
    opener = _SequenceOpener([error], failure_stage)
    sleeps: list[float] = []
    client = ChatCompletionsClient(
        api_url="https://classifier.invalid/v1/chat/completions",
        model="model",
        api_key="secret",
        max_retries=2,
        opener=opener,
        sleeper=sleeps.append,
    )

    with pytest.raises(error_type) as caught:
        client.complete([{"role": "user", "content": "hello"}])

    assert caught.value is error
    assert opener.calls == 1
    assert sleeps == []
    assert all(response.closed for response in opener.responses)


def test_chat_client_sends_bounded_json_request() -> None:
    captured: dict[str, object] = {}

    def opener(request, **_kwargs):
        captured["body"] = orjson.loads(request.data)
        return _HTTPResponse(
            orjson.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]})
        )

    client = ChatCompletionsClient(
        api_url="https://classifier.invalid/v1/chat/completions",
        model="model",
        api_key="secret",
        max_output_tokens=4_096,
        opener=opener,
    )

    assert client.complete([{"role": "user", "content": "hello"}]) == ('{"ok": true}')
    assert captured["body"] == {
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
        "max_tokens": 4_096,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }


def test_chat_client_reports_context_limit_separately() -> None:
    def opener(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "https://classifier.invalid/v1/chat/completions",
            400,
            "context too long",
            {},
            io.BytesIO(b'{"error":{"message":"maximum context length exceeded"}}'),
        )

    client = ChatCompletionsClient(
        api_url="https://classifier.invalid/v1/chat/completions",
        model="model",
        api_key="secret",
        opener=opener,
    )

    with pytest.raises(ClassificationContextLimitError):
        client.complete([{"role": "user", "content": "hello"}])


@pytest.mark.parametrize("status", [401, 403, 422])
def test_chat_client_does_not_retry_configuration_errors(status) -> None:
    attempts = 0
    sleeps: list[float] = []
    body = io.BytesIO(b"configuration error")

    def opener(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(
            "https://classifier.invalid/v1/chat/completions",
            status,
            "configuration error",
            {},
            body,
        )

    client = ChatCompletionsClient(
        api_url="https://classifier.invalid/v1/chat/completions",
        model="model",
        api_key="secret",
        opener=opener,
        sleeper=sleeps.append,
    )

    with pytest.raises(ClassificationConfigurationError, match=f"HTTP {status}"):
        client.complete([{"role": "user", "content": "hello"}])

    assert attempts == 1
    assert sleeps == []
    assert body.closed


def test_chat_client_retries_404_then_succeeds() -> None:
    attempts = 0
    sleeps: list[float] = []
    failed_body = io.BytesIO(b"not found")

    def opener(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise urllib.error.HTTPError(
                "https://classifier.invalid/v1/chat/completions",
                404,
                "not found",
                {},
                failed_body,
            )
        return _HTTPResponse(
            orjson.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]})
        )

    client = ChatCompletionsClient(
        api_url="https://classifier.invalid/v1/chat/completions",
        model="model",
        api_key="secret",
        max_retries=1,
        opener=opener,
        sleeper=sleeps.append,
        jitter=lambda: 0.0,
    )

    assert client.complete([{"role": "user", "content": "hello"}]) == '{"ok": true}'
    assert attempts == 2
    assert sleeps == [0.5]
    assert failed_body.closed


def test_chat_client_persistent_404_still_fails_as_configuration_error() -> None:
    attempts = 0
    sleeps: list[float] = []
    bodies: list[io.BytesIO] = []

    def opener(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        body = io.BytesIO(b"not found")
        bodies.append(body)
        raise urllib.error.HTTPError(
            "https://classifier.invalid/v1/chat/completions",
            404,
            "not found",
            {},
            body,
        )

    client = ChatCompletionsClient(
        api_url="https://classifier.invalid/v1/chat/completions",
        model="model",
        api_key="secret",
        max_retries=2,
        opener=opener,
        sleeper=sleeps.append,
        jitter=lambda: 0.0,
    )

    with pytest.raises(ClassificationConfigurationError, match="HTTP 404"):
        client.complete([{"role": "user", "content": "hello"}])

    assert attempts == 3
    assert sleeps == [0.5, 1.0]
    assert all(body.closed for body in bodies)


@pytest.mark.parametrize(
    "status, expected_error, expected_message",
    [
        (400, ClassificationRequestError, "classifier_http_400"),
        (422, ClassificationConfigurationError, "HTTP 422"),
    ],
)
@pytest.mark.parametrize(
    "read_error",
    [http.client.IncompleteRead(b"partial", 100), OSError("response read failed")],
)
def test_chat_client_preserves_http_status_when_error_body_read_fails(
    status, expected_error, expected_message, read_error
) -> None:
    attempts = 0
    sleeps: list[float] = []
    body = _HTTPResponse(b"", read_error=read_error)

    def opener(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(
            "https://classifier.invalid/v1/chat/completions",
            status,
            "request rejected",
            {},
            body,
        )

    client = ChatCompletionsClient(
        api_url="https://classifier.invalid/v1/chat/completions",
        model="model",
        api_key="secret",
        max_retries=2,
        opener=opener,
        sleeper=sleeps.append,
    )

    with pytest.raises(expected_error, match=expected_message):
        client.complete([{"role": "user", "content": "hello"}])

    assert attempts == 1
    assert sleeps == []
    assert body.closed


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
