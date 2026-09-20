"""Minimal OpenAI-compatible Chat Completions client with bounded retries."""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from email.message import Message
from typing import Any
from urllib.parse import urlsplit

import orjson


class ClassificationAPIError(RuntimeError):
    """Base class for safe-to-display classifier API failures."""


class ClassificationConfigurationError(ClassificationAPIError):
    """A request cannot succeed until API configuration is corrected."""


class ClassificationRequestError(ClassificationAPIError):
    """A single classification request failed permanently."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ClassificationRetryExhausted(ClassificationAPIError):
    """A retryable API failure persisted through the configured attempts."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _validate_api_url(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ClassificationConfigurationError("classifier API URL is required")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ClassificationConfigurationError(
            "classifier API URL must be an absolute HTTP(S) URL"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ClassificationConfigurationError(
            "classifier API URL must not contain credentials"
        )
    if not parsed.path.rstrip("/").endswith("/chat/completions"):
        raise ClassificationConfigurationError(
            "classifier API URL must include the Chat Completions path"
        )
    if parsed.query or parsed.fragment:
        raise ClassificationConfigurationError(
            "classifier API URL must not contain a query or fragment"
        )
    return value


def _retry_after(headers: Message | None) -> float | None:
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return min(max(seconds, 0.0), 300.0)


class ChatCompletionsClient:
    """Call one configured OpenAI-compatible model without logging secrets."""

    def __init__(
        self,
        *,
        api_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 5,
        opener: Callable[..., Any] = urllib.request.urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.api_url = _validate_api_url(api_url)
        if not isinstance(model, str) or not model:
            raise ClassificationConfigurationError("classifier model is required")
        if not isinstance(api_key, str) or not api_key:
            raise ClassificationConfigurationError("classifier API key is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.model = model
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._opener = opener
        self._sleeper = sleeper
        self._jitter = jitter

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(api_url={self.api_url!r}, "
            f"model={self.model!r}, api_key='[REDACTED]')"
        )

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return assistant content, retrying only transient failures."""

        payload = orjson.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
        )
        request = urllib.request.Request(
            self.api_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        for attempt in range(self.max_retries + 1):
            try:
                with self._opener(request, timeout=self.timeout_seconds) as response:
                    raw = response.read()
                return self._parse_response(raw)
            except urllib.error.HTTPError as error:
                error.close()
                status = error.code
                if status in {401, 403, 404, 422}:
                    raise ClassificationConfigurationError(
                        f"classifier API configuration failed with HTTP {status}"
                    ) from error
                if status == 429 or 500 <= status < 600:
                    if attempt == self.max_retries:
                        reason = (
                            "rate_limit_exhausted"
                            if status == 429
                            else "service_retry_exhausted"
                        )
                        raise ClassificationRetryExhausted(reason) from error
                    self._sleep_before_retry(attempt, _retry_after(error.headers))
                    continue
                raise ClassificationRequestError(f"classifier_http_{status}") from error
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == self.max_retries:
                    raise ClassificationRetryExhausted(
                        "network_retry_exhausted"
                    ) from error
                self._sleep_before_retry(attempt, None)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                if attempt == self.max_retries:
                    raise ClassificationRetryExhausted(
                        "invalid_api_response"
                    ) from error
                self._sleep_before_retry(attempt, None)

        raise AssertionError("retry loop must return or raise")

    @staticmethod
    def _parse_response(raw: bytes) -> str:
        value = orjson.loads(raw)
        if type(value) is not dict:
            raise ValueError("response must be an object")
        choices = value.get("choices")
        if type(choices) is not list or not choices or type(choices[0]) is not dict:
            raise ValueError("response must contain choices")
        message = choices[0].get("message")
        if type(message) is not dict:
            raise ValueError("response choice must contain a message")
        content = message.get("content")
        if type(content) is not str or not content:
            raise ValueError("response message must contain text content")
        return content

    def _sleep_before_retry(self, attempt: int, retry_after: float | None) -> None:
        if retry_after is not None:
            delay = retry_after
        else:
            delay = min(60.0, 2.0**attempt) * (0.5 + 0.5 * self._jitter())
        self._sleeper(delay)


__all__ = [
    "ChatCompletionsClient",
    "ClassificationAPIError",
    "ClassificationConfigurationError",
    "ClassificationRequestError",
    "ClassificationRetryExhausted",
]
