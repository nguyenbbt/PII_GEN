from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from .config import Settings


@dataclass(frozen=True)
class CompletionResponse:
    tagged_text: str
    entities: list[dict[str, str]]
    input_tokens: int
    output_tokens: int
    total_tokens: int


class LLMClient(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> CompletionResponse: ...


class AzureOpenAIClient:
    """Calls Azure OpenAI Chat Completions directly with the standard library."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, messages: list[dict[str, str]]) -> CompletionResponse:
        endpoint, auth_headers = self._endpoint_and_auth_headers()
        body = json.dumps({
            "model": self.settings.effective_generator_model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
        request = Request(
            endpoint,
            data=body,
            headers={**auth_headers, "Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.settings.infrastructure_retries + 1):
            try:
                with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                    payload: dict[str, Any] = json.loads(
                        response.read().decode("utf-8")
                    )
                return self._parse_response(payload)
            except HTTPError as exc:
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise RuntimeError(
                        f"Azure OpenAI request failed with HTTP {exc.code}"
                    ) from exc
                last_error = exc
            except (TypeError, ValueError, URLError) as exc:
                last_error = exc
            if attempt < self.settings.infrastructure_retries:
                time.sleep(min(2**attempt, 8))
        raise RuntimeError("Azure OpenAI request failed after infrastructure retries") from last_error

    def _endpoint_and_auth_headers(self) -> tuple[str, dict[str, str]]:
        api_style = self.settings.api_style
        if api_style == "auto":
            hostname = (urlparse(self.settings.base_url).hostname or "").lower()
            api_style = (
                "azure"
                if hostname.endswith(
                    (".openai.azure.com", ".cognitiveservices.azure.com")
                )
                else "openai"
            )
        if api_style == "openai":
            endpoint = self.settings.base_url.rstrip("/")
            if not endpoint.endswith("/chat/completions"):
                endpoint = f"{endpoint}/chat/completions"
            return endpoint, {
                "Authorization": f"Bearer {self.settings.api_key}"
            }
        endpoint = (
            f"{self.settings.base_url}/openai/deployments/"
            f"{quote(self.settings.deployment_name, safe='')}"
            f"/chat/completions?api-version="
            f"{quote(self.settings.api_version, safe='')}"
        )
        return endpoint, {"api-key": self.settings.api_key}

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> CompletionResponse:
        try:
            content = response["choices"][0]["message"]["content"]
            payload = json.loads(content)
            tagged_text = payload["tagged_text"]
            entities = payload["entities"]
            usage = response.get("usage", {})
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Azure OpenAI response is not the required JSON object") from exc
        if isinstance(tagged_text, list):
            tagged_text = "".join(
                str(part.get("text", ""))
                if isinstance(part, dict)
                else str(part)
                for part in tagged_text
            )
        if not isinstance(tagged_text, str) or not tagged_text.strip():
            raise ValueError("Azure OpenAI response has no tagged_text string")
        if not isinstance(entities, list):
            raise ValueError("Azure OpenAI response entities must be an array")
        parsed_entities: list[dict[str, str]] = []
        for entity in entities:
            if not isinstance(entity, dict):
                raise ValueError("each Azure OpenAI entity must be an object")
            parsed_entities.append(
                {
                    "label": str(entity.get("label", "")),
                    "value": str(entity.get("value", "")),
                }
            )
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens))
        return CompletionResponse(tagged_text, parsed_entities, input_tokens, output_tokens, total_tokens)
