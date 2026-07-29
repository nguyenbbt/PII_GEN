from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, List, Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field, ValidationError

from ..application.verification import (
    JUDGE_PROMPT_VERSION,
    REPAIR_PROMPT_VERSION,
    VerifierInfrastructureError,
)
from ..domain.models import (
    GeneratedEntity,
    RepairResult,
    TokenUsage,
    VerifierDecision,
)
from ..ports import CompletionClientError, CompletionResult


logger = logging.getLogger(__name__)


def _normalize_judge_payload(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], int, int, int, int, int]:
    """Normalize bounded, well-understood verifier contract variations."""
    normalized = dict(payload)
    normalized_suggested_fixes = 0
    normalized_fixable_severities = 0
    critical_fixable_issue = False
    raw_issues = payload.get("issues")
    if isinstance(raw_issues, list):
        status = str(payload.get("status") or "").strip().upper()
        if status == "FIXABLE":
            fallback_fix = (
                "Apply the smallest safe local correction described by the issue reason."
            )
        elif status == "REJECTED":
            fallback_fix = "Reject the sample and do not preserve the unsafe content."
        else:
            fallback_fix = "Regenerate the sample so the reported issue no longer occurs."
        normalized_issues: list[Any] = []
        for raw_issue in raw_issues:
            if not isinstance(raw_issue, dict):
                normalized_issues.append(raw_issue)
                continue
            issue = dict(raw_issue)
            suggested_fix = issue.get("suggested_fix")
            if suggested_fix is None or (
                isinstance(suggested_fix, str) and not suggested_fix.strip()
            ):
                issue["suggested_fix"] = fallback_fix
                normalized_suggested_fixes += 1
            severity = str(issue.get("severity") or "").strip().lower()
            if status == "FIXABLE" and severity in {"medium", "high"}:
                # FIXABLE already selects the local-repair route. Medium/high here
                # is a redundant LLM classification mismatch, not a new route.
                issue["severity"] = "low"
                normalized_fixable_severities += 1
            elif status == "FIXABLE" and severity == "critical":
                # Never downgrade a critical finding into a local edit.
                critical_fixable_issue = True
            normalized_issues.append(issue)
        normalized["issues"] = normalized_issues
        if critical_fixable_issue:
            normalized["status"] = "REJECTED"
            normalized["edits"] = []

    raw_edits = normalized.get("edits")
    if not isinstance(raw_edits, list):
        return (
            normalized,
            0,
            normalized_suggested_fixes,
            normalized_fixable_severities,
            0,
            int(critical_fixable_issue),
        )

    normalized_edits: list[Any] = []
    normalized_occurrences = 0
    normalized_target_values = 0
    for raw_edit in raw_edits:
        if not isinstance(raw_edit, dict):
            normalized_edits.append(raw_edit)
            continue
        edit = dict(raw_edit)
        if "target_value" in edit:
            target_value = edit.pop("target_value")
            action = str(edit.get("action") or "").strip().lower()
            destination = (
                "source_value"
                if action == "replace_template_artifact"
                else "value"
            )
            if target_value is not None and not edit.get(destination):
                edit[destination] = target_value
            normalized_target_values += 1
        occurrence = edit.get("occurrence")
        if not isinstance(occurrence, bool) and (
            occurrence == 0
            or (isinstance(occurrence, str) and occurrence.strip() == "0")
        ):
            edit["occurrence"] = 1
            normalized_occurrences += 1
        normalized_edits.append(edit)
    normalized["edits"] = normalized_edits
    return (
        normalized,
        normalized_occurrences,
        normalized_suggested_fixes,
        normalized_fixable_severities,
        normalized_target_values,
        int(critical_fixable_issue),
    )


def _validation_error_summary(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()[:8]
        )
    return str(exc)


def _safe_http_error_details(
    error: HTTPError,
    *,
    secrets: tuple[str, ...] = (),
) -> str:
    """Extract bounded provider diagnostics without exposing credentials."""
    try:
        raw_body = error.read(8192).decode("utf-8", errors="replace")
        payload = json.loads(raw_body)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return ""

    if not isinstance(payload, dict):
        return ""
    provider_error = payload.get("error", payload)
    if isinstance(provider_error, str):
        code = ""
        message = provider_error
    elif isinstance(provider_error, dict):
        code = str(provider_error.get("code", "")).strip()
        message = str(
            provider_error.get("message")
            or provider_error.get("detail")
            or ""
        ).strip()
    else:
        return ""

    # OpenAI-compatible gateways can echo request prompts in nested error
    # diagnostics. Keep the useful leading error and discard request-shaped
    # tails before the message reaches logs or tracebacks.
    folded_message = message.casefold()
    sensitive_markers = (
        "received_args",
        '"messages"',
        "'messages'",
        "messages=",
        "prompt=",
        "input=",
    )
    cut_positions = [
        folded_message.find(marker)
        for marker in sensitive_markers
        if folded_message.find(marker) >= 0
    ]
    if cut_positions:
        message = message[:min(cut_positions)].rstrip(" :,-")

    details = ": ".join(part for part in (code, message) if part)
    details = " ".join(details.split())[:1000]
    for secret in secrets:
        if secret:
            details = details.replace(secret, "<redacted>")
    return details


def _http_error_message(
    prefix: str,
    error: HTTPError,
    *,
    secrets: tuple[str, ...] = (),
) -> str:
    details = _safe_http_error_details(error, secrets=secrets)
    suffix = f": {details}" if details else ""
    return f"{prefix} HTTP {error.code}{suffix}"


def load_local_environment(path: Path = Path(".env")) -> None:
    """Load local settings without adding python-dotenv as a dependency."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class AzureOpenAISettings(BaseModel):
    api_key: str = Field(..., repr=False)
    base_url: str
    api_version: str = "2024-08-01-preview"
    deployment_name: str = "gpt-4o"
    model: str = "azure/gpt-4o"
    generator_model: str | None = None
    verifier_model: str | None = None
    temperature: float = Field(0.2, ge=0, le=2)
    max_tokens: int = Field(6000, gt=0)
    timeout_seconds: float = Field(120, gt=0)
    infrastructure_retries: int = Field(3, ge=0, le=10)
    verifier_temperature: float = Field(0.0, ge=0, le=2)
    verifier_judge_max_tokens: int = Field(4000, gt=0)
    verifier_repair_max_tokens: int = Field(2500, gt=0)
    api_style: Literal["auto", "azure", "openai"] = "auto"

    @classmethod
    def from_environment(cls) -> "AzureOpenAISettings":
        load_local_environment()
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        base_url = os.getenv("BASE_URL") or os.getenv("LLM_BASE_URL")
        if not api_key or not base_url:
            raise ValueError("OPENAI_API_KEY and BASE_URL are required")
        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            api_version=os.getenv("API_VERSION", "2024-08-01-preview"),
            deployment_name=os.getenv("DEPLOYMENT_NAME", "gpt-4o"),
            model=os.getenv("MODEL", "azure/gpt-4o"),
            generator_model=os.getenv("GENERATOR_MODEL") or None,
            verifier_model=os.getenv("VERIFIER_MODEL") or None,
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("MAX_TOKENS", "6000")),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
            infrastructure_retries=int(
                os.getenv("LLM_INFRA_MAX_RETRIES", "3")
            ),
            verifier_temperature=float(os.getenv("VERIFIER_TEMPERATURE", "0.0")),
            verifier_judge_max_tokens=int(
                os.getenv("VERIFIER_JUDGE_MAX_TOKENS", "4000")
            ),
            verifier_repair_max_tokens=int(
                os.getenv("VERIFIER_REPAIR_MAX_TOKENS", "2500")
            ),
            api_style=os.getenv("OPENAI_API_STYLE", "auto").strip().lower(),
        )

    @property
    def effective_generator_model(self) -> str:
        return self.generator_model or self.model

    @property
    def effective_verifier_model(self) -> str:
        return self.verifier_model or self.model

def _resolved_api_style(settings: AzureOpenAISettings) -> Literal["azure", "openai"]:
    if settings.api_style != "auto":
        return settings.api_style
    hostname = (urlparse(settings.base_url).hostname or "").lower()
    if hostname.endswith((".openai.azure.com", ".cognitiveservices.azure.com")):
        return "azure"
    return "openai"


def _response_model_identity(
    raw: dict[str, Any],
    *,
    requested_model: str,
) -> tuple[str, str]:
    """Return the provider-reported model and a transparent comparison status."""
    reported = str(raw.get("model") or "").strip()
    if not reported:
        return "<not-reported>", "not_reported"
    if reported.casefold() == requested_model.strip().casefold():
        return reported, "reported_match"
    return reported, "reported_different"


def _chat_completions_request(
    settings: AzureOpenAISettings,
    payload: dict[str, Any],
) -> Request:
    if _resolved_api_style(settings) == "azure":
        endpoint = (
            f"{settings.base_url}/openai/deployments/"
            f"{quote(settings.deployment_name, safe='')}/chat/completions"
            f"?api-version={quote(settings.api_version, safe='')}"
        )
        auth_headers = {"api-key": settings.api_key}
    else:
        endpoint = settings.base_url
        if not endpoint.rstrip("/").endswith("/chat/completions"):
            endpoint = f"{endpoint.rstrip('/')}/chat/completions"
        auth_headers = {"Authorization": f"Bearer {settings.api_key}"}
    return Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={**auth_headers, "Content-Type": "application/json"},
        method="POST",
    )


@dataclass(frozen=True)
class JsonCompletion:
    payload: dict[str, Any]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: int


class JsonCompletionTransport(Protocol):
    def complete(
        self,
        messages: List[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> JsonCompletion: ...


def _load_json_object_content(content: object) -> dict:
    """Parse JSON-object responses, tolerating common Markdown/prose wrappers."""
    if not isinstance(content, str):
        raise TypeError("LLM response content must be a string")
    stripped = content.strip()
    candidates = [stripped]
    if stripped.startswith("```") and stripped.endswith("```"):
        first_newline = stripped.find("\n")
        if first_newline >= 0:
            candidates.append(stripped[first_newline + 1:-3].strip())
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if 0 <= first_brace < last_brace:
        candidates.append(stripped[first_brace:last_brace + 1])

    last_error: json.JSONDecodeError | None = None
    for candidate in dict.fromkeys(candidates):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(payload, dict):
            raise TypeError("LLM JSON response must be an object")
        return payload
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON object found", stripped, 0)


def _usage_counts(response: dict[str, Any]) -> tuple[int, int, int]:
    """Return billable input/output totals, including reported reasoning tokens."""
    usage = response.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(
        usage.get("total_tokens")
        or input_tokens + completion_tokens
    )
    output_tokens = max(completion_tokens, total_tokens - input_tokens)
    return input_tokens, output_tokens, total_tokens


class AzureOpenAIJsonTransport:
    def __init__(self, settings: AzureOpenAISettings) -> None:
        self.settings = settings

    def complete(
        self,
        messages: List[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> JsonCompletion:
        request = _chat_completions_request(
            self.settings,
            {
                "model": self.settings.effective_verifier_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            },
        )
        last_error: Exception | None = None
        content = ""
        accumulated_input_tokens = 0
        accumulated_output_tokens = 0
        accumulated_total_tokens = 0
        requested_model = self.settings.effective_verifier_model
        api_style = _resolved_api_style(self.settings)
        deployment = (
            self.settings.deployment_name
            if api_style == "azure"
            else "<not-applicable>"
        )
        for attempt in range(self.settings.infrastructure_retries + 1):
            started = time.perf_counter()
            logger.info(
                "[llm verifier] request attempt=%s/%s requested_model=%s "
                "api_style=%s deployment=%s timeout=%ss",
                attempt + 1,
                self.settings.infrastructure_retries + 1,
                requested_model,
                api_style,
                deployment,
                self.settings.timeout_seconds,
            )
            try:
                with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                input_tokens, output_tokens, total_tokens = _usage_counts(raw)
                accumulated_input_tokens += input_tokens
                accumulated_output_tokens += output_tokens
                accumulated_total_tokens += total_tokens
                content = raw["choices"][0]["message"]["content"]
                payload = _load_json_object_content(content)
                if not isinstance(payload, dict):
                    raise VerifierInfrastructureError(
                        "Azure OpenAI verifier response must be a JSON object"
                    )
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                response_model, model_status = _response_model_identity(
                    raw,
                    requested_model=requested_model,
                )
                logger.info(
                    "[llm verifier] response received latency_ms=%s "
                    "tokens=%s cumulative_tokens=%s requested_model=%s "
                    "response_model=%s model_status=%s",
                    elapsed_ms,
                    total_tokens,
                    accumulated_total_tokens,
                    requested_model,
                    response_model,
                    model_status,
                )
                return JsonCompletion(
                    payload=payload,
                    input_tokens=accumulated_input_tokens,
                    output_tokens=accumulated_output_tokens,
                    total_tokens=accumulated_total_tokens,
                    latency_ms=elapsed_ms,
                )
            except HTTPError as exc:
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise VerifierInfrastructureError(
                        _http_error_message(
                            "Azure OpenAI verifier",
                            exc,
                            secrets=(self.settings.api_key,),
                        )
                    ) from exc
                last_error = exc
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, URLError) as exc:
                last_error = exc
                if isinstance(exc, json.JSONDecodeError):
                    logger.warning(
                        "[llm verifier] invalid JSON msg=%s line=%s col=%s content_length=%s",
                        exc.msg,
                        exc.lineno,
                        exc.colno,
                        len(content),
                    )
            if attempt < self.settings.infrastructure_retries:
                delay = min(2**attempt, 8)
                logger.warning(
                    "[llm verifier] attempt failed error=%s; retrying in %ss",
                    type(last_error).__name__,
                    delay,
                )
                time.sleep(delay)
        logger.error(
            "[llm verifier] request failed after %s attempts",
            self.settings.infrastructure_retries + 1,
        )
        raise VerifierInfrastructureError(
            "Azure OpenAI verifier request failed after infrastructure retries",
            raw_usage=(
                accumulated_input_tokens,
                accumulated_output_tokens,
                accumulated_total_tokens,
            ),
        ) from last_error


class AzureOpenAIVerifierClient:
    def __init__(
        self,
        settings: AzureOpenAISettings,
        *,
        input_price_per_million: Decimal,
        output_price_per_million: Decimal,
        transport: JsonCompletionTransport | None = None,
    ) -> None:
        self.settings = settings
        self.input_price_per_million = Decimal(input_price_per_million)
        self.output_price_per_million = Decimal(output_price_per_million)
        self.transport = transport or AzureOpenAIJsonTransport(settings)

    def judge(self, messages: List[dict[str, str]]) -> VerifierDecision:
        try:
            completion = self.transport.complete(
                messages,
                temperature=self.settings.verifier_temperature,
                max_tokens=self.settings.verifier_judge_max_tokens,
            )
        except VerifierInfrastructureError as exc:
            usage = exc.token_usage or self._usage_from_counts(exc.raw_usage)
            raise VerifierInfrastructureError(
                str(exc),
                stage="judge",
                token_usage=usage,
                raw_usage=exc.raw_usage,
                completed_usage=exc.completed_usage,
            ) from exc
        (
            payload,
            normalized_occurrences,
            normalized_suggested_fixes,
            normalized_fixable_severities,
            normalized_target_values,
            normalized_critical_routes,
        ) = _normalize_judge_payload(completion.payload)
        if normalized_occurrences:
            logger.warning(
                "[llm verifier] normalized %s zero-based edit occurrence value(s) "
                "to one-based indexing",
                normalized_occurrences,
            )
        if normalized_suggested_fixes:
            logger.warning(
                "[llm verifier] filled %s missing suggested_fix value(s) "
                "without discarding their issues",
                normalized_suggested_fixes,
            )
        if normalized_fixable_severities:
            logger.warning(
                "[llm verifier] normalized %s medium/high FIXABLE issue "
                "severity value(s) to low",
                normalized_fixable_severities,
            )
        if normalized_target_values:
            logger.warning(
                "[llm verifier] normalized %s edit target_value alias(es) "
                "to the supported value field",
                normalized_target_values,
            )
        if normalized_critical_routes:
            logger.warning(
                "[llm verifier] routed contradictory FIXABLE decision with "
                "critical issue to REJECTED",
            )
        try:
            return VerifierDecision(
                **payload,
                token_usage=self._usage(completion),
                latency_ms=completion.latency_ms,
                model=self.settings.effective_verifier_model,
                prompt_version=JUDGE_PROMPT_VERSION,
            )
        except (TypeError, ValueError, ValidationError) as exc:
            details = _validation_error_summary(exc)
            logger.error(
                "[llm verifier] invalid judge response contract: %s",
                details,
            )
            raise VerifierInfrastructureError(
                f"Verifier judge returned an invalid response contract: {details}",
                stage="judge",
                token_usage=self._usage(completion),
            ) from exc

    def repair(self, messages: List[dict[str, str]]) -> RepairResult:
        try:
            completion = self.transport.complete(
                messages,
                temperature=self.settings.verifier_temperature,
                max_tokens=self.settings.verifier_repair_max_tokens,
            )
        except VerifierInfrastructureError as exc:
            usage = exc.token_usage or self._usage_from_counts(exc.raw_usage)
            raise VerifierInfrastructureError(
                str(exc),
                stage="repair",
                token_usage=usage,
                raw_usage=exc.raw_usage,
                completed_usage=exc.completed_usage,
            ) from exc
        try:
            return RepairResult(
                **completion.payload,
                token_usage=self._usage(completion),
                latency_ms=completion.latency_ms,
                model=self.settings.effective_verifier_model,
                prompt_version=REPAIR_PROMPT_VERSION,
            )
        except (TypeError, ValueError, ValidationError) as exc:
            details = _validation_error_summary(exc)
            logger.error(
                "[llm verifier] invalid repair response contract: %s",
                details,
            )
            raise VerifierInfrastructureError(
                f"Verifier repair returned an invalid response contract: {details}",
                stage="repair",
                token_usage=self._usage(completion),
            ) from exc

    def _usage(self, completion: JsonCompletion) -> TokenUsage:
        return self._usage_from_counts((
            completion.input_tokens,
            completion.output_tokens,
            completion.total_tokens,
        ))

    def _usage_from_counts(
        self,
        counts: tuple[int, int, int] | None,
    ) -> TokenUsage | None:
        if counts is None:
            return None
        input_tokens, output_tokens, total_tokens = counts
        money = (
            Decimal(input_tokens)
            * self.input_price_per_million
            / Decimal(1_000_000)
            + Decimal(output_tokens)
            * self.output_price_per_million
            / Decimal(1_000_000)
        ).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            money_cost=money,
        )


class OfflineVerifierClient:
    """No-cost semantic gate used only for local smoke tests."""

    @staticmethod
    def judge(messages: List[dict[str, str]]) -> VerifierDecision:
        return VerifierDecision(
            status="PASS",
            score=100,
            issues=[],
            token_usage=TokenUsage.zero(),
            latency_ms=0,
            model="offline-verifier",
            prompt_version=JUDGE_PROMPT_VERSION,
        )

    @staticmethod
    def repair(messages: List[dict[str, str]]) -> RepairResult:
        envelope = json.loads(messages[-1]["content"])
        candidate = envelope["candidate"]
        return RepairResult(
            tagged_text=candidate["tagged_text"],
            entities=[GeneratedEntity(**entity) for entity in candidate["entities"]],
            token_usage=TokenUsage.zero(),
            latency_ms=0,
            model="offline-verifier",
            prompt_version=REPAIR_PROMPT_VERSION,
        )


class AzureOpenAICompletionClient:
    def __init__(self, settings: AzureOpenAISettings) -> None:
        self.settings = settings

    def generate(
        self,
        messages: List[dict[str, str]],
    ) -> CompletionResult:
        request = _chat_completions_request(
            self.settings,
            {
                "model": self.settings.effective_generator_model,
                "messages": messages,
                "temperature": self.settings.temperature,
                "max_tokens": self.settings.max_tokens,
                "response_format": {"type": "json_object"},
            },
        )
        last_error: Exception | None = None
        content = ""
        accumulated_input_tokens = 0
        accumulated_output_tokens = 0
        accumulated_total_tokens = 0
        requested_model = self.settings.effective_generator_model
        api_style = _resolved_api_style(self.settings)
        deployment = (
            self.settings.deployment_name
            if api_style == "azure"
            else "<not-applicable>"
        )
        for attempt in range(self.settings.infrastructure_retries + 1):
            started = time.perf_counter()
            logger.info(
                "[llm generator] request attempt=%s/%s requested_model=%s "
                "api_style=%s deployment=%s timeout=%ss",
                attempt + 1,
                self.settings.infrastructure_retries + 1,
                requested_model,
                api_style,
                deployment,
                self.settings.timeout_seconds,
            )
            try:
                with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                input_tokens, output_tokens, total_tokens = _usage_counts(raw)
                accumulated_input_tokens += input_tokens
                accumulated_output_tokens += output_tokens
                accumulated_total_tokens += total_tokens
                content = raw["choices"][0]["message"]["content"]
                parsed = _load_json_object_content(content)
                tagged_text = parsed["tagged_text"]
                if not isinstance(tagged_text, str) or not tagged_text.strip():
                    raise ValueError("tagged_text must be a non-empty string")
                entities = parsed.get("entities")
                if not isinstance(entities, list) or any(
                    not isinstance(entity, dict) for entity in entities
                ):
                    raise ValueError("entities must be an array of objects")
                parsed_entities = [
                    {
                        "label": str(entity.get("label", "")),
                        "value": str(entity.get("value", "")),
                    }
                    for entity in entities
                ]
                response_model, model_status = _response_model_identity(
                    raw,
                    requested_model=requested_model,
                )
                logger.info(
                    "[llm generator] response received latency_ms=%s "
                    "tokens=%s cumulative_tokens=%s requested_model=%s "
                    "response_model=%s model_status=%s",
                    round((time.perf_counter() - started) * 1000),
                    total_tokens,
                    accumulated_total_tokens,
                    requested_model,
                    response_model,
                    model_status,
                )
                return CompletionResult(
                    tagged_text=tagged_text,
                    entities=parsed_entities,
                    input_tokens=accumulated_input_tokens,
                    output_tokens=accumulated_output_tokens,
                    total_tokens=accumulated_total_tokens,
                    model=self.settings.effective_generator_model,
                    latency_ms=round((time.perf_counter() - started) * 1000),
                )
            except HTTPError as exc:
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise RuntimeError(
                        _http_error_message(
                            "Azure OpenAI",
                            exc,
                            secrets=(self.settings.api_key,),
                        )
                    ) from exc
                last_error = exc
            except (KeyError, TypeError, ValueError, URLError) as exc:
                last_error = exc
                if isinstance(exc, json.JSONDecodeError):
                    logger.warning(
                        "[llm generator] invalid JSON msg=%s line=%s col=%s content_length=%s",
                        exc.msg,
                        exc.lineno,
                        exc.colno,
                        len(content),
                    )
            if attempt < self.settings.infrastructure_retries:
                delay = min(2**attempt, 8)
                logger.warning(
                    "[llm generator] attempt failed error=%s; retrying in %ss",
                    type(last_error).__name__,
                    delay,
                )
                time.sleep(delay)
        logger.error(
            "[llm generator] request failed after %s attempts",
            self.settings.infrastructure_retries + 1,
        )
        raw_usage = (
            accumulated_input_tokens,
            accumulated_output_tokens,
            accumulated_total_tokens,
        )
        if accumulated_total_tokens:
            raise CompletionClientError(
                "Azure OpenAI returned unusable Generator content after "
                "infrastructure retries",
                raw_usage=raw_usage,
            ) from last_error
        raise RuntimeError(
            "Azure OpenAI request failed after infrastructure retries"
        ) from last_error


class OfflineCompletionClient:
    """No-cost client for smoke tests and API demonstrations."""

    _ROLES = {
        "customer": "khách hàng",
        "staff_member": "nhân viên phụ trách",
        "system_operator": "điều phối viên hệ thống",
        "third_party": "đối tác xử lý",
    }
    _INTENTS = {
        "request_action": "đề nghị cập nhật",
        "report_issue": "báo cáo vấn đề liên quan đến",
        "confirm_information": "xác nhận thông tin",
        "provide_update": "cung cấp bản cập nhật về",
        "ask_for_help": "yêu cầu hỗ trợ kiểm tra",
    }
    _CONTRACT_HEADINGS = {
        "agreement_clause": "ĐIỀU KHOẢN THỎA THUẬN",
        "administrative_record": "HỒ SƠ HÀNH CHÍNH",
        "company_notice": "THÔNG BÁO NỘI BỘ",
        "handover_minutes": "BIÊN BẢN BÀN GIAO",
    }

    def generate(self, messages: List[dict[str, str]]) -> CompletionResult:
        user_prompt = messages[-1]["content"]
        if "```json\n" in user_prompt:
            user_prompt = user_prompt.split("```json\n", 1)[1].split("\n```", 1)[0]
        payload = json.loads(user_prompt)
        task = payload["task"]
        profile = task.get("diversity_profile") or {}
        sample_structure = task.get("sample_structure") or {}
        frame = payload["context_frame"]
        if task["sample_type"] == "pure_negative":
            seeds = payload["content_seeds"]
            index = self._profile_index(profile)
            role = seeds["generic_roles"][index % len(seeds["generic_roles"])]
            action = seeds["actions"][index % len(seeds["actions"])]
            object_name = seeds["objects"][index % len(seeds["objects"])]
            text = self._render(
                profile,
                frame["document_type"],
                role,
                f"{action} {object_name}",
                sample_structure,
            )
            return self._completion(text, [])
        if task["sample_type"] == "hard_negative" and payload.get("hard_negative_mode") == "decoy_only":
            clauses = [
                f"{decoy['required_context_cues'][0]} {decoy['value']} được cập nhật trong hệ thống"
                for decoy in payload["decoys"]
            ]
            text = self._render(
                profile,
                frame["document_type"],
                self._role(profile),
                " và ".join(clauses),
                sample_structure,
            )
            return self._completion(text, [])
        entities = [
            {"label": item["label"], "value": item["value"]}
            for item in payload["positive_entities"]
        ]
        tagged_parts = [f"<{item['label']}>{item['value']}</{item['label']}>" for item in entities]
        text = self._render(
            profile,
            frame["document_type"],
            self._role(profile),
            ", ".join(tagged_parts),
            sample_structure,
        )
        for decoy in payload["decoys"]:
            text += f" {decoy['required_context_cues'][0].capitalize()} của hệ thống là {decoy['value']}."
        return self._completion(text, entities)

    @staticmethod
    def _completion(
        tagged_text: str,
        entities: List[dict[str, str]],
    ) -> CompletionResult:
        return CompletionResult(
            tagged_text=tagged_text,
            entities=entities,
            input_tokens=80,
            output_tokens=40,
            total_tokens=120,
            model="offline-demo",
            latency_ms=0,
        )

    def _render(
        self,
        profile: dict,
        document_type: str,
        role: str,
        content: str,
        sample_structure: dict,
    ) -> str:
        intent = self._INTENTS.get(profile.get("intent"), "ghi nhận")
        structure = profile.get("document_structure", "single_sentence")
        register = profile.get("language_register", "neutral")
        register_marker = {
            "formal": "Theo nội dung tiếp nhận",
            "neutral": "Thông tin mới",
            "informal": "Mình gửi nội dung",
            "concise_technical": "Bản ghi vận hành",
        }.get(register, "Thông tin mới")
        selected = self._render_selected_structure(
            structure=structure,
            intent=intent,
            register_marker=register_marker,
            role=role,
            content=content,
            sample_structure=sample_structure,
            document_type=document_type,
            context_frame_id=str(profile.get("context_frame_id") or document_type),
            length_bucket=str(profile.get("length_bucket") or "medium"),
        )
        if selected is not None:
            return selected
        if structure == "short_dialogue":
            return f"{role.capitalize()}: {intent} {content}.\nNhân viên: Đã ghi nhận trong {document_type}."
        if structure == "two_sentence_note":
            return f"{register_marker}, {role} {intent} {content}. Nội dung thuộc {document_type} đang được xử lý."
        if structure == "form_like_record":
            return f"{document_type.upper()} | Vai trò: {role} | Mục đích: {intent} | Nội dung: {content}."
        return f"{register_marker}, {role} {intent} {content} trong {document_type}."

    def _render_selected_structure(
        self,
        *,
        structure: str,
        intent: str,
        register_marker: str,
        role: str,
        content: str,
        sample_structure: dict,
        document_type: str,
        context_frame_id: str,
        length_bucket: str,
    ) -> str | None:
        structure_type = sample_structure.get("type")
        if structure_type == "chat":
            speakers = (
                ("Bạn A", "Bạn B")
                if structure == "friend_chat"
                else ("Khách hàng", "Nhân viên hỗ trợ")
            )
            return (
                f"{speakers[0]}: {intent} {content}.\n"
                f"{speakers[1]}: Đã tiếp nhận và đang xử lý nội dung này."
            )
        if structure_type == "contract":
            heading = self._CONTRACT_HEADINGS.get(
                structure,
                "HỒ SƠ DOANH NGHIỆP",
            )
            return (
                f"{heading} | Hồ sơ {document_type}, bối cảnh "
                f"{context_frame_id.replace('_', ' ')}, mức {length_bucket} | "
                f"{register_marker}; "
                f"{role} {intent} {content}."
            )
        if structure_type == "custom":
            instruction = sample_structure.get(
                "custom_instruction",
                "Nội dung tùy chỉnh",
            )
            return f"{instruction}: {role} {intent} {content}."
        return None

    def _role(self, profile: dict) -> str:
        return self._ROLES.get(profile.get("speaker_role"), "người gửi yêu cầu")

    @staticmethod
    def _profile_index(profile: dict) -> int:
        return sum(ord(character) for value in profile.values() if isinstance(value, str) for character in value)
