from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries without adding a runtime dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    api_version: str = "2024-08-01-preview"
    model: str = "azure/gpt-4o"
    generator_model: str | None = None
    deployment_name: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 6000
    timeout_seconds: float = 120
    infrastructure_retries: int = 3
    api_style: Literal["auto", "azure", "openai"] = "auto"
    input_token_price_per_million_usd: Decimal = Decimal("2.50")
    output_token_price_per_million_usd: Decimal = Decimal("10.00")

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        base_url = os.getenv("BASE_URL") or os.getenv("LLM_BASE_URL")
        if not api_key:
            raise ValueError("OPENAI_API_KEY (or LLM_API_KEY) is required")
        if not base_url or urlparse(base_url).scheme not in {"https", "http"}:
            raise ValueError("BASE_URL must be an http(s) Azure OpenAI endpoint")
        api_style = os.getenv("OPENAI_API_STYLE", "auto").strip().lower()
        if api_style not in {"auto", "azure", "openai"}:
            raise ValueError(
                "OPENAI_API_STYLE must be one of: auto, azure, openai"
            )
        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            api_version=os.getenv("API_VERSION", "2024-08-01-preview"),
            model=os.getenv("MODEL", "azure/gpt-4o"),
            generator_model=os.getenv("GENERATOR_MODEL") or None,
            deployment_name=os.getenv("DEPLOYMENT_NAME", "gpt-4o"),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("MAX_TOKENS", "6000")),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
            infrastructure_retries=int(os.getenv("LLM_INFRA_MAX_RETRIES", "3")),
            api_style=api_style,
            input_token_price_per_million_usd=Decimal(
                os.getenv("INPUT_TOKEN_PRICE_PER_MILLION_USD", "2.50")
            ),
            output_token_price_per_million_usd=Decimal(
                os.getenv("OUTPUT_TOKEN_PRICE_PER_MILLION_USD", "10.00")
            ),
        )

    @property
    def effective_generator_model(self) -> str:
        return self.generator_model or self.model
