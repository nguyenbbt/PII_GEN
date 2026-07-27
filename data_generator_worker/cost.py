from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from .contracts import LLMTokenUsage


USD_QUANTUM = Decimal("0.00000001")


class CostCalculator:
    """Calculates request cost from provider usage and configured Azure prices."""

    def __init__(self, input_per_million_usd: Decimal, output_per_million_usd: Decimal) -> None:
        self.input_per_million_usd = Decimal(input_per_million_usd)
        self.output_per_million_usd = Decimal(output_per_million_usd)

    def calculate(self, input_tokens: int, output_tokens: int, total_tokens: int | None = None) -> LLMTokenUsage:
        total = total_tokens if total_tokens is not None else input_tokens + output_tokens
        money = (
            Decimal(input_tokens) * self.input_per_million_usd / Decimal(1_000_000)
            + Decimal(output_tokens) * self.output_per_million_usd / Decimal(1_000_000)
        ).quantize(USD_QUANTUM, rounding=ROUND_HALF_UP)
        return LLMTokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            money_cost=format(money, "f"),
        )
