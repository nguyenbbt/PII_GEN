from decimal import Decimal
import unittest
from unittest.mock import patch

from data_generator_worker.cost import CostCalculator
from pii_factory.application.services import CostRates
from pii_factory.bootstrap import _role_cost_rates


class CostCalculatorTests(unittest.TestCase):
    def test_calculates_input_output_total_and_money_cost(self) -> None:
        usage = CostCalculator(Decimal("2.50"), Decimal("10.00")).calculate(1_000_000, 500_000)

        self.assertEqual(usage.input_tokens, 1_000_000)
        self.assertEqual(usage.output_tokens, 500_000)
        self.assertEqual(usage.total_tokens, 1_500_000)
        self.assertEqual(usage.money_cost, "7.50000000")

    def test_role_specific_rates_override_legacy_fallback(self) -> None:
        fallback = CostRates(
            input_per_million_usd=Decimal("2.50"),
            output_per_million_usd=Decimal("10.00"),
        )
        with patch.dict(
            "os.environ",
            {
                "GENERATOR_INPUT_TOKEN_PRICE_PER_MILLION_USD": "0.30",
                "GENERATOR_OUTPUT_TOKEN_PRICE_PER_MILLION_USD": "2.50",
            },
            clear=True,
        ):
            generator_rates = _role_cost_rates("generator", fallback)
            verifier_rates = _role_cost_rates("verifier", fallback)

        self.assertEqual(
            generator_rates.input_per_million_usd,
            Decimal("0.30"),
        )
        self.assertEqual(
            generator_rates.output_per_million_usd,
            Decimal("2.50"),
        )
        self.assertEqual(
            verifier_rates.input_per_million_usd,
            fallback.input_per_million_usd,
        )
        self.assertEqual(
            verifier_rates.output_per_million_usd,
            fallback.output_per_million_usd,
        )
