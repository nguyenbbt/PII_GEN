from decimal import Decimal
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from data_generator_worker.cost import CostCalculator
from pii_factory.application.services import CostRates
from pii_factory.bootstrap import _cost_rates, _role_cost_rates, build_pipeline


class CostCalculatorTests(unittest.TestCase):
    def test_calculates_input_output_total_and_money_cost(self) -> None:
        usage = CostCalculator(Decimal("2.50"), Decimal("10.00")).calculate(1_000_000, 500_000)

        self.assertEqual(usage.input_tokens, 1_000_000)
        self.assertEqual(usage.output_tokens, 500_000)
        self.assertEqual(usage.total_tokens, 1_500_000)
        self.assertEqual(usage.money_cost, "7.50000000")

    def test_generator_and_verifier_prices_are_independent_with_legacy_fallback(
        self,
    ) -> None:
        from unittest.mock import patch

        with patch.dict(
            "os.environ",
            {
                "INPUT_TOKEN_PRICE_PER_MILLION_USD": "2",
                "OUTPUT_TOKEN_PRICE_PER_MILLION_USD": "4",
                "GENERATOR_INPUT_TOKEN_PRICE_PER_MILLION_USD": "0.3",
                "GENERATOR_OUTPUT_TOKEN_PRICE_PER_MILLION_USD": "2.5",
                "VERIFIER_INPUT_TOKEN_PRICE_PER_MILLION_USD": "1.25",
                "VERIFIER_OUTPUT_TOKEN_PRICE_PER_MILLION_USD": "10",
            },
            clear=False,
        ):
            generator = _cost_rates("GENERATOR")
            verifier = _cost_rates("VERIFIER")

        self.assertEqual(generator.input_per_million_usd, Decimal("0.3"))
        self.assertEqual(generator.output_per_million_usd, Decimal("2.5"))
        self.assertEqual(verifier.input_per_million_usd, Decimal("1.25"))
        self.assertEqual(verifier.output_per_million_usd, Decimal("10"))

        with patch.dict(
            "os.environ",
            {
                "INPUT_TOKEN_PRICE_PER_MILLION_USD": "2",
                "OUTPUT_TOKEN_PRICE_PER_MILLION_USD": "4",
            },
            clear=True,
        ):
            legacy = _cost_rates("GENERATOR")

        self.assertEqual(legacy.input_per_million_usd, Decimal("2"))
        self.assertEqual(legacy.output_per_million_usd, Decimal("4"))

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

    def test_default_rates_match_the_configured_gemini_models(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            generator = _cost_rates("GENERATOR")
            verifier = _cost_rates("VERIFIER")

        self.assertEqual(generator.input_per_million_usd, Decimal("0.30"))
        self.assertEqual(generator.output_per_million_usd, Decimal("2.50"))
        self.assertEqual(verifier.input_per_million_usd, Decimal("1.25"))
        self.assertEqual(verifier.output_per_million_usd, Decimal("10.00"))

    def test_build_pipeline_loads_role_prices_from_dotenv_first(self) -> None:
        original_cwd = Path.cwd()
        with TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            root = Path(directory)
            (root / ".env").write_text(
                "\n".join([
                    "GENERATOR_INPUT_TOKEN_PRICE_PER_MILLION_USD=0.77",
                    "GENERATOR_OUTPUT_TOKEN_PRICE_PER_MILLION_USD=3.33",
                ]),
                encoding="utf-8",
            )
            try:
                os.chdir(root)
                pipeline, _, _ = build_pipeline(
                    offline=True,
                    output_directory=root / "output",
                )
            finally:
                os.chdir(original_cwd)

        self.assertEqual(
            pipeline.generator.cost_rates.input_per_million_usd,
            Decimal("0.77"),
        )
        self.assertEqual(
            pipeline.generator.cost_rates.output_per_million_usd,
            Decimal("3.33"),
        )
