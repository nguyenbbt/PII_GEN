from decimal import Decimal
import unittest

from data_generator_worker.cost import CostCalculator


class CostCalculatorTests(unittest.TestCase):
    def test_calculates_input_output_total_and_money_cost(self) -> None:
        usage = CostCalculator(Decimal("2.50"), Decimal("10.00")).calculate(1_000_000, 500_000)

        self.assertEqual(usage.input_tokens, 1_000_000)
        self.assertEqual(usage.output_tokens, 500_000)
        self.assertEqual(usage.total_tokens, 1_500_000)
        self.assertEqual(usage.money_cost, "7.50000000")
