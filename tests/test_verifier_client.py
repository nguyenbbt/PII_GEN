from decimal import Decimal
import unittest

from pii_factory.infrastructure.clients import (
    AzureOpenAISettings,
    AzureOpenAIVerifierClient,
    JsonCompletion,
    VerifierInfrastructureError,
)


class FakeJsonTransport:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls = []

    def complete(self, messages, *, temperature, max_tokens):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return JsonCompletion(
            payload=self.payload,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=12,
        )


def settings() -> AzureOpenAISettings:
    return AzureOpenAISettings(
        api_key="unused",
        base_url="https://example.openai.azure.com",
        model="legacy-model",
        generator_model="gemini-2.5-flash",
        verifier_model="gemini-2.5-pro",
        verifier_temperature=0.0,
        verifier_judge_max_tokens=1200,
        verifier_repair_max_tokens=2500,
    )


class AzureOpenAIVerifierClientTests(unittest.TestCase):
    def test_default_judge_budget_accommodates_reasoning_models(self) -> None:
        default_settings = AzureOpenAISettings(
            api_key="unused",
            base_url="https://gateway.example",
        )

        self.assertEqual(default_settings.verifier_judge_max_tokens, 4000)

    def test_judge_uses_separate_settings_and_tracks_cost(self) -> None:
        transport = FakeJsonTransport(
            {"status": "PASS", "score": 99, "issues": []}
        )
        client = AzureOpenAIVerifierClient(
            settings(),
            input_price_per_million=Decimal("2.50"),
            output_price_per_million=Decimal("10.00"),
            transport=transport,
        )

        decision = client.judge([{"role": "system", "content": "judge"}])

        self.assertEqual(decision.status, "PASS")
        self.assertEqual(decision.token_usage.total_tokens, 150)
        self.assertEqual(decision.token_usage.money_cost, Decimal("0.00075000"))
        self.assertEqual(decision.model, "gemini-2.5-pro")
        self.assertEqual(transport.calls[0]["temperature"], 0.0)
        self.assertEqual(transport.calls[0]["max_tokens"], 1200)

    def test_repair_uses_its_own_token_limit(self) -> None:
        transport = FakeJsonTransport(
            {
                "tagged_text": "<PERSON>Lan</PERSON>",
                "entities": [{"label": "PERSON", "value": "Lan"}],
            }
        )
        client = AzureOpenAIVerifierClient(
            settings(),
            input_price_per_million=Decimal("2.50"),
            output_price_per_million=Decimal("10.00"),
            transport=transport,
        )

        repaired = client.repair([{"role": "system", "content": "repair"}])

        self.assertEqual(repaired.entities[0].value, "Lan")
        self.assertEqual(repaired.model, "gemini-2.5-pro")
        self.assertEqual(transport.calls[0]["max_tokens"], 2500)

    def test_invalid_or_contradictory_llm_payload_is_infrastructure_error(self) -> None:
        payloads = (
            {"status": "PASS", "score": 90, "issues": [{"unexpected": True}]},
            {"status": "UNKNOWN", "score": 90, "issues": []},
            {
                "tagged_text": "<PERSON>Lan</PERSON>",
                "entities": [{"label": "PERSON", "value": "Lan"}],
                "extra": "not allowed",
            },
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                client = AzureOpenAIVerifierClient(
                    settings(),
                    input_price_per_million=Decimal("2.50"),
                    output_price_per_million=Decimal("10.00"),
                    transport=FakeJsonTransport(payload),
                )
                with self.assertRaises(VerifierInfrastructureError):
                    if "status" in payload:
                        client.judge([{"role": "system", "content": "judge"}])
                    else:
                        client.repair([{"role": "system", "content": "repair"}])


if __name__ == "__main__":
    unittest.main()
