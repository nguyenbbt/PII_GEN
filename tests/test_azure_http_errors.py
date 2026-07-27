from __future__ import annotations

from io import BytesIO
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from data_generator_worker.config import Settings as WorkerSettings
from data_generator_worker.llm_client import AzureOpenAIClient
from pii_factory.infrastructure.clients import (
    AzureOpenAICompletionClient,
    AzureOpenAIJsonTransport,
    AzureOpenAISettings,
)


def _successful_response() -> BytesIO:
    return BytesIO(
        json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"tagged_text": "test", "entities": []}
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        ).encode("utf-8")
    )


class AzureOpenAIHttpErrorTests(unittest.TestCase):
    def test_standalone_worker_uses_openai_compatible_gateway_request(self) -> None:
        settings = WorkerSettings(
            api_key="top-secret-value",
            base_url="https://gateway.example",
            model="legacy-model",
            generator_model="gemini-2.5-flash",
            deployment_name="gpt-4.1",
            infrastructure_retries=0,
        )
        with patch(
            "data_generator_worker.llm_client.urlopen",
            return_value=_successful_response(),
        ) as mocked_urlopen:
            AzureOpenAIClient(settings).generate(
                [{"role": "user", "content": "Return JSON."}]
            )

        request = mocked_urlopen.call_args.args[0]
        request_body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request_body["model"], "gemini-2.5-flash")
        self.assertEqual(
            request.full_url,
            "https://gateway.example/chat/completions",
        )
        self.assertEqual(
            request.get_header("Authorization"),
            "Bearer top-secret-value",
        )

    def test_generator_sends_configured_model_to_gateway(self) -> None:
        settings = AzureOpenAISettings(
            api_key="top-secret-value",
            base_url="https://gateway.example",
            model="legacy-model",
            generator_model="gemini-2.5-flash",
            verifier_model="gemini-2.5-pro",
            deployment_name="gpt-4.1",
            infrastructure_retries=0,
        )
        with patch(
            "pii_factory.infrastructure.clients.urlopen",
            return_value=_successful_response(),
        ) as mocked_urlopen:
            AzureOpenAICompletionClient(settings).generate(
                [{"role": "user", "content": "Return JSON."}]
            )

        request = mocked_urlopen.call_args.args[0]
        request_body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request_body["model"], "gemini-2.5-flash")
        self.assertEqual(
            request.full_url,
            "https://gateway.example/chat/completions",
        )
        self.assertEqual(
            request.get_header("Authorization"),
            "Bearer top-secret-value",
        )
        self.assertIsNone(request.get_header("Api-key"))

    def test_verifier_sends_its_own_model_to_gateway(self) -> None:
        settings = AzureOpenAISettings(
            api_key="top-secret-value",
            base_url="https://gateway.example",
            model="legacy-model",
            generator_model="gemini-2.5-flash",
            verifier_model="gemini-2.5-pro",
            infrastructure_retries=0,
        )
        with patch(
            "pii_factory.infrastructure.clients.urlopen",
            return_value=_successful_response(),
        ) as mocked_urlopen:
            AzureOpenAIJsonTransport(settings).complete(
                [{"role": "user", "content": "Judge JSON."}],
                temperature=0.0,
                max_tokens=1200,
            )

        request = mocked_urlopen.call_args.args[0]
        request_body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request_body["model"], "gemini-2.5-pro")

    def test_legacy_model_remains_the_fallback_for_both_roles(self) -> None:
        settings = AzureOpenAISettings(
            api_key="top-secret-value",
            base_url="https://gateway.example",
            model="legacy-model",
        )

        self.assertEqual(settings.effective_generator_model, "legacy-model")
        self.assertEqual(settings.effective_verifier_model, "legacy-model")

    def test_native_azure_endpoint_keeps_deployment_route_and_api_key(self) -> None:
        settings = AzureOpenAISettings(
            api_key="top-secret-value",
            base_url="https://resource.openai.azure.com",
            deployment_name="gpt-4.1",
            infrastructure_retries=0,
        )
        with patch(
            "pii_factory.infrastructure.clients.urlopen",
            return_value=_successful_response(),
        ) as mocked_urlopen:
            AzureOpenAICompletionClient(settings).generate(
                [{"role": "user", "content": "Return JSON."}]
            )

        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://resource.openai.azure.com/openai/deployments/"
            "gpt-4.1/chat/completions?api-version=2024-08-01-preview",
        )
        self.assertEqual(request.get_header("Api-key"), "top-secret-value")
        self.assertIsNone(request.get_header("Authorization"))

    def test_generator_surfaces_safe_azure_error_details(self) -> None:
        settings = AzureOpenAISettings(
            api_key="top-secret-value",
            base_url="https://gateway.example",
            deployment_name="gpt-4.1",
            infrastructure_retries=0,
        )
        response_body = BytesIO(
            b'{"error":{"code":"invalid_request_error",'
            b'"message":"Unsupported endpoint shape.","param":"model"}}'
        )
        error = HTTPError(
            url="https://gateway.example",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=response_body,
        )

        with patch(
            "pii_factory.infrastructure.clients.urlopen",
            side_effect=error,
        ):
            with self.assertRaises(RuntimeError) as raised:
                AzureOpenAICompletionClient(settings).generate(
                    [{"role": "user", "content": "Return JSON."}]
                )

        message = str(raised.exception)
        self.assertIn("HTTP 400", message)
        self.assertIn("invalid_request_error", message)
        self.assertIn("Unsupported endpoint shape.", message)
        self.assertNotIn(settings.api_key, message)


if __name__ == "__main__":
    unittest.main()
