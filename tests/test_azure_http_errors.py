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


def _successful_response(model: str | None = None) -> BytesIO:
    payload = {
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
    if model is not None:
        payload["model"] = model
    return BytesIO(
        json.dumps(payload).encode("utf-8")
    )


def _malformed_json_response() -> BytesIO:
    return BytesIO(
        json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"tagged_text":"unfinished'
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }
        ).encode("utf-8")
    )


def _wrapped_json_response() -> BytesIO:
    payload = {"tagged_text": "wrapped", "entities": []}
    return BytesIO(
        json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "```json\n"
                                + json.dumps(payload, ensure_ascii=False)
                                + "\n```"
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
    def test_generator_defaults_to_long_form_completion_budget(self) -> None:
        settings = AzureOpenAISettings(
            api_key="top-secret-value",
            base_url="https://gateway.example",
        )

        self.assertEqual(settings.max_tokens, 6000)

    def test_generator_retries_malformed_json_as_infrastructure_failure(self) -> None:
        settings = AzureOpenAISettings(
            api_key="top-secret-value",
            base_url="https://gateway.example",
            infrastructure_retries=1,
        )
        with (
            patch(
                "pii_factory.infrastructure.clients.urlopen",
                side_effect=[
                    _malformed_json_response(),
                    _successful_response(),
                ],
            ) as mocked_urlopen,
            patch("pii_factory.infrastructure.clients.time.sleep"),
        ):
            (
                tagged_text,
                entities,
                input_tokens,
                output_tokens,
                total_tokens,
            ) = AzureOpenAICompletionClient(settings).generate(
                [{"role": "user", "content": "Return JSON."}]
            )

        self.assertEqual(tagged_text, "test")
        self.assertEqual(entities, [])
        self.assertEqual(mocked_urlopen.call_count, 2)
        self.assertEqual(input_tokens, 20)
        self.assertEqual(output_tokens, 25)
        self.assertEqual(total_tokens, 45)

    def test_generator_accepts_json_wrapped_in_markdown_fence(self) -> None:
        settings = AzureOpenAISettings(
            api_key="top-secret-value",
            base_url="https://gateway.example",
        )
        with patch(
            "pii_factory.infrastructure.clients.urlopen",
            return_value=_wrapped_json_response(),
        ):
            (
                tagged_text,
                entities,
                input_tokens,
                output_tokens,
                total_tokens,
            ) = AzureOpenAICompletionClient(settings).generate(
                [{"role": "user", "content": "Return JSON."}]
            )

        self.assertEqual(tagged_text, "wrapped")
        self.assertEqual(entities, [])

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

    def test_standalone_worker_retries_malformed_json(self) -> None:
        settings = WorkerSettings(
            api_key="top-secret-value",
            base_url="https://gateway.example",
            infrastructure_retries=1,
        )
        with (
            patch(
                "data_generator_worker.llm_client.urlopen",
                side_effect=[
                    _malformed_json_response(),
                    _successful_response(),
                ],
            ) as mocked_urlopen,
            patch("data_generator_worker.llm_client.time.sleep"),
        ):
            completion = AzureOpenAIClient(settings).generate(
                [{"role": "user", "content": "Return JSON."}]
            )

        self.assertEqual(completion.tagged_text, "test")
        self.assertEqual(completion.entities, [])
        self.assertEqual(mocked_urlopen.call_count, 2)

    def test_verifier_transport_counts_tokens_from_invalid_json_retry(self) -> None:
        settings = AzureOpenAISettings(
            api_key="top-secret-value",
            base_url="https://gateway.example",
            infrastructure_retries=1,
        )
        with (
            patch(
                "pii_factory.infrastructure.clients.urlopen",
                side_effect=[
                    _malformed_json_response(),
                    _successful_response(),
                ],
            ),
            patch("pii_factory.infrastructure.clients.time.sleep"),
        ):
            completion = AzureOpenAIJsonTransport(settings).complete(
                [{"role": "user", "content": "Judge JSON."}],
                temperature=0.0,
                max_tokens=1200,
            )

        self.assertEqual(completion.input_tokens, 20)
        self.assertEqual(completion.output_tokens, 25)
        self.assertEqual(completion.total_tokens, 45)

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

    def test_generator_logs_provider_reported_model_identity(self) -> None:
        settings = AzureOpenAISettings(
            api_key="top-secret-value",
            base_url="https://gateway.example",
            generator_model="gemini-2.5-flash",
            infrastructure_retries=0,
        )
        with (
            patch(
                "pii_factory.infrastructure.clients.urlopen",
                return_value=_successful_response("gemini-2.5-flash"),
            ),
            self.assertLogs(
                "pii_factory.infrastructure.clients",
                level="INFO",
            ) as captured,
        ):
            AzureOpenAICompletionClient(settings).generate(
                [{"role": "user", "content": "Return JSON."}]
            )

        output = "\n".join(captured.output)
        self.assertIn("requested_model=gemini-2.5-flash", output)
        self.assertIn("response_model=gemini-2.5-flash", output)
        self.assertIn("model_status=reported_match", output)

    def test_verifier_logs_when_gateway_reports_a_different_model(self) -> None:
        settings = AzureOpenAISettings(
            api_key="top-secret-value",
            base_url="https://gateway.example",
            verifier_model="gemini-2.5-pro",
            infrastructure_retries=0,
        )
        with (
            patch(
                "pii_factory.infrastructure.clients.urlopen",
                return_value=_successful_response("gateway-fallback-model"),
            ),
            self.assertLogs(
                "pii_factory.infrastructure.clients",
                level="INFO",
            ) as captured,
        ):
            AzureOpenAIJsonTransport(settings).complete(
                [{"role": "user", "content": "Judge JSON."}],
                temperature=0.0,
                max_tokens=1200,
            )

        output = "\n".join(captured.output)
        self.assertIn("requested_model=gemini-2.5-pro", output)
        self.assertIn("response_model=gateway-fallback-model", output)
        self.assertIn("model_status=reported_different", output)

    def test_logs_when_gateway_does_not_report_serving_model(self) -> None:
        settings = AzureOpenAISettings(
            api_key="top-secret-value",
            base_url="https://gateway.example",
            generator_model="gemini-2.5-flash",
            infrastructure_retries=0,
        )
        with (
            patch(
                "pii_factory.infrastructure.clients.urlopen",
                return_value=_successful_response(),
            ),
            self.assertLogs(
                "pii_factory.infrastructure.clients",
                level="INFO",
            ) as captured,
        ):
            AzureOpenAICompletionClient(settings).generate(
                [{"role": "user", "content": "Return JSON."}]
            )

        output = "\n".join(captured.output)
        self.assertIn("response_model=<not-reported>", output)
        self.assertIn("model_status=not_reported", output)

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
