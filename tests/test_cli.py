import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pii_factory.api import create_app
from pii_factory.main import main


class PiiFactoryCliTests(unittest.TestCase):
    def test_api_runtime_registers_default_json_taxonomy(self) -> None:
        app = create_app(offline=True)

        self.assertEqual(app.state.default_taxonomy_label_count, 44)
        self.assertTrue(app.state.default_taxonomy_version_id)
        openapi = app.openapi()
        self.assertNotIn("post", openapi["paths"]["/api/v1/taxonomies"])
        run_schema = (
            openapi["paths"]["/api/v1/runs"]["post"]["requestBody"]
            ["content"]["application/json"]["schema"]["$ref"]
        )
        self.assertTrue(run_schema.endswith("/RunConfig"))

    def test_config_run_uses_default_json_taxonomy_without_path_flag(
        self,
    ) -> None:
        config = {
            "run_name": "cli-json-taxonomy",
            "num_samples": 1,
            "focus_label": "API_KEY",
            "difficulty_distribution": {
                "easy": 0.0,
                "medium": 1.0,
                "hard": 0.0,
            },
            "sample_type_distribution": {
                "positive": 1.0,
                "pure_negative": 0.0,
                "hard_negative": 0.0,
            },
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            output = io.StringIO()
            errors = io.StringIO()
            arguments = [
                "pii-factory",
                "--offline",
                "--config",
                str(config_path),
                "--output-dir",
                str(root / "output"),
            ]

            with (
                patch("sys.argv", arguments),
                patch(
                    "pii_factory.main.uvicorn.run",
                    side_effect=AssertionError("server must not start"),
                ),
                redirect_stdout(output),
                redirect_stderr(errors),
            ):
                main()

            payload = json.loads(output.getvalue())
            self.assertEqual(payload["labels"], 44)
            self.assertEqual(payload["accepted_samples"], 1)
            self.assertEqual(payload["status"], "COMPLETED")
            self.assertIn("smoke", payload["output_path"].casefold())
            self.assertIn("not a dataset", errors.getvalue().casefold())
            self.assertEqual(payload["diagnostics"]["generated_candidates"], 1)
            self.assertEqual(payload["diagnostics"]["discarded_candidates"], 0)
            self.assertEqual(payload["diagnostics"]["task_replacements"], 0)


if __name__ == "__main__":
    unittest.main()
