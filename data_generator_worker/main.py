from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import Settings
from .contracts import DataGenerationRequest
from .llm_client import AzureOpenAIClient
from .worker import DataGeneratorWorker, InMemoryAttemptStore


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Consumes one data.generation.requested JSON payload from stdin.")
    parser.add_argument("--input", default="-", help="Input JSON file; '-' reads stdin.")
    args = parser.parse_args()
    configure_logging()
    source = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")
    try:
        request = DataGenerationRequest.from_dict(json.load(source))
    finally:
        if source is not sys.stdin:
            source.close()

    settings = Settings.from_env()
    worker = DataGeneratorWorker(AzureOpenAIClient(settings), settings, InMemoryAttemptStore())
    event = worker.process(request)
    print(json.dumps(event.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
