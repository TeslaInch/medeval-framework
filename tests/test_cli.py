"""
tests/test_cli.py
~~~~~~~~~~~~~~~~~
Unit and integration tests for the ``medeval`` command-line interface.

Uses mocking to test argument parsing, connector resolution, and pipeline
execution 100% offline.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from medeval.cli import create_parser, resolve_model_connector, run_evaluation
from medeval.models.huggingface import HuggingFaceConnector
from medeval.models.mock import MockConnector
from medeval.models.openai_connector import OpenAIConnector
from medeval.structures import EvaluationReport

# ---------------------------------------------------------------------------
# Test Connector Resolution
# ---------------------------------------------------------------------------


class TestCLIConnectorResolution:
    """Verifies that model name strings resolve to correct connector types."""

    def test_resolves_mock_connector(self) -> None:
        """Names starting with 'mock-' must resolve to MockConnector."""
        conn = resolve_model_connector("mock-llama-3", device="cpu")
        assert isinstance(conn, MockConnector)
        assert conn.model_name == "mock-llama-3"

    def test_resolves_openai_connector(self) -> None:
        """Names starting with 'gpt-' or 'openai:' must resolve to OpenAIConnector."""
        conn1 = resolve_model_connector("gpt-4o", device="cpu")
        assert isinstance(conn1, OpenAIConnector)
        assert conn1.model_name == "gpt-4o"

        conn2 = resolve_model_connector("openai:gpt-3.5-turbo", device="cpu")
        assert isinstance(conn2, OpenAIConnector)
        assert conn2.model_name == "gpt-3.5-turbo"

    def test_resolves_huggingface_connector_default(self) -> None:
        """Any other model name must default to HuggingFaceConnector."""
        conn = resolve_model_connector("meta-llama/Llama-2-7b-chat-hf", device="cuda:0")
        assert isinstance(conn, HuggingFaceConnector)
        assert conn.model_name == "meta-llama/Llama-2-7b-chat-hf"
        assert conn._device == "cuda:0"


# ---------------------------------------------------------------------------
# Test CLI Argument Parser
# ---------------------------------------------------------------------------


class TestCLIArgumentParser:
    """Tests verify the argparse parsing and default values."""

    def test_parser_requires_mandatory_arguments(self) -> None:
        """Argparse must raise an error if required arguments are missing."""
        parser = create_parser()
        # Missing --model, --dataset, and --output
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_accepts_valid_arguments(self) -> None:
        """Argparse must parse valid values correctly and set standard defaults."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "--model",
                "mock-model",
                "--dataset",
                "medqa",
                "--output",
                "report.json",
                "--limit",
                "10",
                "--device",
                "cuda",
            ]
        )
        assert args.model == "mock-model"
        assert args.dataset == "medqa"
        assert args.output == "report.json"
        assert args.limit == 10
        assert args.device == "cuda"
        # Verify defaults
        assert args.split == "test"
        assert args.safety == "sickle_cell"
        assert args.use_semantic_similarity is True
        assert args.hallucination is True
        assert args.ignore_errors is False

    def test_parser_invalid_dataset_choice(self) -> None:
        """Argparse must raise SystemExit (error) for invalid dataset options."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--model",
                    "mock-model",
                    "--dataset",
                    "invalid_dataset",
                    "--output",
                    "report.json",
                ]
            )


# ---------------------------------------------------------------------------
# Test Run Evaluation (Mocked Execution)
# ---------------------------------------------------------------------------


class TestCLIEvaluationExecution:
    """Verifies orchestration flow of run_evaluation using mocked runner/loaders."""

    @patch("medeval.cli.BenchmarkLoader")
    @patch("medeval.cli.BenchmarkRunner")
    @patch("medeval.cli.export_report_to_json")
    def test_run_evaluation_workflow_success(
        self,
        mock_export: MagicMock,
        mock_runner_cls: MagicMock,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verifies that run_evaluation coordinates resources and completes successfully."""
        # 1. Setup mocks
        mock_loader = MagicMock()
        mock_loader.load_medqa.return_value = []
        mock_loader_cls.return_value = mock_loader

        mock_runner = MagicMock()
        mock_report = EvaluationReport(
            model_name="mock-model",
            framework_version="0.1.0",
            total_samples=0,
            metrics={"accuracy": 1.0},
            safety_violations=[],
        )
        mock_runner.run.return_value = mock_report
        mock_runner_cls.return_value = mock_runner

        # 2. Build mock args
        args = argparse.Namespace(
            model="mock-model",
            dataset="medqa",
            split="test",
            limit=5,
            output="test_output.json",
            safety="sickle_cell",
            use_semantic_similarity=True,
            bertscore_model="distilbert-base-uncased",
            hallucination=True,
            nli_model="cross-encoder/nli-deberta-v3-large",
            nli_threshold=0.5,
            device="cpu",
            ignore_errors=False,
            framework_version="0.1.0",
            verbose=True,
            trust_remote_code=False,
        )

        # 3. Execute
        exit_code = run_evaluation(args)

        # 4. Verify interactions
        assert exit_code == 0
        mock_loader_cls.assert_called_once_with(split="test", max_samples=5)
        mock_loader.load_medqa.assert_called_once()

        mock_runner_cls.assert_called_once()
        mock_runner.run.assert_called_once()
        mock_export.assert_called_once_with(mock_report, "test_output.json")

    @patch("medeval.cli.BenchmarkLoader")
    def test_run_evaluation_fails_on_unsupported_safety(
        self,
        mock_loader_cls: MagicMock,
    ) -> None:
        """Verify that passing an unsupported safety checker triggers non-zero exit."""
        args = argparse.Namespace(
            model="mock-model",
            dataset="medqa",
            split="test",
            limit=5,
            output="test_output.json",
            safety="unsupported_safety_checker",
            use_semantic_similarity=False,
            hallucination=False,
            device="cpu",
            ignore_errors=False,
            framework_version="0.1.0",
            verbose=False,
        )
        exit_code = run_evaluation(args)
        assert exit_code == 1
