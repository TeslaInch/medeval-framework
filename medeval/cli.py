"""
medeval/cli.py
~~~~~~~~~~~~~~
Command Line Interface (CLI) entrypoint for the medeval evaluation framework.

Enables researchers and engineers to execute evaluations, configure metrics,
and export results without writing Python code.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from .accuracy import BaseScorer, ExactMatchScorer, SemanticSimilarityScorer
from .benchmark import BenchmarkLoader
from .hallucination import NLIHallucinationDetector
from .models.base import BaseModelConnector
from .models.huggingface import HuggingFaceConnector
from .models.mock import MockConnector
from .models.openai_connector import OpenAIConnector
from .report import export_report_to_json
from .runner import BenchmarkRunner
from .safety.sickle_cell import SickleCellSafetyChecker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connector & Safety Mapper functions
# ---------------------------------------------------------------------------


def resolve_model_connector(
    model_name: str, device: str, trust_remote_code: bool = False
) -> BaseModelConnector:
    """Map a model string identifier to the corresponding connector class.

    Rules:
      - If it starts with 'mock-', returns MockConnector.
      - If it starts with 'gpt-' or 'openai:', returns OpenAIConnector.
      - Otherwise, defaults to HuggingFaceConnector.

    Args:
        model_name: Model identifier string.
        device: Device Index string (e.g. 'cpu', 'cuda:0').
        trust_remote_code: Boolean to trust remote code execution.

    Returns:
        An instance of BaseModelConnector.
    """
    clean_name = model_name.strip()
    if clean_name.startswith("mock-"):
        logger.info("CLI: Instantiating MockConnector for model '%s'", clean_name)
        return MockConnector(model_name=clean_name)
    elif clean_name.startswith("gpt-") or clean_name.startswith("openai:"):
        actual_name = clean_name.replace("openai:", "")
        logger.info("CLI: Instantiating OpenAIConnector for model '%s'", actual_name)
        return OpenAIConnector(model_name=actual_name)
    else:
        logger.info(
            "CLI: Instantiating HuggingFaceConnector for model '%s' on %s", clean_name, device
        )
        return HuggingFaceConnector(
            model_name=clean_name, device=device, trust_remote_code=trust_remote_code
        )


# ---------------------------------------------------------------------------
# CLI Main execution wrapper
# ---------------------------------------------------------------------------


def run_evaluation(args: argparse.Namespace) -> int:
    """Orchestrates the evaluation run process using parsed arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        Integer exit code (0 for success, non-zero for failure).
    """
    # 1. Setup logging level
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    print("=== medeval LLM Evaluation Runner ===")
    print(f"Model:    {args.model}")
    print(f"Dataset:  {args.dataset} (split: {args.split})")
    if args.limit:
        print(f"Limit:    {args.limit} samples")
    print("=====================================")

    try:
        # 2. Instantiate Model Connector
        if args.trust_remote_code:
            print(
                "\n[WARNING] --trust-remote-code is enabled. Ensure you trust the model source as this allows remote code execution.",
                file=sys.stderr,
            )

        model = resolve_model_connector(
            args.model, args.device, trust_remote_code=args.trust_remote_code
        )

        # 3. Load dataset
        loader = BenchmarkLoader(split=args.split, max_samples=args.limit)
        if args.dataset == "medqa":
            samples = loader.load_medqa()
        elif args.dataset == "pubmedqa":
            samples = loader.load_pubmedqa()
        else:
            print(f"Error: Unknown dataset '{args.dataset}'.", file=sys.stderr)
            return 1

        # 4. Instantiate scorers
        scorers: list[BaseScorer] = [ExactMatchScorer()]
        # Add Semantic similarity scorer if enabled (and not disabled by user)
        if args.use_semantic_similarity:
            scorers.append(
                SemanticSimilarityScorer(model_type=args.bertscore_model, device=args.device)
            )

        # 5. Instantiate NLI hallucination detector
        hallucination_detector = None
        if args.hallucination:
            hallucination_detector = NLIHallucinationDetector(
                model_name=args.nli_model,
                threshold=args.nli_threshold,
                device=-1 if args.device == "cpu" else 0,
            )

        # 6. Instantiate Safety checkers
        safety_checker = None
        if args.safety == "sickle_cell":
            safety_checker = SickleCellSafetyChecker()
        elif args.safety != "none":
            print(f"Error: Unsupported safety checker '{args.safety}'.", file=sys.stderr)
            return 1

        # 7. Configure and trigger Runner
        runner = BenchmarkRunner(
            model=model,
            scorers=scorers,
            hallucination_detector=hallucination_detector,
            safety_checker=safety_checker,
            framework_version=args.framework_version,
            ignore_errors=args.ignore_errors,
        )

        print("\nEvaluating samples...")
        report = runner.run(samples)

        # 8. Export report to JSON file
        export_report_to_json(report, args.output)

        # 9. Print results summary
        print("\n=== Evaluation Results Summary ===")
        print(f"Total Samples evaluated: {report.total_samples}")
        for metric_name, val in report.metrics.items():
            print(f"  - {metric_name.replace('_', ' ').capitalize()}: {val:.4f}")
        print(f"Safety violations count: {len(report.safety_violations)}")
        print(f"Report exported to: {args.output}")
        print("==================================")
        return 0

    except Exception as exc:
        print(f"\nEvaluation failed with error: {exc}", file=sys.stderr)
        logger.exception("CLI execution crash.")
        return 1


# ---------------------------------------------------------------------------
# Argument Parser Construction
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        argparse.ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="medeval",
        description="Rigorous evaluation framework for medical LLMs.",
    )
    # Core execution inputs
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID. Prefix with 'mock-' for mocks, 'openai:' or 'gpt-' for OpenAI, or use HuggingFace repo path.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["medqa", "pubmedqa"],
        help="Medical dataset benchmark to evaluate on.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the final JSON EvaluationReport will be written.",
    )

    # Optional dataset configurations
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate (e.g. 'test', 'validation', 'train').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of samples to process. Defaults to running all samples.",
    )

    # Safety metrics
    parser.add_argument(
        "--safety",
        default="sickle_cell",
        choices=["sickle_cell", "none"],
        help="Safety checker to execute. Defaults to 'sickle_cell'.",
    )

    # Scorer specifics
    parser.add_argument(
        "--no-semantic-similarity",
        dest="use_semantic_similarity",
        action="store_false",
        help="Disable BERTScore calculations to speed up run time.",
    )
    parser.set_defaults(use_semantic_similarity=True)
    parser.add_argument(
        "--bertscore-model",
        default="distilbert-base-uncased",
        help="BERTScore embedding model checkpoint. Defaults to 'distilbert-base-uncased'.",
    )

    # NLI hallucination specifics
    parser.add_argument(
        "--no-hallucination",
        dest="hallucination",
        action="store_false",
        help="Disable NLI hallucination check to speed up run time.",
    )
    parser.set_defaults(hallucination=True)
    parser.add_argument(
        "--nli-model",
        default="cross-encoder/nli-deberta-v3-large",
        help="NLI model checkpoint for hallucination verification. Defaults to 'cross-encoder/nli-deberta-v3-large'.",
    )
    parser.add_argument(
        "--nli-threshold",
        type=float,
        default=0.5,
        help="Hallucination threshold (sum of neutral + contradiction). Default 0.5.",
    )

    # Execution controls
    parser.add_argument(
        "--device",
        default="cpu",
        help="Execution device Index for Hugging Face and PyTorch models (e.g. 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for Hugging Face models. WARNING: Allows remote code execution.",
    )
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Catch exceptions on individual sample evaluations and proceed with remaining run.",
    )
    parser.add_argument(
        "--framework-version",
        default="0.1.0",
        help="Version of medeval framework. Default '0.1.0'.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )
    return parser


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Main CLI entrypoint function.

    Args:
        argv: Command-line arguments override (used for testing).
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    sys.exit(run_evaluation(args))


if __name__ == "__main__":
    main()
