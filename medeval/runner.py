"""
medeval/runner.py
~~~~~~~~~~~~~~~~~
Pipeline orchestrator and execution runner for medical LLM benchmarks.

Provides the ``BenchmarkRunner`` class, which ties together:
    - Model Connectors (local, API, mock)
    - Benchmark datasets (MedicalEvalSample)
    - Accuracy scorers (ExactMatchScorer, SemanticSimilarityScorer)
    - Hallucination detection (NLIHallucinationDetector)
    - Safety verification (SickleCellSafetyChecker)

Outputs a fully aggregated ``EvaluationReport``.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from .accuracy import BaseScorer, ExactMatchScorer, SemanticSimilarityScorer
from .hallucination import NLIHallucinationDetector
from .models.base import BaseModelConnector
from .report import ReportGenerator
from .safety.sickle_cell import SickleCellSafetyChecker
from .structures import EvaluationReport, MedicalEvalSample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt Formatter Helper
# ---------------------------------------------------------------------------


def default_prompt_formatter(sample: MedicalEvalSample) -> str:
    """Default formatter converting a sample's metadata and question into a prompt.

    Args:
        sample: The target evaluation sample.

    Returns:
        A formatted string ready to be consumed by the model connector.
    """
    parts: list[str] = []

    # 1. Inject clinical context if present (e.g., PubMedQA context)
    context = sample.metadata.get("context")
    if context:
        parts.append(f"Context: {context}")

    # 2. Append the main question
    parts.append(f"Question: {sample.question}")

    # 3. Append multiple-choice options if present (e.g., MedQA)
    choices = sample.metadata.get("choices")
    if isinstance(choices, dict):
        parts.append("Choices:")
        for key, val in choices.items():
            parts.append(f"  {key}: {val}")
        parts.append("Select the single best answer choice.")
    elif isinstance(choices, list):
        parts.append("Choices:")
        for choice in choices:
            parts.append(f"  - {choice}")
        parts.append("Select the single best answer choice.")

    # 4. Standard instruct format ending
    parts.append("Answer:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Orchestrates model execution, scoring, safety checks, and reporting.

    Runs a model connector over a sequence of evaluation samples, feeds the outputs
    through the configured evaluation engines, and generates a structured report.

    Args:
        model: An implementation of ``BaseModelConnector``.
        scorers: Optional list of scorers (e.g. ExactMatchScorer, SemanticSimilarityScorer).
            If none are provided, a default ``ExactMatchScorer`` is used.
        hallucination_detector: Optional NLI-based detector to scan for hallucinations.
        safety_checker: Optional safety checker (e.g. ``SickleCellSafetyChecker``).
        framework_version: Reproducibility tracker version string. Defaults to '0.1.0'.
        prompt_formatter: Optional callable to format prompts. Defaults to
            ``default_prompt_formatter``.
        ignore_errors: If True, log errors during single-sample evaluation and
            continue with the remaining samples. Defaults to False.
    """

    def __init__(
        self,
        model: BaseModelConnector,
        scorers: list[BaseScorer] | None = None,
        hallucination_detector: NLIHallucinationDetector | None = None,
        safety_checker: SickleCellSafetyChecker | None = None,
        framework_version: str = "0.1.0",
        prompt_formatter: callable | None = None,  # type: ignore[valid-type]
        ignore_errors: bool = False,
    ) -> None:
        """Initialise runner configuration."""
        self._model = model
        self._scorers = scorers if scorers is not None else [ExactMatchScorer()]
        self._hallucination_detector = hallucination_detector
        self._safety_checker = safety_checker
        self._framework_version = framework_version
        self._prompt_formatter = prompt_formatter or default_prompt_formatter
        self._ignore_errors = ignore_errors

    def _determine_y_prob(
        self,
        prediction: str,
        probs: list[float],
        choices: dict[str, Any] | None,
    ) -> float | None:
        """Extract the model's confidence probability (y_prob) for its prediction.

        For multiple choice tasks, maps the model's predicted letter choice (e.g., 'A')
        to the corresponding probability index. Falls back to the maximum probability
        score if mapping fails.

        Args:
            prediction: Normalized model prediction text.
            probs: Sequence of token or class probabilities.
            choices: Choice mapping dictionary (e.g., {'A': 'Metformin'}).

        Returns:
            The extracted float probability score, or None.
        """
        if not probs:
            return None

        # If it's a multiple choice task and probability dimensions align
        if choices and len(probs) == len(choices):
            # Clean up the prediction to search for key matching
            pred_clean = prediction.strip().upper()
            keys = [str(k).upper() for k in choices.keys()]

            # Try exact match on key (e.g., prediction is 'A')
            if pred_clean in keys:
                idx = keys.index(pred_clean)
                return probs[idx]

            # Try substring match on key
            for i, key in enumerate(keys):
                if key in pred_clean:
                    return probs[i]

        # Fallback to the peak generation confidence / max probability
        return max(probs)

    def evaluate_sample(self, sample: MedicalEvalSample) -> MedicalEvalSample | None:
        """Generate response and calculate metrics for a single sample.

        Args:
            sample: The input evaluation sample.

        Returns:
            An updated copy of the sample with prediction, probabilities,
            and metrics embedded in its metadata. Returns None if evaluation fails
            and ignore_errors is True.
        """
        try:
            # 1. Format prompt and generate model response
            prompt = self._prompt_formatter(sample)
            prediction = self._model.generate(prompt)
            probs = self._model.generate_probabilities(prompt)

            # 2. Extract choice metadata if present
            choices = sample.metadata.get("choices")

            # 3. Extract confidence probability (y_prob)
            y_prob = self._determine_y_prob(prediction, probs, choices)

            # 4. Compute correctness (y_true) using standard exact-match comparison
            # Always compute a baseline y_true for calibration calculation (ECE)
            em_scorer = ExactMatchScorer()
            y_true = 1 if em_scorer.score([prediction], [sample.ground_truth]) == 1.0 else 0

            # 5. Populate metadata dictionary
            metadata = dict(sample.metadata)
            metadata["y_true"] = y_true
            if y_prob is not None:
                metadata["y_prob"] = y_prob

            # 6. Apply accuracy scorers
            for scorer in self._scorers:
                if isinstance(scorer, SemanticSimilarityScorer):
                    f1 = scorer.score([prediction], [sample.ground_truth])
                    metadata["bert_score_f1"] = float(f1)

            # 7. Apply NLI hallucination detector
            if self._hallucination_detector is not None:
                # If context is missing in metadata, fallback to prompt or question
                premise = sample.metadata.get("context", sample.question)
                nli_res = self._hallucination_detector.detect(
                    premise=premise, hypothesis=prediction
                )
                metadata["is_hallucination"] = nli_res.is_hallucination

            # 8. Apply safety checks
            if self._safety_checker is not None:
                violations = self._safety_checker.check_contraindications(prediction)
                metadata["safety_violations"] = violations

            # 9. Return reconstructed immutable sample
            return replace(
                sample,
                model_prediction=prediction,
                prediction_probabilities=probs,
                metadata=metadata,
            )

        except Exception as exc:
            logger.exception("Error evaluating sample ID %s: %s", sample.id, exc)
            if self._ignore_errors:
                return None
            raise exc

    def run(self, samples: list[MedicalEvalSample]) -> EvaluationReport:
        """Run the complete benchmark execution loop over a set of samples.

        Args:
            samples: List of MedicalEvalSample data contracts to evaluate.

        Returns:
            The aggregated macro EvaluationReport.
        """
        if not samples:
            raise ValueError("Runner requires at least one evaluation sample.")

        evaluated_samples: list[MedicalEvalSample] = []
        for sample in samples:
            res = self.evaluate_sample(sample)
            if res is not None:
                evaluated_samples.append(res)

        if not evaluated_samples:
            raise ValueError("All samples failed to evaluate and ignore_errors was set to True.")

        # Build and return the final report
        generator = ReportGenerator(
            model_name=self._model.model_name,
            framework_version=self._framework_version,
            samples=evaluated_samples,
        )
        return generator.generate()
