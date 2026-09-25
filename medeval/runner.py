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

import json
import logging
import re
from dataclasses import asdict, replace

from .accuracy import BaseScorer, ExactMatchScorer, SemanticSimilarityScorer
from .answer_extraction import extract_answer_choice
from .hallucination import NLIHallucinationDetector
from .models.base import BaseModelConnector
from .report import ReportGenerator
from .safety.base import BaseSafetyChecker
from .safety.suite import SafetySuite
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
        safety_checker: Optional safety checker or list of safety checkers.
        framework_version: Reproducibility tracker version string. Defaults to '0.1.0'.
        prompt_formatter: Optional callable to format prompts. Defaults to
            ``default_prompt_formatter``.
        ignore_errors: If True, log errors during single-sample evaluation and
            continue with the remaining samples. Defaults to False.
        checkpoint_path: Optional path to a JSONL file for saving and resuming
            evaluations incrementally.
    """

    def __init__(
        self,
        model: BaseModelConnector,
        scorers: list[BaseScorer] | None = None,
        hallucination_detector: NLIHallucinationDetector | bool | None = True,
        safety_checker: BaseSafetyChecker | list[BaseSafetyChecker] | None = None,
        framework_version: str | None = None,
        prompt_formatter: callable | None = None,  # type: ignore[valid-type]
        ignore_errors: bool = False,
        checkpoint_path: str | None = None,
        self_consistency_samples: int = 0,
    ) -> None:
        """Initialise runner configuration."""
        from . import __version__

        self._model = model
        self._scorers = scorers if scorers is not None else [ExactMatchScorer()]

        if hallucination_detector is True:
            from .hallucination import NLIHallucinationDetector

            self._hallucination_detector: NLIHallucinationDetector | None = (
                NLIHallucinationDetector()
            )
        elif hallucination_detector is False or hallucination_detector is None:
            self._hallucination_detector = None
        else:
            self._hallucination_detector = hallucination_detector

        self._safety_checker: BaseSafetyChecker | None
        if isinstance(safety_checker, list):
            self._safety_checker = SafetySuite(safety_checker)
        else:
            self._safety_checker = safety_checker
        self._framework_version = framework_version or __version__
        self._prompt_formatter = prompt_formatter or default_prompt_formatter
        self._ignore_errors = ignore_errors
        self._checkpoint_path = checkpoint_path
        self._self_consistency_samples = self_consistency_samples

    def _determine_y_prob(
        self,
        probs: list[float],
    ) -> float | None:
        """Extract the model's confidence probability (y_prob) for its prediction.

        For multiple choice tasks, maps the model's predicted letter choice (e.g., 'A')
        to the corresponding probability index. Falls back to the maximum probability
        score if mapping fails.

        Args:
            probs: Sequence of token or class probabilities.

        Returns:
            The extracted float probability score, or None.
        """
        if not probs:
            return None

        # For generative models, probs is a sequence of token probabilities.
        # Max is statistically flawed (often ~1.0 for punctuation). Mean represents average confidence.
        return sum(probs) / len(probs)

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

            # Extract the actual answer choice before computing exact match
            extracted_answer = extract_answer_choice(prediction, choices)

            # Extract confidence probability (y_prob)
            y_prob: float | None = None
            if not probs:
                if self._self_consistency_samples > 0:
                    logger.info("Falling back to self-consistency resampling.")
                    sampled_preds = self._model.generate_n(
                        prompt, n=self._self_consistency_samples, temperature=0.7
                    )
                    match_count = sum(
                        1
                        for sp in sampled_preds
                        if extract_answer_choice(sp, choices) == extracted_answer
                    )
                    y_prob = float(match_count) / self._self_consistency_samples
            else:
                y_prob = self._determine_y_prob(probs)

            # 4. Clean CoT reasoning blocks before passing to semantic checkers
            # This prevents 512-token truncation limits and focuses evaluation on the final answer.
            prediction_clean = re.sub(
                r"<think>.*?</think>", "", prediction, flags=re.DOTALL
            ).strip()
            if not prediction_clean:
                prediction_clean = prediction.strip()

            # 5. Apply NLI hallucination detector
            is_hallucination = None
            if self._hallucination_detector is not None:
                if prediction_clean:
                    premise = f"Context: {sample.question}\nFact: {sample.ground_truth}"
                    nli_res = self._hallucination_detector.detect(
                        ground_truth=premise, model_prediction=prediction_clean
                    )
                    is_hallucination = nli_res.is_hallucination

            if is_hallucination is not None:
                sample.metadata["is_hallucination"] = is_hallucination
            # 6. SMART FALLBACK LOGIC: Try Exact Match FIRST
            metadata = dict(sample.metadata)
            em_scorer = ExactMatchScorer()

            if choices:
                eval_val = choices.get(extracted_answer, extracted_answer)
            else:
                eval_val = prediction_clean

            y_true = 1 if em_scorer.score([eval_val], [sample.ground_truth]) == 1.0 else 0

            # If Exact Match failed, fallback to Semantic Similarity
            if y_true == 0:
                sem_scorer = SemanticSimilarityScorer()
                f1 = sem_scorer.score([prediction_clean], [sample.ground_truth])
                metadata["bert_score_f1"] = float(f1)

                if float(f1) > 0.75:
                    y_true = 1

                    # NLI override: Semantic similarity cannot override a logical medical contradiction
                    if is_hallucination is True:
                        logger.info(
                            "NLI override: F1 > 0.75 but NLI flagged contradiction. Setting y_true=0"
                        )
                        y_true = 0
            else:
                metadata["bert_score_f1"] = 1.0

            # 7. Populate metadata dictionary
            metadata["y_true"] = y_true
            if y_prob is not None:
                metadata["y_prob"] = y_prob
            metadata["is_hallucination"] = is_hallucination

            # 8. Apply safety checks
            if self._safety_checker is not None:
                violations = self._safety_checker.check_contraindications(prediction_clean)
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
        evaluated_ids: set[str] = set()

        # Load checkpoint if it exists
        if self._checkpoint_path:
            import os

            if os.path.exists(self._checkpoint_path):
                logger.info("Found checkpoint at %s, loading...", self._checkpoint_path)
                try:
                    with open(self._checkpoint_path, encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            data = json.loads(line)
                            sample = MedicalEvalSample(**data)
                            evaluated_samples.append(sample)
                            evaluated_ids.add(sample.id)
                    logger.info("Loaded %d previously evaluated samples.", len(evaluated_samples))
                except Exception as exc:
                    logger.warning("Failed to load checkpoint %s: %s", self._checkpoint_path, exc)

        for sample in samples:
            if sample.id in evaluated_ids:
                logger.debug("Skipping already evaluated sample %s", sample.id)
                continue

            res = self.evaluate_sample(sample)
            if res is not None:
                evaluated_samples.append(res)
                if self._checkpoint_path:
                    try:
                        with open(self._checkpoint_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(asdict(res)) + "\n")
                    except Exception as exc:
                        logger.warning(
                            "Failed to write to checkpoint %s: %s", self._checkpoint_path, exc
                        )

        if not evaluated_samples:
            raise ValueError("All samples failed to evaluate and ignore_errors was set to True.")

        # Build and return the final report
        generator = ReportGenerator(
            model_name=self._model.model_name,
            framework_version=self._framework_version,
            samples=evaluated_samples,
        )
        return generator.generate()
