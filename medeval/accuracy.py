"""
medeval/accuracy.py
~~~~~~~~~~~~~~~~~~~
Accuracy scoring utilities for medical LLM evaluation.

Provides two scoring strategies:

* ``ExactMatchScorer``  — normalised string equality, appropriate for
  multiple-choice benchmarks (e.g. MedQA, USMLE).
* ``SemanticSimilarityScorer`` — BERTScore F1 via Hugging Face ``evaluate``,
  appropriate for open-ended clinical text where synonym variance is expected.

Both classes share the same ``score()`` signature so they are interchangeable
in pipeline code.
"""

from __future__ import annotations

import logging
import re
import string
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class BaseScorer(ABC):
    """Abstract base class that all accuracy scorers must implement.

    Enforces a common ``score()`` interface so scorers are interchangeable
    inside evaluation pipelines.
    """

    @abstractmethod
    def score(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """Compute an aggregate accuracy score over a batch of samples.

        Args:
            predictions: Model-generated answer strings, one per sample.
            references: Ground-truth answer strings, one per sample.
                Must have the same length as ``predictions``.

        Returns:
            A scalar accuracy score in [0, 1].

        Raises:
            ValueError: If ``predictions`` and ``references`` differ in length
                or if either list is empty.
        """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Normalise a string for robust exact-match comparison.

    Applies lower-casing, punctuation removal, and whitespace collapsing so
    that ``"Metformin."`` and ``"metformin"`` are treated as equal.

    Args:
        text: Raw input string to normalise.

    Returns:
        Normalised string with no leading/trailing whitespace.
    """
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse internal whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _validate_inputs(predictions: List[str], references: List[str]) -> None:
    """Validate that scorer inputs are non-empty and length-matched.

    Args:
        predictions: List of model prediction strings.
        references: List of ground-truth reference strings.

    Raises:
        ValueError: If either list is empty, or if the lengths differ.
    """
    if not predictions or not references:
        raise ValueError(
            "Both 'predictions' and 'references' must be non-empty lists. "
            f"Received lengths: predictions={len(predictions)}, "
            f"references={len(references)}."
        )
    if len(predictions) != len(references):
        raise ValueError(
            "Lengths of 'predictions' and 'references' must match. "
            f"Got predictions={len(predictions)}, references={len(references)}."
        )


# ---------------------------------------------------------------------------
# Exact-match scorer
# ---------------------------------------------------------------------------


class ExactMatchScorer(BaseScorer):
    """Normalised exact-match accuracy scorer.

    Suitable for multiple-choice benchmarks (e.g. MedQA, USMLE) where the
    expected answer is a fixed string such as ``"A"`` or ``"Metformin"``.

    Normalisation (lower-case, strip punctuation, collapse whitespace) is
    applied before comparison to avoid penalising superficial formatting
    differences.

    Example:
        >>> scorer = ExactMatchScorer()
        >>> scorer.score(["Metformin.", "Aspirin"], ["metformin", "Ibuprofen"])
        0.5
    """

    def score(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """Compute fraction of predictions that exactly match their reference.

        Args:
            predictions: Model-generated answer strings.
            references: Ground-truth answer strings.

        Returns:
            Exact-match accuracy as a float in [0.0, 1.0].

        Raises:
            ValueError: If inputs are empty or length-mismatched.
        """
        _validate_inputs(predictions, references)

        matches: int = sum(
            _normalize(pred) == _normalize(ref)
            for pred, ref in zip(predictions, references)
        )
        accuracy: float = matches / len(predictions)

        logger.debug(
            "ExactMatchScorer: %d/%d correct → accuracy=%.4f",
            matches,
            len(predictions),
            accuracy,
        )
        return accuracy


# ---------------------------------------------------------------------------
# Semantic-similarity scorer (BERTScore via Hugging Face evaluate)
# ---------------------------------------------------------------------------


class SemanticSimilarityScorer(BaseScorer):
    """BERTScore-based semantic similarity scorer using Hugging Face ``evaluate``.

    Loads ``evaluate.load("bertscore")`` lazily (at first ``score()`` call) so
    the package remains importable in environments where the model checkpoint
    is not yet downloaded.

    BERTScore F1 is returned as the aggregate accuracy signal. It correlates
    well with human judgement for open-ended clinical text where synonym
    variance makes exact-match inappropriate (e.g. ``"myocardial infarction"``
    vs. ``"heart attack"``).

    Args:
        model_type: The BERT-family model used for token embeddings.
            Defaults to ``"distilbert-base-uncased"`` for speed; swap for
            ``"microsoft/deberta-xlarge-mnli"`` for maximum quality.
        device: Torch device string (e.g. ``"cpu"``, ``"cuda:0"``). Defaults
            to ``"cpu"``.

    Example:
        >>> scorer = SemanticSimilarityScorer(model_type="distilbert-base-uncased")
        >>> f1 = scorer.score(["myocardial infarction"], ["heart attack"])
        >>> 0.0 <= f1 <= 1.0
        True
    """

    def __init__(
        self,
        model_type: str = "distilbert-base-uncased",
        device: Optional[str] = None,
    ) -> None:
        """Initialise the scorer with model and device configuration.

        Args:
            model_type: HuggingFace model identifier for BERTScore embeddings.
            device: Torch device override. ``None`` lets ``evaluate`` choose.
        """
        self._model_type: str = model_type
        self._device: Optional[str] = device
        # Lazy-loaded — set to None until first score() call.
        self._metric = None  # type: ignore[assignment]

    def _load_metric(self) -> None:
        """Lazily load the BERTScore metric from the ``evaluate`` library.

        Raises:
            ImportError: If the ``evaluate`` package is not installed.
        """
        try:
            import evaluate  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'evaluate' package is required for SemanticSimilarityScorer. "
                "Install it with: pip install evaluate"
            ) from exc

        if self._metric is None:
            logger.info(
                "Loading BERTScore metric with model_type='%s'.", self._model_type
            )
            self._metric = evaluate.load("bertscore")

    def score(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """Compute mean BERTScore F1 over a batch of prediction-reference pairs.

        Args:
            predictions: Model-generated answer strings.
            references: Ground-truth answer strings.

        Returns:
            Mean BERTScore F1 across all samples, as a float in [0.0, 1.0].

        Raises:
            ValueError: If inputs are empty or length-mismatched.
            ImportError: If the ``evaluate`` package is not installed.
        """
        _validate_inputs(predictions, references)
        self._load_metric()

        kwargs = {
            "predictions": predictions,
            "references": references,
            "model_type": self._model_type,
        }
        if self._device is not None:
            kwargs["device"] = self._device

        results = self._metric.compute(**kwargs)  # type: ignore[union-attr]
        f1_scores: List[float] = results["f1"]
        mean_f1: float = sum(f1_scores) / len(f1_scores)

        logger.debug(
            "SemanticSimilarityScorer: mean BERTScore F1=%.4f over %d samples.",
            mean_f1,
            len(predictions),
        )
        return mean_f1
