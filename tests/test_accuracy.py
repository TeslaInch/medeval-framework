"""
tests/test_accuracy.py
~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ``medeval.accuracy`` — ExactMatchScorer and
SemanticSimilarityScorer.

The ``SemanticSimilarityScorer`` tests mock the HuggingFace ``evaluate``
library so no model checkpoint is downloaded during the test run.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest

from medeval.accuracy import ExactMatchScorer, SemanticSimilarityScorer, _normalize


# ---------------------------------------------------------------------------
# _normalize helper
# ---------------------------------------------------------------------------


class TestNormalize:
    """Tests for the internal string normalisation helper."""

    def test_lowercases_input(self) -> None:
        """Should convert all characters to lowercase."""
        assert _normalize("METFORMIN") == "metformin"

    def test_removes_punctuation(self) -> None:
        """Should strip trailing and internal punctuation."""
        assert _normalize("Metformin.") == "metformin"
        assert _normalize("A,B,C") == "abc"

    def test_collapses_whitespace(self) -> None:
        """Should reduce multiple spaces to a single space."""
        assert _normalize("  heart   attack  ") == "heart attack"

    def test_empty_string_returns_empty(self) -> None:
        """Normalising an empty string should return an empty string."""
        assert _normalize("") == ""


# ---------------------------------------------------------------------------
# ExactMatchScorer
# ---------------------------------------------------------------------------


class TestExactMatchScorer:
    """Tests for ExactMatchScorer."""

    def setup_method(self) -> None:
        """Create a fresh scorer instance before each test."""
        self.scorer = ExactMatchScorer()

    # --- Happy-path tests ---

    def test_all_correct_returns_one(self) -> None:
        """100% correct predictions should return accuracy = 1.0."""
        predictions = ["Metformin", "Aspirin", "Paracetamol"]
        references = ["Metformin", "Aspirin", "Paracetamol"]
        assert self.scorer.score(predictions, references) == 1.0

    def test_all_wrong_returns_zero(self) -> None:
        """All incorrect predictions should return accuracy = 0.0."""
        predictions = ["Aspirin", "Ibuprofen"]
        references = ["Metformin", "Paracetamol"]
        assert self.scorer.score(predictions, references) == 0.0

    def test_half_correct_returns_half(self) -> None:
        """50% correct predictions should return accuracy = 0.5."""
        predictions = ["Metformin", "Aspirin"]
        references = ["Metformin", "Paracetamol"]
        assert self.scorer.score(predictions, references) == 0.5

    def test_normalisation_is_applied(self) -> None:
        """Case and punctuation differences should not affect the score."""
        predictions = ["METFORMIN.", "  aspirin  "]
        references = ["metformin", "Aspirin"]
        assert self.scorer.score(predictions, references) == 1.0

    def test_single_sample(self) -> None:
        """Should work correctly with a single-element list."""
        assert self.scorer.score(["A"], ["A"]) == 1.0
        assert self.scorer.score(["A"], ["B"]) == 0.0

    # --- Validation tests ---

    def test_raises_on_empty_lists(self) -> None:
        """ValueError must be raised when both lists are empty."""
        with pytest.raises(ValueError, match="non-empty"):
            self.scorer.score([], [])

    def test_raises_on_length_mismatch(self) -> None:
        """ValueError must be raised when list lengths differ."""
        with pytest.raises(ValueError, match="Lengths"):
            self.scorer.score(["A", "B"], ["A"])

    def test_raises_on_empty_predictions(self) -> None:
        """ValueError must be raised when predictions is empty."""
        with pytest.raises(ValueError, match="non-empty"):
            self.scorer.score([], ["A"])


# ---------------------------------------------------------------------------
# SemanticSimilarityScorer — mocked
# ---------------------------------------------------------------------------


def _make_mock_evaluate(f1_scores: List[float]) -> MagicMock:
    """Build a mock ``evaluate`` module that returns the given F1 scores.

    Args:
        f1_scores: The F1 values the mock metric should return.

    Returns:
        A ``MagicMock`` object mimicking ``evaluate.load("bertscore")``.
    """
    mock_metric = MagicMock()
    mock_metric.compute.return_value = {"f1": f1_scores}

    mock_evaluate = MagicMock()
    mock_evaluate.load.return_value = mock_metric
    return mock_evaluate


class TestSemanticSimilarityScorer:
    """Tests for SemanticSimilarityScorer with mocked evaluate library."""

    def test_returns_mean_f1(self) -> None:
        """Should return the mean of the F1 scores from BERTScore."""
        f1_values = [0.9, 0.8, 0.7]
        mock_evaluate = _make_mock_evaluate(f1_values)

        scorer = SemanticSimilarityScorer(model_type="distilbert-base-uncased")

        with patch.dict("sys.modules", {"evaluate": mock_evaluate}):
            # Force lazy load inside patched context
            scorer._metric = None
            result = scorer.score(
                predictions=["heart attack", "aspirin", "fever"],
                references=["myocardial infarction", "acetylsalicylic acid", "pyrexia"],
            )

        expected = sum(f1_values) / len(f1_values)
        assert abs(result - expected) < 1e-9

    def test_single_sample_returns_scalar_f1(self) -> None:
        """Should return a scalar float even for a single-sample batch."""
        mock_evaluate = _make_mock_evaluate([0.85])
        scorer = SemanticSimilarityScorer()

        with patch.dict("sys.modules", {"evaluate": mock_evaluate}):
            scorer._metric = None
            result = scorer.score(["heart attack"], ["myocardial infarction"])

        assert abs(result - 0.85) < 1e-9

    def test_raises_on_length_mismatch(self) -> None:
        """ValueError must propagate from _validate_inputs before any model call."""
        scorer = SemanticSimilarityScorer()
        with pytest.raises(ValueError, match="Lengths"):
            scorer.score(["A", "B"], ["A"])

    def test_raises_on_empty_inputs(self) -> None:
        """ValueError must propagate from _validate_inputs before any model call."""
        scorer = SemanticSimilarityScorer()
        with pytest.raises(ValueError, match="non-empty"):
            scorer.score([], [])

    def test_raises_import_error_when_evaluate_missing(self) -> None:
        """Should raise ImportError with an installation hint when evaluate is absent."""
        scorer = SemanticSimilarityScorer()
        scorer._metric = None

        with patch.dict("sys.modules", {"evaluate": None}):
            with pytest.raises((ImportError, TypeError)):
                scorer.score(["test"], ["test"])

    def test_metric_is_loaded_only_once(self) -> None:
        """evaluate.load() should be called exactly once across multiple score() calls."""
        mock_evaluate = _make_mock_evaluate([0.9])
        scorer = SemanticSimilarityScorer()

        with patch.dict("sys.modules", {"evaluate": mock_evaluate}):
            scorer._metric = None
            scorer.score(["A"], ["B"])
            scorer.score(["C"], ["D"])

        # load() called once on first score(); second call reuses cached metric.
        mock_evaluate.load.assert_called_once_with("bertscore")
