"""
tests/test_hallucination.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ``medeval.hallucination`` — NLIHallucinationDetector and
NLIResult.

The transformers ``pipeline`` is fully mocked so no checkpoint is downloaded.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from medeval.hallucination import (
    _CONTRADICTION_LABEL,
    _ENTAILMENT_LABEL,
    _NEUTRAL_LABEL,
    NLIHallucinationDetector,
    NLIResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline_output(
    entailment: float,
    neutral: float,
    contradiction: float,
) -> list[dict]:
    """Build a fake text-classification pipeline output."""
    return [
        {"label": _ENTAILMENT_LABEL, "score": entailment},
        {"label": _NEUTRAL_LABEL, "score": neutral},
        {"label": _CONTRADICTION_LABEL, "score": contradiction},
    ]


def _make_mock_pipeline(entailment: float, neutral: float, contradiction: float) -> MagicMock:
    """Create a mock transformers pipeline returning the specified scores."""
    output = _make_pipeline_output(entailment, neutral, contradiction)
    mock_pipe = MagicMock(return_value=[output])
    mock_pipe.model.config.id2label = None
    return mock_pipe


# ---------------------------------------------------------------------------
# NLIResult property tests
# ---------------------------------------------------------------------------


class TestNLIResult:
    """Tests for the NLIResult dataclass."""

    def test_non_entailment_score_sums_neutral_and_contradiction(self) -> None:
        """non_entailment_score must equal neutral + contradiction."""
        result = NLIResult(
            is_hallucination=True,
            entailment_score=0.1,
            neutral_score=0.4,
            contradiction_score=0.5,
            threshold=0.5,
        )
        assert abs(result.non_entailment_score - 0.9) < 1e-9

    def test_non_hallucination_when_entailment_dominates(self) -> None:
        """is_hallucination should be False when entailment dominates."""
        result = NLIResult(
            is_hallucination=False,
            entailment_score=0.9,
            neutral_score=0.05,
            contradiction_score=0.05,
            threshold=0.5,
        )
        assert result.is_hallucination is False


# ---------------------------------------------------------------------------
# NLIHallucinationDetector — constructor validation
# ---------------------------------------------------------------------------


class TestNLIHallucinationDetectorInit:
    """Tests for constructor-level validation."""

    def test_raises_on_threshold_zero(self) -> None:
        """ValueError must be raised for threshold = 0."""
        with pytest.raises(ValueError, match="threshold"):
            NLIHallucinationDetector(threshold=0.0)

    def test_raises_on_threshold_one(self) -> None:
        """ValueError must be raised for threshold = 1."""
        with pytest.raises(ValueError, match="threshold"):
            NLIHallucinationDetector(threshold=1.0)

    def test_valid_threshold_accepted(self) -> None:
        """Any threshold in (0, 1) must be accepted without error."""
        for threshold in [0.01, 0.5, 0.99]:
            detector = NLIHallucinationDetector(threshold=threshold)
            assert detector._threshold == threshold


# ---------------------------------------------------------------------------
# NLIHallucinationDetector.detect — mocked pipeline
# ---------------------------------------------------------------------------


class TestNLIHallucinationDetectorDetect:
    """Tests for the detect() method with a mocked transformers pipeline."""

    def _make_detector_with_mock_pipeline(
        self,
        entailment: float,
        neutral: float,
        contradiction: float,
        threshold: float = 0.5,
    ) -> NLIHallucinationDetector:
        """Create a detector whose internal pipeline is pre-seeded with a mock.

        Args:
            entailment: Mock entailment score.
            neutral: Mock neutral score.
            contradiction: Mock contradiction score.
            threshold: Decision threshold for the detector.

        Returns:
            A configured ``NLIHallucinationDetector`` instance.
        """
        detector = NLIHallucinationDetector(threshold=threshold)
        detector._pipeline = _make_mock_pipeline(entailment, neutral, contradiction)
        return detector

    def test_clear_entailment_not_hallucination(self) -> None:
        """High entailment score must not trigger hallucination flag."""
        detector = self._make_detector_with_mock_pipeline(
            entailment=0.95, neutral=0.03, contradiction=0.02
        )
        result = detector.detect(
            premise="Aspirin is an NSAID.",
            hypothesis="Aspirin reduces pain and fever.",
        )
        assert result.is_hallucination is False
        assert result.entailment_score == pytest.approx(0.95)

    def test_clear_contradiction_is_hallucination(self) -> None:
        """High contradiction score must trigger the hallucination flag."""
        detector = self._make_detector_with_mock_pipeline(
            entailment=0.05, neutral=0.05, contradiction=0.90
        )
        result = detector.detect(
            premise="Aspirin is an NSAID used to reduce fever.",
            hypothesis="Aspirin is an ACE inhibitor used to treat hypertension.",
        )
        assert result.is_hallucination is True
        assert result.contradiction_score == pytest.approx(0.90)

    def test_high_neutral_is_hallucination(self) -> None:
        """Combined neutral+contradiction > threshold must flag hallucination."""
        detector = self._make_detector_with_mock_pipeline(
            entailment=0.2, neutral=0.5, contradiction=0.3, threshold=0.5
        )
        result = detector.detect(
            premise="Fever is managed with paracetamol.",
            hypothesis="The patient should fast for 24 hours.",
        )
        # neutral(0.5) + contradiction(0.3) = 0.8 > 0.5 → hallucination
        assert result.is_hallucination is True
        assert result.non_entailment_score == pytest.approx(0.8)

    def test_exact_threshold_boundary_is_not_hallucination(self) -> None:
        """Score exactly equal to threshold must NOT be flagged (strict >)."""
        detector = self._make_detector_with_mock_pipeline(
            entailment=0.5, neutral=0.25, contradiction=0.25, threshold=0.5
        )
        result = detector.detect(premise="Context text.", hypothesis="Claim text.")
        # non_entailment = 0.5, threshold = 0.5 → NOT > threshold → no hallucination
        assert result.is_hallucination is False

    def test_result_contains_raw_output(self) -> None:
        """NLIResult.raw_output must contain the full pipeline response."""
        detector = self._make_detector_with_mock_pipeline(0.8, 0.1, 0.1)
        result = detector.detect(premise="Premise.", hypothesis="Hypothesis.")
        assert "raw_list" in result.raw_output

    def test_raises_on_empty_premise(self) -> None:
        """ValueError must be raised when premise is an empty string."""
        detector = NLIHallucinationDetector()
        with pytest.raises(ValueError, match="premise"):
            detector.detect(premise="", hypothesis="Some claim.")

    def test_raises_on_empty_hypothesis(self) -> None:
        """ValueError must be raised when hypothesis is an empty string."""
        detector = NLIHallucinationDetector()
        with pytest.raises(ValueError, match="hypothesis"):
            detector.detect(premise="Some context.", hypothesis="")

    def test_raises_import_error_when_transformers_missing(self) -> None:
        """Should raise ImportError when transformers is not installed."""
        detector = NLIHallucinationDetector()
        detector._pipeline = None  # ensure lazy load is attempted

        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises((ImportError, TypeError)):
                detector.detect(premise="Context.", hypothesis="Claim.")


# ---------------------------------------------------------------------------
# NLIHallucinationDetector.detect_batch
# ---------------------------------------------------------------------------


class TestNLIHallucinationDetectorBatch:
    """Tests for the detect_batch() method."""

    def test_batch_returns_correct_number_of_results(self) -> None:
        """detect_batch should return one NLIResult per pair."""
        detector = NLIHallucinationDetector(threshold=0.5)
        detector._pipeline = _make_mock_pipeline(0.9, 0.05, 0.05)

        results = detector.detect_batch(
            premises=["Context A.", "Context B.", "Context C."],
            hypotheses=["Claim A.", "Claim B.", "Claim C."],
        )
        assert len(results) == 3
        assert all(isinstance(r, NLIResult) for r in results)

    def test_batch_raises_on_length_mismatch(self) -> None:
        """ValueError must be raised when batch lists differ in length."""
        detector = NLIHallucinationDetector()
        with pytest.raises(ValueError, match="Lengths"):
            detector.detect_batch(
                premises=["A", "B"],
                hypotheses=["A"],
            )

    def test_batch_raises_on_empty_lists(self) -> None:
        """ValueError must be raised when batch lists are empty."""
        detector = NLIHallucinationDetector()
        with pytest.raises(ValueError, match="non-empty"):
            detector.detect_batch(premises=[], hypotheses=[])
