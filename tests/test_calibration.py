"""
tests/test_calibration.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ``medeval.calibration.calculate_ece``.

Tests cover:
  - Perfect calibration (ECE = 0.0)
  - Severe overconfidence (ECE ≈ 0.49)
  - Graceful handling of sparse / empty probability bins
  - All invalid-input validation paths (raises ValueError)
"""

from __future__ import annotations

import pytest

from medeval.calibration import calculate_brier_score, calculate_ece, calculate_mce

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FLOAT_TOLERANCE = 1e-6  # Acceptable numerical tolerance for ECE comparisons.


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestPerfectCalibration:
    """Tests where confidence scores perfectly match empirical accuracy."""

    def test_perfect_calibration_returns_zero(self) -> None:
        """ECE must be 0.0 when every bin has accuracy == confidence.

        We construct samples where the confidence of each prediction exactly
        matches the fraction of correct predictions in its bin. With 10 equal-
        width bins over [0, 1], placing samples at 0.05, 0.15, ..., 0.95 (bin
        midpoints) and alternating correct/incorrect gives a known ECE of 0.0
        only when accuracy ≡ confidence. We instead use a simpler but
        mathematically equivalent construction: all samples in a single bin
        where acc = conf = 0.9.
        """
        # 9 correct + 1 wrong  =>  acc = 0.9
        # all predictions at confidence 0.9  =>  conf = 0.9
        # => |acc - conf| = 0.0 for the single occupied bin  =>  ECE = 0.0
        n_correct = 9
        n_incorrect = 1
        y_true = [1] * n_correct + [0] * n_incorrect
        y_prob = [0.9] * (n_correct + n_incorrect)

        ece = calculate_ece(y_true, y_prob, n_bins=10)

        assert abs(ece - 0.0) < FLOAT_TOLERANCE, (
            f"Expected ECE=0.0 for perfectly calibrated inputs, got {ece}."
        )

    def test_perfect_calibration_multi_bin(self) -> None:
        """ECE must be 0.0 across multiple perfectly calibrated bins.

        Two bins: high-confidence bin (conf ≈ 0.9, acc = 0.9) and
        low-confidence bin (conf ≈ 0.1, acc = 0.1).
        """
        high_conf_correct = 9
        high_conf_incorrect = 1
        low_conf_correct = 1
        low_conf_incorrect = 9

        y_true = (
            [1] * high_conf_correct
            + [0] * high_conf_incorrect
            + [1] * low_conf_correct
            + [0] * low_conf_incorrect
        )
        y_prob = [0.9] * (high_conf_correct + high_conf_incorrect) + [0.1] * (
            low_conf_correct + low_conf_incorrect
        )

        ece = calculate_ece(y_true, y_prob, n_bins=10)

        assert abs(ece - 0.0) < FLOAT_TOLERANCE, (
            f"Expected ECE=0.0 for multi-bin perfectly calibrated inputs, got {ece}."
        )


class TestSevereOverconfidence:
    """Tests for a model that is systematically overconfident."""

    def test_severe_overconfidence_returns_expected_ece(self) -> None:
        """ECE ≈ 0.49 when confidence is 0.99 but accuracy is 0.50.

        All 100 samples fall into the top probability bin (0.9, 1.0].
        - Bin confidence: mean(0.99 * 100) = 0.99
        - Bin accuracy:   50 correct / 100 total = 0.50
        - Bin weight:     100 / 100 = 1.0
        - ECE:            1.0 * |0.50 − 0.99| = 0.49
        """
        n_samples = 100
        # Alternating correct / incorrect => accuracy = 0.50
        y_true = [1, 0] * (n_samples // 2)
        # Uniformly high confidence => confidence ≈ 0.99
        y_prob = [0.99] * n_samples

        expected_ece = 0.49
        ece = calculate_ece(y_true, y_prob, n_bins=10)

        assert abs(ece - expected_ece) < FLOAT_TOLERANCE, (
            f"Expected ECE≈{expected_ece} for overconfident model, got {ece:.6f}."
        )

    def test_complete_overconfidence_all_wrong(self) -> None:
        """ECE = 0.9 when confidence is 0.9 but accuracy is 0.0.

        All samples are wrong (y_true=0) but predicted with confidence 0.9.
        ECE = 1.0 * |0.0 − 0.9| = 0.9.
        """
        n_samples = 50
        y_true = [0] * n_samples
        y_prob = [0.9] * n_samples

        expected_ece = 0.9
        ece = calculate_ece(y_true, y_prob, n_bins=10)

        assert abs(ece - expected_ece) < FLOAT_TOLERANCE, (
            f"Expected ECE={expected_ece} for fully wrong overconfident model, got {ece:.6f}."
        )


class TestEmptyBinsHandled:
    """Tests ensuring the function handles empty probability bins without crashing."""

    def test_sparse_inputs_do_not_crash(self) -> None:
        """Function must not raise when many bins are empty.

        With only 2 samples whose probabilities both fall in the same bin,
        8 of 10 bins will be empty. The function must return a finite float.
        """
        y_true = [1, 0]
        y_prob = [0.85, 0.82]  # Both in the [0.8, 0.9) bin — 8 bins empty.

        result = calculate_ece(y_true, y_prob, n_bins=10)

        assert isinstance(result, float), "ECE must be a float."
        assert 0.0 <= result <= 1.0, f"ECE must be in [0, 1], got {result}."

    def test_single_sample_single_bin(self) -> None:
        """ECE must be computable with a single sample (9 bins empty)."""
        y_true = [1]
        y_prob = [0.75]

        # Single correct sample at confidence 0.75 → ECE = |1.0 − 0.75| = 0.25
        expected_ece = 0.25
        result = calculate_ece(y_true, y_prob, n_bins=10)

        assert abs(result - expected_ece) < FLOAT_TOLERANCE, (
            f"Expected ECE={expected_ece} for single sample, got {result:.6f}."
        )

    def test_all_samples_in_one_bin(self) -> None:
        """All samples in a single bin must not cause errors in the other 9 empty bins."""
        n_samples = 20
        y_true = [1] * 10 + [0] * 10  # accuracy = 0.5
        y_prob = [0.55] * n_samples  # confidence ≈ 0.55, all in [0.5, 0.6) bin

        result = calculate_ece(y_true, y_prob, n_bins=10)

        # ECE = 1.0 * |0.5 − 0.55| = 0.05
        expected_ece = 0.05
        assert abs(result - expected_ece) < FLOAT_TOLERANCE, (
            f"Expected ECE={expected_ece}, got {result:.6f}."
        )


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests that all invalid inputs are rejected with clear ValueError messages."""

    def test_raises_on_empty_inputs(self) -> None:
        """ValueError must be raised when either input is empty."""
        with pytest.raises(ValueError, match="must not be empty"):
            calculate_ece([], [], n_bins=10)

    def test_raises_on_length_mismatch(self) -> None:
        """ValueError must be raised when y_true and y_prob have different lengths."""
        with pytest.raises(ValueError, match="same length"):
            calculate_ece([1, 0, 1], [0.9, 0.1], n_bins=10)

    def test_raises_on_invalid_n_bins(self) -> None:
        """ValueError must be raised for non-positive n_bins values."""
        with pytest.raises(ValueError, match="positive integer"):
            calculate_ece([1, 0], [0.9, 0.1], n_bins=0)

        with pytest.raises(ValueError, match="positive integer"):
            calculate_ece([1, 0], [0.9, 0.1], n_bins=-5)

    def test_raises_on_prob_out_of_range(self) -> None:
        """ValueError must be raised when any probability is outside [0, 1]."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            calculate_ece([1, 0], [1.5, 0.1], n_bins=10)

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            calculate_ece([1, 0], [-0.1, 0.9], n_bins=10)

    def test_raises_on_non_binary_labels(self) -> None:
        """ValueError must be raised when y_true contains values other than 0 or 1."""
        with pytest.raises(ValueError, match="binary"):
            calculate_ece([1, 2, 0], [0.9, 0.5, 0.1], n_bins=10)

    def test_raises_when_n_bins_is_float(self) -> None:
        """ValueError must be raised when n_bins is a float, not an int."""
        with pytest.raises(ValueError, match="positive integer"):
            calculate_ece([1, 0], [0.9, 0.1], n_bins=10.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MCE & Brier Score tests
# ---------------------------------------------------------------------------


class TestMaximumCalibrationError:
    def test_mce_with_perfect_calibration(self) -> None:
        """MCE must be 0.0 when perfectly calibrated."""
        y_true = [1] * 9 + [0] * 1
        y_prob = [0.9] * 10
        mce = calculate_mce(y_true, y_prob, n_bins=10)
        assert abs(mce - 0.0) < FLOAT_TOLERANCE

    def test_mce_with_overconfidence(self) -> None:
        """MCE identifies the worst-case bin gap."""
        # Bin 1 (high confidence): 100 samples, acc 0.5, conf 0.99 -> error = 0.49
        # Bin 2 (low confidence): 100 samples, acc 0.1, conf 0.1 -> error = 0.0
        y_true = [1, 0] * 50 + [1] * 10 + [0] * 90
        y_prob = [0.99] * 100 + [0.1] * 100
        mce = calculate_mce(y_true, y_prob, n_bins=10)
        assert abs(mce - 0.49) < FLOAT_TOLERANCE


class TestBrierScore:
    def test_brier_score_perfect(self) -> None:
        """Brier score is exactly 0.0 if predictions match labels perfectly."""
        y_true = [1, 1, 0, 0]
        y_prob = [1.0, 1.0, 0.0, 0.0]
        brier = calculate_brier_score(y_true, y_prob)
        assert abs(brier - 0.0) < FLOAT_TOLERANCE

    def test_brier_score_worst(self) -> None:
        """Brier score is 1.0 if completely wrong."""
        y_true = [1, 0]
        y_prob = [0.0, 1.0]
        brier = calculate_brier_score(y_true, y_prob)
        assert abs(brier - 1.0) < FLOAT_TOLERANCE

    def test_brier_score_intermediate(self) -> None:
        """Test standard intermediate calculations."""
        y_true = [1, 0]
        y_prob = [0.8, 0.2]
        # ( (0.8 - 1)^2 + (0.2 - 0)^2 ) / 2 = (0.04 + 0.04) / 2 = 0.04
        brier = calculate_brier_score(y_true, y_prob)
        assert abs(brier - 0.04) < FLOAT_TOLERANCE
