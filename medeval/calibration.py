"""
medeval/calibration.py
~~~~~~~~~~~~~~~~~~~~~~
Calibration metrics for medical LLM evaluation.

This module implements Expected Calibration Error (ECE), a standard metric for
measuring how well a model's predicted confidence scores match its empirical
accuracy. A perfectly calibrated model achieves ECE = 0.0.

References:
    Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015).
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning."
    Proceedings of the AAAI Conference on Artificial Intelligence.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def calculate_ece(
    y_true: List[int],
    y_prob: List[float],
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE) using equal-width probability bins.

    ECE measures the weighted average gap between a model's confidence
    (predicted probability) and its actual accuracy across a set of samples.
    Formally:

        ECE = Σ_{b=1}^{B} (|B_b| / N) * |acc(B_b) − conf(B_b)|

    where:
      - B is the number of bins,
      - |B_b| is the number of samples in bin b,
      - N is the total number of samples,
      - acc(B_b) is the fraction of correct predictions in bin b,
      - conf(B_b) is the mean predicted probability in bin b.

    Bins with zero samples are skipped gracefully (no division by zero).
    All operations are vectorized via NumPy for efficiency.

    Args:
        y_true: Binary ground-truth labels. Each value must be either 0
            (incorrect / negative class) or 1 (correct / positive class).
            Length must equal ``len(y_prob)``.
        y_prob: Predicted probability for the positive class. Each value
            must be in the closed interval [0, 1].
            Length must equal ``len(y_true)``.
        n_bins: Number of equal-width bins to partition the [0, 1] probability
            range. Must be a positive integer. Defaults to 10.

    Returns:
        The Expected Calibration Error as a non-negative float in [0, 1].
        A value of 0.0 indicates perfect calibration.

    Raises:
        ValueError: If ``y_true`` and ``y_prob`` have different lengths.
        ValueError: If ``y_true`` or ``y_prob`` is empty.
        ValueError: If ``n_bins`` is not a positive integer.
        ValueError: If any value in ``y_prob`` is outside [0, 1].
        ValueError: If any value in ``y_true`` is not 0 or 1.

    Example:
        >>> # Perfect calibration: confidence matches accuracy exactly.
        >>> y_true = [1, 1, 0, 0]
        >>> y_prob = [0.9, 0.8, 0.2, 0.1]
        >>> ece = calculate_ece(y_true, y_prob, n_bins=10)
        >>> round(ece, 4)
        0.0

        >>> # Overconfident model: always predicts 0.99 but is right only 50% of the time.
        >>> y_true = [1, 0] * 50
        >>> y_prob = [0.99] * 100
        >>> round(calculate_ece(y_true, y_prob, n_bins=10), 2)
        0.49
    """
    # --- Input validation ---
    if len(y_true) == 0 or len(y_prob) == 0:
        raise ValueError(
            "y_true and y_prob must not be empty. "
            f"Received lengths: y_true={len(y_true)}, y_prob={len(y_prob)}."
        )

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"y_true and y_prob must have the same length. "
            f"Received: len(y_true)={len(y_true)}, len(y_prob)={len(y_prob)}."
        )

    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(
            f"n_bins must be a positive integer. Received: {n_bins!r}."
        )

    labels = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(y_prob, dtype=np.float64)

    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError(
            "All values in y_true must be binary (0 or 1). "
            f"Found unique values: {np.unique(labels).tolist()}."
        )

    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError(
            "All values in y_prob must be in the range [0, 1]. "
            f"Received min={probs.min():.4f}, max={probs.max():.4f}."
        )

    # --- Core ECE computation ---
    n_samples: int = len(labels)
    ece: float = 0.0

    # Define bin edges over [0, 1]; using endpoint=True ensures prob=1.0 falls
    # in the last bin when we use a right-half-open convention with np.digitize.
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # np.digitize assigns each probability to a bin index in [1, n_bins+1].
    # Clipping to [1, n_bins] prevents an off-by-one on prob = 1.0.
    bin_indices = np.clip(np.digitize(probs, bin_edges) - 1, 0, n_bins - 1)

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        bin_size = int(mask.sum())

        if bin_size == 0:
            # Skip empty bins — no contribution to ECE, no division by zero.
            logger.debug("Bin %d is empty; skipping.", bin_idx)
            continue

        bin_accuracy: float = float(labels[mask].mean())
        bin_confidence: float = float(probs[mask].mean())
        bin_weight: float = bin_size / n_samples

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)
