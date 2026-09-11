"""
medeval/cot_uncertainty.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
O(1) single-pass uncertainty scorer for reasoning language models.

When a reasoning model (like GLM 5.3 or DeepSeek V4 Flash) does not return
native token log-probabilities, multi-sample self-consistency can be extremely
expensive. This module bypasses resampling by using a zero-shot classifier
to evaluate the model's internal Chain-of-Thought (<think> block) for
expressed uncertainty or hedging.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CoTUncertaintyScorer:
    """Evaluates uncertainty directly from a model's reasoning trace.

    Uses `bge-m3-zeroshot-v2.0` (8192 context length) to classify whether
    the internal monologue expresses confidence or unresolved doubt.
    """

    def __init__(
        self,
        model_name: str = "MoritzLaurer/bge-m3-zeroshot-v2.0",
        device: int = -1,
    ) -> None:
        """Initialise the scorer with model and device settings.

        Args:
            model_name: Hugging Face zero-shot classification model identifier.
            device: Integer device index. ``-1`` = CPU.
        """
        self._model_name = model_name
        self._device = device
        self._pipeline: Any = None
        self._labels = ["confident and certain", "uncertain, guessing, or hedging"]

    def _load_pipeline(self) -> None:
        """Lazily load the zero-shot classification pipeline."""
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for CoTUncertaintyScorer. "
                "Install it with: pip install transformers"
            ) from exc

        if self._pipeline is None:
            logger.info(
                "Loading Zero-Shot CoT classification pipeline: model='%s', device=%d.",
                self._model_name,
                self._device,
            )
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self._model_name,
                device=self._device,
            )

    def score(self, thought_trace: str) -> float:
        """Evaluate the reasoning trace and return a confidence score (0.0 to 1.0).

        Args:
            thought_trace: The raw text of the model's internal reasoning block.

        Returns:
            A float probability [0.0, 1.0] representing the model's confidence.
        """
        if not thought_trace.strip():
            # If there's no trace, we can't reliably guess confidence this way.
            return 1.0

        self._load_pipeline()

        try:
            # We pass truncation=True just in case it wildly exceeds 8192 tokens.
            # BGE-M3 natively handles 8192, so standard traces easily fit.
            res = self._pipeline(
                thought_trace,
                candidate_labels=self._labels,
                truncation=True,
            )

            # Extract the probability assigned to "confident and certain"
            idx = res["labels"].index("confident and certain")
            confidence = res["scores"][idx]
            return float(confidence)

        except Exception as exc:
            logger.warning("Failed to score CoT uncertainty: %s", exc)
            return 1.0
