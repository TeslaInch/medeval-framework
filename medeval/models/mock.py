"""
medeval/models/mock.py
~~~~~~~~~~~~~~~~~~~~~~
Mock model connector for testing.

Provides a predictable, offline model connector that returns predefined
predictions and probabilities. Extremely useful for testing runner pipelines
and reports without making network requests or loading heavy model files.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .base import BaseModelConnector

logger = logging.getLogger(__name__)


class MockConnector(BaseModelConnector):
    """Offline mock connector returning pre-seeded predictions and probabilities.

    Supports two modes of operation:
        1. **Mapping Mode**: Matches prompts against a dictionary to return a specific
           prediction and probability list.
        2. **Sequential Mode**: Iterates sequentially through lists of pre-seeded
           predictions and probabilities.

    Args:
        model_name: Identifier for the mock model. Defaults to 'mock-model'.
        predictions: Sequential list of response strings to yield.
        probabilities: Sequential list of probability float lists to yield.
        mapping: A dictionary mapping prompt substrings to tuple of (response_string, probability_list).
            Mapping mode takes priority over sequential mode.
    """

    def __init__(
        self,
        model_name: str = "mock-model",
        predictions: Optional[List[str]] = None,
        probabilities: Optional[List[List[float]]] = None,
        mapping: Optional[Dict[str, Tuple[str, List[float]]]] = None,
    ) -> None:
        """Initialise mock model outputs."""
        super().__init__(model_name=model_name)
        self._predictions = predictions or []
        self._probabilities = probabilities or []
        self._mapping = mapping or {}
        self._index = 0

    def generate(self, prompt: str) -> str:
        """Mock response text generation.

        Matches prompt keywords or falls back to sequential iteration.

        Args:
            prompt: Text prompt to feed to the mock.

        Returns:
            Predefined mock answer.
        """
        # Mapping mode check
        for trigger_key, (pred, _) in self._mapping.items():
            if trigger_key in prompt:
                logger.debug("MockConnector mapping match found for: '%s'", trigger_key)
                return pred

        # Sequential mode check
        if self._index < len(self._predictions):
            pred = self._predictions[self._index]
            # In sequential mode, generate() does not increment index;
            # generate_probabilities() does, so call sequence is assumed to be
            # generate() -> generate_probabilities() for a single sample.
            return pred

        return "Default Mock Response"

    def generate_probabilities(self, prompt: str) -> List[float]:
        """Mock response probability generation.

        Args:
            prompt: Text prompt to feed to the mock.

        Returns:
            Predefined list of confidence probabilities.
        """
        # Mapping mode check
        for trigger_key, (_, probs) in self._mapping.items():
            if trigger_key in prompt:
                return probs

        # Sequential mode check
        if self._index < len(self._probabilities):
            probs = self._probabilities[self._index]
            self._index += 1
            return probs

        self._index += 1
        return [1.0]
