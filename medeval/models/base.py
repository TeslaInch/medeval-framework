"""
medeval/models/base.py
~~~~~~~~~~~~~~~~~~~~~~
Abstract base class definition for LLM model connectors.

All model connectors (local, API, mock) must subclass ``BaseModelConnector``
to ensure a unified interface across the evaluation framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseModelConnector(ABC):
    """Abstract base class that all model connectors must implement.

    Ensures that the orchestrator runner can retrieve text predictions
    and associated probabilities in a unified manner.

    Attributes:
        model_name: The string identifier of the model.
    """

    def __init__(self, model_name: str) -> None:
        """Initialise the connector with a model name.

        Args:
            model_name: The name/identifier of the model.
        """
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        """The model name / identifier.

        Returns:
            The string identifier of the model.
        """
        return self._model_name

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response text for the given prompt.

        Args:
            prompt: The formatted query string sent to the model.

        Returns:
            The model's textual prediction.
        """

    @abstractmethod
    def generate_probabilities(self, prompt: str) -> List[float]:
        """Generate probability confidence scores associated with the model's prediction.

        For multiple-choice tasks, this should return a list of probabilities corresponding
        to the different choices. For free-text tasks, this can return sequence-level
        token probability confidence scores.

        Args:
            prompt: The formatted query string sent to the model.

        Returns:
            A list of float probabilities in [0.0, 1.0].
        """
