"""
medeval/models/base.py
~~~~~~~~~~~~~~~~~~~~~~
Abstract base class definition for LLM model connectors.

All model connectors (local, API, mock) must subclass ``BaseModelConnector``
to ensure a unified interface across the evaluation framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseModelConnector(ABC):
    """Abstract base class that all model connectors must implement.

    Ensures that the orchestrator runner can retrieve text predictions
    and associated probabilities in a unified manner.

    Attributes:
        model_name: The string identifier of the model.
    """

    def __init__(self, model_name: str, max_tokens: int = 2048) -> None:
        """Initialise the connector with a model name.

        Args:
            model_name: The name/identifier of the model.
            max_tokens: Maximum generated tokens per response. Defaults to 2048.
        """
        self._model_name = model_name
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        """The model name / identifier.

        Returns:
            The string identifier of the model.
        """
        return self._model_name

    @property
    def max_tokens(self) -> int:
        """The maximum tokens to generate per response.

        Returns:
            The integer maximum token limit.
        """
        return self._max_tokens

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate response text for the given prompt.

        Args:
            prompt: The formatted query string sent to the model.
            temperature: The sampling temperature. Defaults to 0.0 (greedy).

        Returns:
            The model's textual prediction.
        """

    @abstractmethod
    def generate_probabilities(self, prompt: str) -> list[float]:
        """Generate probability confidence scores associated with the model's prediction.

        For multiple-choice tasks, this should return a list of probabilities corresponding
        to the different choices. For free-text tasks, this can return sequence-level
        token probability confidence scores.

        Args:
            prompt: The formatted query string sent to the model.

        Returns:
            A list of float probabilities in [0.0, 1.0].
        """

    def generate_n(self, prompt: str, n: int, temperature: float = 0.7) -> list[str]:
        """Generate N independent responses for self-consistency sampling.

        Provides a default multi-threaded fallback using the single-generation
        method. Subclasses can override this to leverage native API batching
        (e.g., OpenAI's 'n' parameter) for significantly faster execution.

        Args:
            prompt: The formatted query string sent to the model.
            n: The number of responses to sample.
            temperature: The sampling temperature. Defaults to 0.7 for diversity.

        Returns:
            A list of N generated response strings.
        """
        # Use a sequential loop instead of ThreadPoolExecutor to prevent deadlocks
        # with rate-limited proxy endpoints like AgentRouter.
        results = []
        for _ in range(n):
            results.append(self.generate(prompt, temperature))
        return results
