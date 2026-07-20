"""
medeval/models/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Public API surface for model connectors in the medeval framework.

Exposes the base class, local Hugging Face, API-driven OpenAI, and unit-testing
Mock connectors.
"""

from .base import BaseModelConnector
from .huggingface import HuggingFaceConnector
from .mock import MockConnector
from .openai_connector import OpenAIConnector

__all__ = [
    "BaseModelConnector",
    "HuggingFaceConnector",
    "OpenAIConnector",
    "MockConnector",
]
