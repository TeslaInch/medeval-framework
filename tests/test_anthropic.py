"""
tests/test_anthropic.py
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ``medeval.models.anthropic_connector``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from medeval.models.anthropic_connector import AnthropicConnector


class TestAnthropicConnector:
    """Tests for AnthropicConnector."""

    def test_init_sets_properties(self) -> None:
        """Constructor must set model_name and default parameters."""
        conn = AnthropicConnector("claude-3-5-sonnet-20241022", api_key="fake-key")
        assert conn.model_name == "claude-3-5-sonnet-20241022"
        assert conn._api_key == "fake-key"

    def test_generate_probabilities_returns_empty_list(self) -> None:
        """generate_probabilities must return an empty list without crashing."""
        conn = AnthropicConnector("claude-3-5-sonnet-20241022")
        probs = conn.generate_probabilities("What is PKU?")
        assert probs == []

    @patch("medeval.models.anthropic_connector.AnthropicConnector._lazy_init")
    def test_generate_extracts_text_content(self, mock_init: MagicMock) -> None:
        """generate must query messages.create and extract block text."""
        conn = AnthropicConnector("claude-3-5-sonnet-20241022")
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Treatment is Metformin."

        mock_response = MagicMock()
        mock_response.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        conn._client = mock_client

        output = conn.generate("What is the treatment?")
        assert output == "Treatment is Metformin."
        mock_client.messages.create.assert_called_once()
