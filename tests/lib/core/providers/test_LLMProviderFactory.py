import pytest
from unittest.mock import patch, MagicMock
from lib.core.providers.LLMProviderFactory import LLMProviderFactory


class TestLLMProviderFactory:
    @patch('lib.core.providers.LLMProviderFactory.ollama_provider')
    @patch('lib.core.providers.LLMProviderFactory.LLM_PROVIDER', 'ollama')
    @patch('lib.core.providers.LLMProviderFactory.const')
    def test_get_instance_ollama(self, mock_const, mock_provider):
        """Test get_instance returns OllamaProvider when provider is ollama."""
        mock_const.llm_provider_ollama = 'ollama'
        result = LLMProviderFactory.get_instance()
        assert result == mock_provider

    @patch('lib.core.providers.LLMProviderFactory.LLM_PROVIDER', 'unknown')
    def test_get_instance_unknown(self):
        """Test get_instance returns None for unknown provider."""
        result = LLMProviderFactory.get_instance()
        assert result is None
