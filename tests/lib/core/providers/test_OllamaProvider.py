import pytest
from unittest.mock import patch, MagicMock
from lib.core.providers.OllamaProvider import OllamaProvider
from lib.core.providers.model.LLMProviderConfiguration import ProviderConfiguration


class TestOllamaProvider:
    def test_singleton_instance(self):
        """Test that OllamaProvider is a singleton."""
        instance1 = OllamaProvider.get_instance()
        instance2 = OllamaProvider.get_instance()
        assert instance1 is instance2

    def test_singleton_init_raises_exception_on_second_call(self):
        """Test that initializing OllamaProvider twice raises an exception."""
        OllamaProvider.get_instance()  # First instance
        with pytest.raises(Exception, match="This class is a singleton!"):
            OllamaProvider()

    @patch('lib.core.providers.OllamaProvider.OllamaClient.chat')
    def test_simple_chat(self, mock_chat):
        """Test simple_chat method."""
        mock_response = MagicMock()
        mock_chat.return_value = mock_response
        config = ProviderConfiguration(think=True, stream=False)
        provider = OllamaProvider.get_instance()
        result = provider.simple_chat("prompt", "model", "system", config)
        mock_chat.assert_called_once_with(
            model="model",
            messages=[{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'prompt'}],
            stream=False,
            think=True
        )
        assert result == mock_response

    @patch('lib.core.providers.OllamaProvider.OllamaClient.chat')
    def test_agentic_chat_no_tools(self, mock_chat):
        """Test agentic_chat method without tools."""
        mock_response = MagicMock()
        mock_response.message.tool_calls = None
        mock_chat.return_value = mock_response
        config = ProviderConfiguration(think=True, stream=True)
        provider = OllamaProvider.get_instance()
        result = provider.agentic_chat("prompt", "model", "system", "assistant", {}, config)
        assert mock_chat.call_count == 2  # Initial and final
        assert result == mock_response

    @patch('lib.core.providers.OllamaProvider.OllamaClient.chat')
    def test_agentic_chat_with_tools(self, mock_chat):
        """Test agentic_chat method with tools."""
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "tool1"
        mock_tool_call.function.arguments = {"arg": "val"}
        mock_response.message.tool_calls = [mock_tool_call]
        mock_final_response = MagicMock()
        mock_chat.side_effect = [mock_response, mock_final_response]
        config = ProviderConfiguration(think=False, stream=False)
        tools = {"tool1": MagicMock(return_value="result")}
        provider = OllamaProvider.get_instance()
        result = provider.agentic_chat("prompt", "model", "system", "assistant", tools, config)
        assert mock_chat.call_count == 2
        assert result == mock_final_response

    @patch('lib.core.providers.OllamaProvider.OllamaClient.embed')
    def test_embed(self, mock_embed):
        """Test embed method."""
        mock_embed.return_value = {'embeddings': [['vec']]}
        provider = OllamaProvider.get_instance()
        result = provider.embed("text", "embed_model")
        mock_embed.assert_called_once_with(model="embed_model", input="text")
        assert result == ['vec']
