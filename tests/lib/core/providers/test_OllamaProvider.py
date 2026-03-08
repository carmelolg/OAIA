import types
import pytest
from unittest.mock import patch, MagicMock
from lib.core.providers.OllamaProvider import OllamaProvider
from lib.core.providers.model.LLMProviderConfiguration import ProviderConfiguration
from lib.core.providers.model.LLMResponse import LLMResponse


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
        """Test simple_chat returns a normalized LLMResponse."""
        mock_raw = MagicMock()
        mock_raw.message.content = "Hello!"
        mock_raw.message.role = "assistant"
        mock_raw.message.thinking = None
        mock_raw.done_reason = "stop"
        mock_raw.prompt_eval_count = 10
        mock_raw.eval_count = 20
        mock_raw.done = True
        mock_chat.return_value = mock_raw

        config = ProviderConfiguration(think=True, stream=False)
        provider = OllamaProvider.get_instance()
        result = provider.simple_chat("prompt", "model", "system", config)

        mock_chat.assert_called_once_with(
            model="model",
            messages=[{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'prompt'}],
            stream=False,
            think=True
        )
        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.role == "assistant"
        assert result.finish_reason == "stop"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 20
        assert result.usage["total_tokens"] == 30
        assert result.thinking is None
        assert result.done is True

    @patch('lib.core.providers.OllamaProvider.OllamaClient.chat')
    def test_simple_chat_streaming(self, mock_chat):
        """Test simple_chat with stream=True returns a generator of LLMResponse chunks."""
        chunk1 = MagicMock()
        chunk1.message.content = "Hel"
        chunk1.message.role = "assistant"
        chunk1.done = False
        chunk1.done_reason = None

        chunk2 = MagicMock()
        chunk2.message.content = "lo!"
        chunk2.message.role = "assistant"
        chunk2.done = True
        chunk2.done_reason = "stop"

        mock_chat.return_value = iter([chunk1, chunk2])

        config = ProviderConfiguration(think=False, stream=True)
        provider = OllamaProvider.get_instance()
        result = provider.simple_chat("prompt", "model", config=config)

        assert isinstance(result, types.GeneratorType)
        chunks = list(result)
        assert len(chunks) == 2
        assert chunks[0].content == "Hel"
        assert chunks[0].done is False
        assert chunks[1].content == "lo!"
        assert chunks[1].done is True
        assert chunks[1].finish_reason == "stop"

    @patch('lib.core.providers.OllamaProvider.OllamaClient.chat')
    def test_agentic_chat_no_tools(self, mock_chat):
        """Test agentic_chat without tools returns a streaming generator when stream=True."""
        mock_first = MagicMock()
        mock_first.message.tool_calls = None

        mock_final_raw = MagicMock()
        mock_final_raw.__iter__ = MagicMock(return_value=iter([]))
        mock_chat.side_effect = [mock_first, mock_final_raw]

        config = ProviderConfiguration(think=True, stream=True)
        provider = OllamaProvider.get_instance()
        result = provider.agentic_chat("prompt", "model", "system", "assistant", {}, config)

        assert mock_chat.call_count == 2
        assert isinstance(result, types.GeneratorType)

    @patch('lib.core.providers.OllamaProvider.OllamaClient.chat')
    def test_agentic_chat_with_tools(self, mock_chat):
        """Test agentic_chat with tools returns a normalized LLMResponse."""
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "tool1"
        mock_tool_call.function.arguments = {"arg": "val"}
        mock_response.message.tool_calls = [mock_tool_call]

        mock_final_raw = MagicMock()
        mock_final_raw.message.content = "Tool result processed"
        mock_final_raw.message.role = "assistant"
        mock_final_raw.message.thinking = None
        mock_final_raw.done_reason = "stop"
        mock_final_raw.prompt_eval_count = 5
        mock_final_raw.eval_count = 15
        mock_final_raw.done = True

        mock_chat.side_effect = [mock_response, mock_final_raw]
        config = ProviderConfiguration(think=False, stream=False)
        tools = {"tool1": MagicMock(return_value="result")}
        provider = OllamaProvider.get_instance()
        result = provider.agentic_chat("prompt", "model", "system", "assistant", tools, config)

        assert mock_chat.call_count == 2
        assert isinstance(result, LLMResponse)
        assert result.content == "Tool result processed"
        assert result.finish_reason == "stop"

    @patch('lib.core.providers.OllamaProvider.OllamaClient.embed')
    def test_embed(self, mock_embed):
        """Test embed method."""
        mock_embed.return_value = {'embeddings': [['vec']]}
        provider = OllamaProvider.get_instance()
        result = provider.embed("text", "embed_model")
        mock_embed.assert_called_once_with(model="embed_model", input="text")
        assert result == ['vec']
