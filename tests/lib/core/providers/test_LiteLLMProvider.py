import json
import types
import pytest
from unittest.mock import patch, MagicMock

from lib.core.providers.LiteLLMProvider import LiteLLMProvider
from lib.core.providers.model.LLMProviderConfiguration import ProviderConfiguration
from lib.core.providers.model.LLMResponse import LLMResponse


class TestLiteLLMProvider:
    def test_singleton_instance(self):
        """Test that LiteLLMProvider is a singleton."""
        instance1 = LiteLLMProvider.get_instance()
        instance2 = LiteLLMProvider.get_instance()
        assert instance1 is instance2

    def test_singleton_init_raises_exception_on_second_call(self):
        """Test that initializing LiteLLMProvider twice raises an exception."""
        LiteLLMProvider.get_instance()
        with pytest.raises(Exception, match="This class is a singleton!"):
            LiteLLMProvider()

    # ------------------------------------------------------------------
    # simple_chat – non-streaming
    # ------------------------------------------------------------------

    @patch('lib.core.providers.LiteLLMProvider.litellm.completion')
    def test_simple_chat_non_streaming(self, mock_completion):
        """Test simple_chat in non-streaming mode returns a normalized LLMResponse."""
        mock_raw = MagicMock()
        mock_raw.choices[0].message.content = "Hi there!"
        mock_raw.choices[0].message.role = "assistant"
        mock_raw.choices[0].finish_reason = "stop"
        mock_raw.usage.prompt_tokens = 5
        mock_raw.usage.completion_tokens = 10
        mock_raw.usage.total_tokens = 15
        mock_completion.return_value = mock_raw

        config = ProviderConfiguration(stream=False, think=False)
        provider = LiteLLMProvider.get_instance()

        result = provider.simple_chat(
            prompt="Hello",
            model="openai/gpt-4o",
            system_prompt="You are helpful.",
            config=config,
        )

        mock_completion.assert_called_once_with(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            stream=False,
        )
        assert isinstance(result, LLMResponse)
        assert result.content == "Hi there!"
        assert result.role == "assistant"
        assert result.finish_reason == "stop"
        assert result.usage["prompt_tokens"] == 5
        assert result.usage["completion_tokens"] == 10
        assert result.done is True

    @patch('lib.core.providers.LiteLLMProvider.litellm.completion')
    def test_simple_chat_streaming(self, mock_completion):
        """Test simple_chat in streaming mode returns a generator of LLMResponse chunks."""
        chunk1 = MagicMock()
        chunk1.choices[0].delta.content = "Hel"
        chunk1.choices[0].delta.role = "assistant"
        chunk1.choices[0].finish_reason = None

        chunk2 = MagicMock()
        chunk2.choices[0].delta.content = "lo!"
        chunk2.choices[0].delta.role = "assistant"
        chunk2.choices[0].finish_reason = "stop"

        mock_completion.return_value = iter([chunk1, chunk2])

        config = ProviderConfiguration(stream=True, think=False)
        provider = LiteLLMProvider.get_instance()

        result = provider.simple_chat(
            prompt="Hello",
            model="anthropic/claude-3-sonnet-20240229",
            config=config,
        )

        mock_completion.assert_called_once_with(
            model="anthropic/claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        assert isinstance(result, types.GeneratorType)
        chunks = list(result)
        assert len(chunks) == 2
        assert chunks[0].content == "Hel"
        assert chunks[0].done is False
        assert chunks[1].content == "lo!"
        assert chunks[1].done is True
        assert chunks[1].finish_reason == "stop"

    @patch('lib.core.providers.LiteLLMProvider.litellm.completion')
    def test_simple_chat_no_system_prompt(self, mock_completion):
        """Test simple_chat omits system message when system_prompt is None."""
        mock_raw = MagicMock()
        mock_raw.choices[0].message.content = "OK"
        mock_raw.choices[0].message.role = "assistant"
        mock_raw.choices[0].finish_reason = "stop"
        mock_completion.return_value = mock_raw

        config = ProviderConfiguration(stream=False, think=False)
        provider = LiteLLMProvider.get_instance()
        provider.simple_chat(prompt="Hi", model="openai/gpt-4o", config=config)

        call_kwargs = mock_completion.call_args[1]
        roles = [m["role"] for m in call_kwargs["messages"]]
        assert "system" not in roles

    @patch.dict('os.environ', {"LITELLM_API_BASE": "http://localhost:11434"})
    @patch('lib.core.providers.LiteLLMProvider.litellm.completion')
    def test_simple_chat_with_api_base(self, mock_completion):
        """Test that LITELLM_API_BASE is forwarded to litellm.completion."""
        mock_raw = MagicMock()
        mock_raw.choices[0].message.content = "OK"
        mock_raw.choices[0].message.role = "assistant"
        mock_raw.choices[0].finish_reason = "stop"
        mock_completion.return_value = mock_raw

        config = ProviderConfiguration(stream=False, think=False)
        provider = LiteLLMProvider.get_instance()
        provider.simple_chat(prompt="Hello", model="ollama/llama2", config=config)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["api_base"] == "http://localhost:11434"

    # ------------------------------------------------------------------
    # agentic_chat
    # ------------------------------------------------------------------

    @patch('lib.core.providers.LiteLLMProvider.litellm.completion')
    def test_agentic_chat_no_tool_calls(self, mock_completion):
        """Test agentic_chat when the model makes no tool calls returns LLMResponse."""
        mock_first = MagicMock()
        mock_first.choices[0].message.tool_calls = None

        mock_final_raw = MagicMock()
        mock_final_raw.choices[0].message.content = "Final answer"
        mock_final_raw.choices[0].message.role = "assistant"
        mock_final_raw.choices[0].finish_reason = "stop"
        mock_completion.side_effect = [mock_first, mock_final_raw]

        config = ProviderConfiguration(stream=False, think=False)
        provider = LiteLLMProvider.get_instance()
        result = provider.agentic_chat(
            prompt="Hello",
            model="openai/gpt-4o",
            system_prompt="You are helpful.",
            assistant_prompt=None,
            tools={},
            config=config,
        )

        assert mock_completion.call_count == 2
        assert isinstance(result, LLMResponse)
        assert result.content == "Final answer"
        assert result.finish_reason == "stop"

    @patch('lib.core.providers.LiteLLMProvider.litellm.completion')
    def test_agentic_chat_with_tool_calls(self, mock_completion):
        """Test agentic_chat correctly executes tool calls and returns normalized LLMResponse."""
        mock_tc = MagicMock()
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = json.dumps({"city": "Rome"})
        mock_tc.id = "call_abc"

        first_response = MagicMock()
        first_response.choices[0].message.tool_calls = [mock_tc]

        final_raw = MagicMock()
        final_raw.choices[0].message.content = "It's sunny in Rome!"
        final_raw.choices[0].message.role = "assistant"
        final_raw.choices[0].finish_reason = "stop"
        mock_completion.side_effect = [first_response, final_raw]

        tool_fn = MagicMock(return_value="Sunny, 25°C")
        config = ProviderConfiguration(stream=False, think=False)
        provider = LiteLLMProvider.get_instance()
        result = provider.agentic_chat(
            prompt="What's the weather in Rome?",
            model="openai/gpt-4o",
            system_prompt=None,
            assistant_prompt=None,
            tools={"get_weather": tool_fn},
            config=config,
        )

        tool_fn.assert_called_once_with(city="Rome")
        assert mock_completion.call_count == 2
        assert isinstance(result, LLMResponse)
        assert result.content == "It's sunny in Rome!"

        # Second completion call messages should include the tool result
        second_call_messages = mock_completion.call_args_list[1][1]["messages"]
        tool_message = next(
            m for m in second_call_messages if m.get("role") == "tool"
        )
        assert tool_message["content"] == "Sunny, 25°C"
        assert tool_message["tool_call_id"] == "call_abc"

    @patch('lib.core.providers.LiteLLMProvider.litellm.completion')
    def test_agentic_chat_unknown_tool(self, mock_completion, capsys):
        """Test agentic_chat prints a warning for unknown tool names."""
        mock_tc = MagicMock()
        mock_tc.function.name = "unknown_tool"
        mock_tc.function.arguments = "{}"
        mock_tc.id = "call_xyz"

        first_response = MagicMock()
        first_response.choices[0].message.tool_calls = [mock_tc]

        final_raw = MagicMock()
        final_raw.choices[0].message.content = "Done"
        final_raw.choices[0].message.role = "assistant"
        final_raw.choices[0].finish_reason = "stop"
        mock_completion.side_effect = [first_response, final_raw]

        config = ProviderConfiguration(stream=False, think=False)
        provider = LiteLLMProvider.get_instance()
        provider.agentic_chat(
            prompt="Do something",
            model="openai/gpt-4o",
            system_prompt=None,
            assistant_prompt=None,
            tools={},
            config=config,
        )

        captured = capsys.readouterr()
        assert "unknown_tool" in captured.out

    @patch('lib.core.providers.LiteLLMProvider.litellm.completion')
    def test_agentic_chat_streaming(self, mock_completion):
        """Test agentic_chat with stream=True returns a generator and passes stream=True to LiteLLM."""
        mock_first = MagicMock()
        mock_first.choices[0].message.tool_calls = None

        mock_completion.side_effect = [mock_first, iter([])]

        config = ProviderConfiguration(stream=True, think=False)
        provider = LiteLLMProvider.get_instance()
        result = provider.agentic_chat(
            prompt="Hello",
            model="openai/gpt-4o",
            system_prompt=None,
            assistant_prompt=None,
            tools={},
            config=config,
        )

        # The final (second) call should have stream=True
        final_call_kwargs = mock_completion.call_args_list[1][1]
        assert final_call_kwargs["stream"] is True
        assert isinstance(result, types.GeneratorType)

    # ------------------------------------------------------------------
    # embed
    # ------------------------------------------------------------------

    @patch('lib.core.providers.LiteLLMProvider.litellm.embedding')
    def test_embed(self, mock_embedding):
        """Test embed returns the first embedding vector from LiteLLM."""
        mock_embedding.return_value = MagicMock(
            data=[{"embedding": [0.1, 0.2, 0.3]}]
        )
        provider = LiteLLMProvider.get_instance()
        result = provider.embed(
            text="Hello world",
            embedding_model="openai/text-embedding-3-small",
        )

        mock_embedding.assert_called_once_with(
            model="openai/text-embedding-3-small",
            input="Hello world",
        )
        assert result == [0.1, 0.2, 0.3]

    # ------------------------------------------------------------------
    # error handling
    # ------------------------------------------------------------------

    @patch('lib.core.providers.LiteLLMProvider.litellm.completion')
    def test_simple_chat_authentication_error(self, mock_completion):
        """Test that AuthenticationError from LiteLLM propagates to the caller."""
        mock_completion.side_effect = litellm_auth_error()
        config = ProviderConfiguration(stream=False, think=False)
        provider = LiteLLMProvider.get_instance()

        import litellm as _litellm
        with pytest.raises(_litellm.AuthenticationError):
            provider.simple_chat(prompt="Hello", model="openai/gpt-4o", config=config)

    @patch('lib.core.providers.LiteLLMProvider.litellm.completion')
    def test_simple_chat_rate_limit_error(self, mock_completion):
        """Test that RateLimitError from LiteLLM propagates to the caller."""
        mock_completion.side_effect = litellm_rate_limit_error()
        config = ProviderConfiguration(stream=False, think=False)
        provider = LiteLLMProvider.get_instance()

        import litellm as _litellm
        with pytest.raises(_litellm.RateLimitError):
            provider.simple_chat(prompt="Hello", model="openai/gpt-4o", config=config)

    @patch('lib.core.providers.LiteLLMProvider.litellm.completion')
    def test_agentic_chat_with_assistant_prompt(self, mock_completion):
        """Test agentic_chat includes assistant_prompt in messages."""
        mock_first = MagicMock()
        mock_first.choices[0].message.tool_calls = None

        mock_final_raw = MagicMock()
        mock_final_raw.choices[0].message.content = "Done"
        mock_final_raw.choices[0].message.role = "assistant"
        mock_final_raw.choices[0].finish_reason = "stop"
        mock_completion.side_effect = [mock_first, mock_final_raw]

        config = ProviderConfiguration(stream=False, think=False)
        provider = LiteLLMProvider.get_instance()
        provider.agentic_chat(
            prompt="Hello",
            model="openai/gpt-4o",
            system_prompt=None,
            assistant_prompt="I am here to help.",
            tools={},
            config=config,
        )

        first_call_messages = mock_completion.call_args_list[0][1]["messages"]
        assistant_msgs = [m for m in first_call_messages if m.get("role") == "assistant"]
        assert any(m["content"] == "I am here to help." for m in assistant_msgs)

    @patch.dict('os.environ', {"LITELLM_API_BASE": "http://localhost:11434"})
    @patch('lib.core.providers.LiteLLMProvider.litellm.completion')
    def test_agentic_chat_with_api_base(self, mock_completion):
        """Test that LITELLM_API_BASE is forwarded in agentic_chat."""
        mock_first = MagicMock()
        mock_first.choices[0].message.tool_calls = None

        mock_final_raw = MagicMock()
        mock_final_raw.choices[0].message.content = "Done"
        mock_final_raw.choices[0].message.role = "assistant"
        mock_final_raw.choices[0].finish_reason = "stop"
        mock_completion.side_effect = [mock_first, mock_final_raw]

        config = ProviderConfiguration(stream=False, think=False)
        provider = LiteLLMProvider.get_instance()
        provider.agentic_chat(
            prompt="Hello",
            model="ollama/llama2",
            system_prompt=None,
            assistant_prompt=None,
            tools={},
            config=config,
        )

        first_call_kwargs = mock_completion.call_args_list[0][1]
        assert first_call_kwargs["api_base"] == "http://localhost:11434"

    @patch.dict('os.environ', {"LITELLM_API_BASE": "http://localhost:11434"})
    @patch('lib.core.providers.LiteLLMProvider.litellm.embedding')
    def test_embed_with_api_base(self, mock_embedding):
        """Test that LITELLM_API_BASE is forwarded to litellm.embedding."""
        mock_embedding.return_value = MagicMock(
            data=[{"embedding": [0.4, 0.5, 0.6]}]
        )
        provider = LiteLLMProvider.get_instance()
        result = provider.embed(
            text="Hello world",
            embedding_model="ollama/nomic-embed-text",
        )

        call_kwargs = mock_embedding.call_args[1]
        assert call_kwargs["api_base"] == "http://localhost:11434"
        assert result == [0.4, 0.5, 0.6]


# ---------------------------------------------------------------------------
# Helpers to construct LiteLLM error instances without making real API calls
# ---------------------------------------------------------------------------

def litellm_auth_error():
    import litellm as _litellm
    return _litellm.AuthenticationError(
        message="Invalid API key",
        llm_provider="openai",
        model="gpt-4o",
    )


def litellm_rate_limit_error():
    import litellm as _litellm
    return _litellm.RateLimitError(
        message="Rate limit exceeded",
        llm_provider="openai",
        model="gpt-4o",
    )
