import pytest
from unittest.mock import patch, MagicMock
from lib.adapters.outbound.LLMExecutor import LLMExecutor


class TestLLMExecutor:
    def test_singleton_instance(self):
        """Test that LLMExecutor is a singleton."""
        instance1 = LLMExecutor.get_instance()
        instance2 = LLMExecutor.get_instance()
        assert instance1 is instance2

    def test_singleton_init_raises_exception_on_second_call(self):
        """Test that initializing LLMExecutor twice raises an exception."""
        LLMExecutor.get_instance()  # First instance
        with pytest.raises(Exception, match="This class is a singleton!"):
            LLMExecutor()

    @patch('lib.adapters.outbound.LLMExecutor.current_provider')
    @patch('lib.adapters.outbound.LLMExecutor.llm', 'test_model')
    @patch('lib.adapters.outbound.LLMExecutor.think', True)
    def test_ask_method(self, mock_provider):
        """Test the ask method."""
        mock_provider.chat.return_value = "response"
        result = LLMExecutor.ask("test prompt", system_prompt="system", chatbot_mode=True, disable_think=False)
        mock_provider.chat.assert_called_once()
        args, kwargs = mock_provider.chat.call_args
        assert kwargs['prompt'] == "test prompt"
        assert kwargs['system_prompt'] == "system"
        assert kwargs['model'] == "test_model"
        assert kwargs['config'].get_think() == True
        assert kwargs['config'].get_stream() == True
        assert result == "response"

    @patch('lib.adapters.outbound.LLMExecutor.current_provider')
    @patch('lib.adapters.outbound.LLMExecutor.llm', 'test_model')
    @patch('lib.adapters.outbound.LLMExecutor.think', False)
    def test_chat_method(self, mock_provider):
        """Test the chat method."""
        mock_provider.chat.return_value = "response"
        tools = {"tool1": "func"}
        result = LLMExecutor.chat("test prompt", chatbot_mode=False, tools=tools, system_prompt="system", disable_think=True)
        mock_provider.chat.assert_called_once()
        args, kwargs = mock_provider.chat.call_args
        assert kwargs['prompt'] == "test prompt"
        assert kwargs['system_prompt'] == "system"
        assert kwargs['model'] == "test_model"
        assert kwargs['tools'] == tools
        assert kwargs['config'].get_think() == False
        assert kwargs['config'].get_stream() == False
        assert result == "response"
