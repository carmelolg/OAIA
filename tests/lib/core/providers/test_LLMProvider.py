import pytest
from unittest.mock import MagicMock
from lib.core.providers.LLMProvider import Provider
from lib.core.providers.model.LLMProviderConfiguration import ProviderConfiguration


class ConcreteProvider(Provider):
    """Concrete implementation of Provider for testing."""

    def agentic_chat(self, prompt, model, system_prompt, assistant_prompt, tools, config=None):
        return super().agentic_chat(prompt, model, system_prompt, assistant_prompt, tools, config)

    def simple_chat(self, prompt, model, system_prompt=None, config=None):
        return super().simple_chat(prompt, model, system_prompt, config)

    def embed(self, text, embedding_model):
        return super().embed(text, embedding_model)


class TestLLMProvider:
    def test_chat_routes_to_simple_chat_when_no_tools(self):
        """Test that chat() routes to simple_chat when tools is None."""
        provider = ConcreteProvider()
        provider.simple_chat = MagicMock(return_value="simple_result")
        result = provider.chat(prompt="hello", model="model", system_prompt="sys")
        provider.simple_chat.assert_called_once_with(
            prompt="hello", model="model", system_prompt="sys", config=None
        )
        assert result == "simple_result"

    def test_chat_routes_to_agentic_chat_when_tools_provided(self):
        """Test that chat() routes to agentic_chat when tools are provided."""
        provider = ConcreteProvider()
        provider.agentic_chat = MagicMock(return_value="agentic_result")
        tools = {"tool": MagicMock()}
        result = provider.chat(
            prompt="hello",
            model="model",
            system_prompt="sys",
            assistant_prompt="asst",
            tools=tools,
        )
        provider.agentic_chat.assert_called_once_with(
            prompt="hello",
            model="model",
            system_prompt="sys",
            assistant_prompt="asst",
            tools=tools,
            config=None,
        )
        assert result == "agentic_result"

    def test_abstract_agentic_chat_returns_none(self):
        """Test that the abstract agentic_chat body (pass) returns None."""
        provider = ConcreteProvider()
        result = provider.agentic_chat("p", "m", "s", "a", {})
        assert result is None

    def test_abstract_simple_chat_returns_none(self):
        """Test that the abstract simple_chat body (pass) returns None."""
        provider = ConcreteProvider()
        result = provider.simple_chat("p", "m")
        assert result is None

    def test_abstract_embed_returns_none(self):
        """Test that the abstract embed body (pass) returns None."""
        provider = ConcreteProvider()
        result = provider.embed("text", "embed_model")
        assert result is None
