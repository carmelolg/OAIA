import pytest
from lib.core.providers.model.LLMResponse import LLMResponse


class TestLLMResponse:
    def test_default_values(self):
        """Test that LLMResponse has sensible defaults."""
        response = LLMResponse(content="Hello!")
        assert response.content == "Hello!"
        assert response.role == "assistant"
        assert response.finish_reason is None
        assert response.usage is None
        assert response.thinking is None
        assert response.done is True

    def test_all_fields(self):
        """Test LLMResponse with all fields provided."""
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        response = LLMResponse(
            content="The answer is 42.",
            role="assistant",
            finish_reason="stop",
            usage=usage,
            thinking="Let me think...",
            done=True,
        )
        assert response.content == "The answer is 42."
        assert response.role == "assistant"
        assert response.finish_reason == "stop"
        assert response.usage == usage
        assert response.thinking == "Let me think..."
        assert response.done is True

    def test_streaming_chunk(self):
        """Test a streaming chunk has done=False."""
        chunk = LLMResponse(content="partial", done=False)
        assert chunk.done is False
        assert chunk.finish_reason is None

    def test_to_dict(self):
        """Test to_dict serializes all fields correctly."""
        usage = {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        response = LLMResponse(
            content="Hi",
            role="assistant",
            finish_reason="stop",
            usage=usage,
            thinking=None,
            done=True,
        )
        d = response.to_dict()
        assert d["content"] == "Hi"
        assert d["role"] == "assistant"
        assert d["finish_reason"] == "stop"
        assert d["usage"] == usage
        assert d["thinking"] is None
        assert d["done"] is True

    def test_to_dict_minimal(self):
        """Test to_dict with only required field."""
        d = LLMResponse(content="").to_dict()
        assert d["content"] == ""
        assert d["role"] == "assistant"
        assert d["finish_reason"] is None
        assert d["usage"] is None
        assert d["thinking"] is None
        assert d["done"] is True

    def test_empty_content(self):
        """Test that empty content string is valid."""
        response = LLMResponse(content="")
        assert response.content == ""
