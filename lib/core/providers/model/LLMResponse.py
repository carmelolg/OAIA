"""
LLMResponse Module

This module defines the LLMResponse class, a provider-agnostic response model returned
by all OAIA LLM providers. Using a single normalized type ensures that downstream code
works identically regardless of which provider (Ollama, LiteLLM, etc.) is configured.
"""

from typing import Optional, Dict, Any


class LLMResponse:
    """
    Provider-agnostic response returned by all OAIA LLM providers.

    This class normalizes the varying response shapes of different LLM backends
    (e.g. Ollama's ChatResponse, LiteLLM's ModelResponse) into a single consistent
    structure so that callers never need to know which provider produced the response.

    For non-streaming calls, ``done`` is always ``True``.
    For streaming calls, each yielded chunk has ``done=False``; the final chunk
    has ``done=True`` and may include a ``finish_reason``.

    Attributes:
        content (str): The assistant's text content.
        role (str): The message role, always ``"assistant"``.
        finish_reason (Optional[str]): The reason the model stopped generating
            (e.g. ``"stop"``, ``"length"``). ``None`` for intermediate streaming chunks.
        usage (Optional[Dict]): Token-usage metadata with keys
            ``"prompt_tokens"``, ``"completion_tokens"``, and ``"total_tokens"``.
            May be ``None`` if the backend does not report usage.
        thinking (Optional[str]): Thinking/reasoning content produced by models
            that support extended thinking (e.g. Ollama with ``think=True``).
            ``None`` when not applicable.
        done (bool): ``True`` when the response is complete; ``False`` for
            intermediate streaming chunks.
    """

    content: str
    role: str
    finish_reason: Optional[str]
    usage: Optional[Dict[str, Any]]
    thinking: Optional[str]
    done: bool

    def __init__(
        self,
        content: str,
        role: str = "assistant",
        finish_reason: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        thinking: Optional[str] = None,
        done: bool = True,
    ) -> None:
        """
        Initialize an LLMResponse.

        Args:
            content (str): The assistant's text content.
            role (str, optional): The message role. Defaults to ``"assistant"``.
            finish_reason (Optional[str], optional): Why the model stopped. Defaults to ``None``.
            usage (Optional[Dict], optional): Token usage metadata. Defaults to ``None``.
            thinking (Optional[str], optional): Thinking content. Defaults to ``None``.
            done (bool, optional): Whether the response is complete. Defaults to ``True``.
        """
        self.content = content
        self.role = role
        self.finish_reason = finish_reason
        self.usage = usage
        self.thinking = thinking
        self.done = done

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the LLMResponse into a plain dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of this response.
        """
        return {
            "content": self.content,
            "role": self.role,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "thinking": self.thinking,
            "done": self.done,
        }
