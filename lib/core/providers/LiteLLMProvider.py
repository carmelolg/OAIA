"""
LiteLLMProvider Module

This module provides the LiteLLMProvider class, which implements the Provider interface
using the LiteLLM Python SDK. LiteLLM supports 100+ LLMs (OpenAI, Anthropic, xAI,
VertexAI, Ollama, etc.) through a unified OpenAI-compatible interface.

All chat responses are normalized to LLMResponse so that callers receive a consistent,
provider-agnostic payload regardless of the underlying backend.

Environment variables required per backend (examples):
    - OpenAI:        OPENAI_API_KEY
    - Anthropic:     ANTHROPIC_API_KEY
    - xAI:           XAI_API_KEY
    - Azure OpenAI:  AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
    - Vertex AI:     VERTEXAI_PROJECT, VERTEXAI_LOCATION
    - NVIDIA NIM:    NVIDIA_NIM_API_KEY, NVIDIA_NIM_API_BASE
    - HuggingFace:   HUGGINGFACE_API_KEY
    - OpenRouter:    OPENROUTER_API_KEY
    - Ollama:        (no key needed; set api_base if not localhost:11434)

Model strings use the LiteLLM format: "<provider>/<model>", e.g.:
    - "openai/gpt-4o"
    - "anthropic/claude-3-sonnet-20240229"
    - "ollama/llama2"
"""

import os
import json
from typing import Iterator, Union, List, Any

import litellm

from lib.core.providers.LLMProvider import Provider
from lib.core.providers.model.LLMProviderConfiguration import ProviderConfiguration
from lib.core.providers.model.LLMResponse import LLMResponse


class LiteLLMProvider(Provider):
    """
    Singleton provider for LiteLLM interactions.

    This class provides methods to perform chats with any LiteLLM-supported model,
    including simple chats, agentic chats with tool calls, and text embeddings.
    It follows the singleton pattern to ensure only one instance exists.

    The model string follows LiteLLM conventions (e.g. "openai/gpt-4o",
    "anthropic/claude-3-sonnet-20240229", "ollama/llama2"). An optional
    LITELLM_API_BASE environment variable can be set to route requests to a
    custom gateway or local server (e.g. Ollama at http://localhost:11434).

    All chat methods return :class:`LLMResponse` (or a generator of :class:`LLMResponse`
    for streaming) so that downstream code is provider-agnostic.

    Attributes:
        __instance: The singleton instance of the class.
    """

    __instance = None

    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance of LiteLLMProvider.

        Returns:
            LiteLLMProvider: The singleton instance.
        """
        if cls.__instance is None:
            cls()
        return cls.__instance

    def __init__(self):
        """
        Initialize the singleton instance of LiteLLMProvider.

        Raises:
            Exception: If an instance already exists (singleton violation).
        """
        if LiteLLMProvider.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            LiteLLMProvider.__instance = self

    def _get_api_base(self) -> Union[str, None]:
        """
        Return the optional API base URL from the environment.

        Returns:
            str or None: The value of LITELLM_API_BASE, or None if not set.
        """
        return os.getenv("LITELLM_API_BASE", None)

    @staticmethod
    def _normalize_response(raw) -> LLMResponse:
        """
        Normalize a raw LiteLLM ModelResponse to LLMResponse.

        Args:
            raw: A non-streaming LiteLLM ModelResponse object.

        Returns:
            LLMResponse: The normalized response.
        """
        msg = raw.choices[0].message
        usage_data = getattr(raw, 'usage', None)
        return LLMResponse(
            content=msg.content or "",
            role=msg.role or "assistant",
            finish_reason=raw.choices[0].finish_reason,
            usage={
                "prompt_tokens": getattr(usage_data, 'prompt_tokens', None),
                "completion_tokens": getattr(usage_data, 'completion_tokens', None),
                "total_tokens": getattr(usage_data, 'total_tokens', None),
            } if usage_data else None,
        )

    @staticmethod
    def _normalize_stream(raw_stream) -> Iterator[LLMResponse]:
        """
        Wrap a streaming LiteLLM response in a generator that yields LLMResponse chunks.

        Args:
            raw_stream: An iterable of LiteLLM streaming chunk objects.

        Yields:
            LLMResponse: One normalized chunk per streaming event.
        """
        for chunk in raw_stream:
            choice = chunk.choices[0]
            delta = choice.delta
            done = choice.finish_reason is not None
            yield LLMResponse(
                content=getattr(delta, 'content', None) or "",
                role=getattr(delta, 'role', None) or "assistant",
                done=done,
                finish_reason=choice.finish_reason,
            )

    def simple_chat(
        self,
        prompt: str,
        model: str,
        system_prompt: str = None,
        config: ProviderConfiguration = None,
    ) -> Union[LLMResponse, Iterator[LLMResponse]]:
        """
        Perform a simple chat without tools using LiteLLM.

        Args:
            prompt (str): The user prompt.
            model (str): The LiteLLM model string (e.g. "openai/gpt-4o").
            system_prompt (str, optional): The system prompt.
            config (ProviderConfiguration, optional): Configuration for the chat.

        Returns:
            Union[LLMResponse, Iterator[LLMResponse]]: Normalized response or streaming generator.

        Raises:
            litellm.AuthenticationError: When API key is missing or invalid.
            litellm.RateLimitError: When the upstream provider rate-limits the request.
            litellm.APIError: For other API-level errors.
        """
        stream = config.get_stream() if config is not None else False

        _messages = []
        if system_prompt is not None:
            _messages.append({"role": "system", "content": system_prompt})
        _messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": model,
            "messages": _messages,
            "stream": stream,
        }
        api_base = self._get_api_base()
        if api_base:
            kwargs["api_base"] = api_base

        print(f"LiteLLMProvider: calling model='{model}' stream={stream}")
        raw = litellm.completion(**kwargs)
        if stream:
            return self._normalize_stream(raw)
        return self._normalize_response(raw)

    def agentic_chat(
        self,
        prompt: str,
        model: str,
        system_prompt: str,
        assistant_prompt: str,
        tools: dict,
        config: ProviderConfiguration = None,
    ) -> Union[LLMResponse, Iterator[LLMResponse]]:
        """
        Perform an agentic chat with tool calls using LiteLLM.

        This method handles conversations where the LLM can call tools, process
        their results, and generate a final response. Tools are expected to be a
        dict mapping function name → callable (same contract as OllamaProvider).
        The tool schemas (OpenAI function-calling format) are passed directly to
        LiteLLM as the ``tools`` parameter.

        Args:
            prompt (str): The user prompt.
            model (str): The LiteLLM model string.
            system_prompt (str): The system prompt.
            assistant_prompt (str): The assistant prompt.
            tools (dict): Dictionary mapping tool name to callable.
            config (ProviderConfiguration, optional): Configuration for the chat.

        Returns:
            Union[LLMResponse, Iterator[LLMResponse]]: Normalized response or streaming generator.

        Raises:
            litellm.AuthenticationError: When API key is missing or invalid.
            litellm.RateLimitError: When the upstream provider rate-limits the request.
            litellm.APIError: For other API-level errors.
        """
        stream = config.get_stream() if config is not None else False

        _messages = []
        if system_prompt is not None:
            _messages.append({"role": "system", "content": system_prompt})
        _messages.append({"role": "user", "content": prompt})
        if assistant_prompt is not None:
            _messages.append({"role": "assistant", "content": assistant_prompt})

        api_base = self._get_api_base()

        base_kwargs = {"model": model}
        if api_base:
            base_kwargs["api_base"] = api_base

        print(f"LiteLLMProvider: agentic call model='{model}' stream={stream}")

        # Initial (non-streaming) call to detect tool invocations
        initial_kwargs = dict(base_kwargs)
        initial_kwargs["messages"] = _messages
        if tools:
            initial_kwargs["tools"] = list(tools.values())
            initial_kwargs["tool_choice"] = "auto"

        response = litellm.completion(**initial_kwargs)
        response_message = response.choices[0].message
        _messages.append(response_message)

        if response_message.tool_calls:
            for tc in response_message.tool_calls:
                name = tc.function.name
                if name in tools:
                    arguments = tc.function.arguments
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    result = tools[name](**arguments)
                    _messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(result),
                        }
                    )
                else:
                    print(f"LiteLLMProvider: no tool available for '{name}'")

        # Final response with streaming control
        final_kwargs = dict(base_kwargs)
        final_kwargs["messages"] = _messages
        final_kwargs["stream"] = stream
        raw_final = litellm.completion(**final_kwargs)
        if stream:
            return self._normalize_stream(raw_final)
        return self._normalize_response(raw_final)

    def embed(self, text: str, embedding_model: str) -> List[float]:
        """
        Generate embeddings for the given text using LiteLLM.

        Args:
            text (str): The text to embed.
            embedding_model (str): The LiteLLM embedding model string
                (e.g. "openai/text-embedding-3-small").

        Returns:
            List[float]: The embedding vector.

        Raises:
            litellm.AuthenticationError: When API key is missing or invalid.
            litellm.APIError: For other API-level errors.
        """
        kwargs = {"model": embedding_model, "input": text}
        api_base = self._get_api_base()
        if api_base:
            kwargs["api_base"] = api_base

        response = litellm.embedding(**kwargs)
        return response.data[0]["embedding"]
