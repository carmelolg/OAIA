"""
OllamaProvider Module

This module provides the OllamaProvider class, which implements the Provider interface
for interacting with Ollama LLM models. It supports simple chats, agentic chats with tools,
and text embedding. All chat responses are normalized to LLMResponse so that callers
receive a consistent, provider-agnostic payload regardless of the underlying backend.
"""

from typing import Iterator, Union, List

import ollama as OllamaClient

from lib.commons.EnvironmentVariables import EnvironmentVariables
from lib.core.providers.LLMProvider import Provider
from lib.core.providers.model.LLMProviderConfiguration import ProviderConfiguration
from lib.core.providers.model.LLMResponse import LLMResponse

env = EnvironmentVariables()


class OllamaProvider(Provider):
    """
    Singleton provider for Ollama LLM interactions.

    This class provides methods to perform chats with Ollama models, including simple chats,
    agentic chats with tool calls, and text embeddings. It follows the singleton pattern
    to ensure only one instance exists.

    All chat methods return :class:`LLMResponse` (or a generator of :class:`LLMResponse`
    for streaming) so that downstream code is provider-agnostic.

    Attributes:
        __instance: The singleton instance of the class.
    """

    __instance = None

    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance of OllamaProvider.

        Returns:
            OllamaProvider: The singleton instance.
        """
        if cls.__instance is None:
            cls()
        return cls.__instance

    def __init__(self):
        """
        Initialize the singleton instance of OllamaProvider.

        Raises:
            Exception: If an instance already exists (singleton violation).
        """
        if OllamaProvider.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            OllamaProvider.__instance = self

    @staticmethod
    def _normalize_response(raw) -> LLMResponse:
        """
        Normalize a raw Ollama ChatResponse to LLMResponse.

        Args:
            raw: A non-streaming Ollama ChatResponse object.

        Returns:
            LLMResponse: The normalized response.
        """
        prompt_tokens = getattr(raw, 'prompt_eval_count', None)
        completion_tokens = getattr(raw, 'eval_count', None)
        return LLMResponse(
            content=raw.message.content or "",
            role=raw.message.role or "assistant",
            finish_reason=getattr(raw, 'done_reason', None),
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
            },
            thinking=getattr(raw.message, 'thinking', None),
            done=getattr(raw, 'done', True),
        )

    @staticmethod
    def _normalize_stream(raw_stream) -> Iterator[LLMResponse]:
        """
        Wrap a streaming Ollama response in a generator that yields LLMResponse chunks.

        Args:
            raw_stream: An iterable of Ollama ChatResponse streaming chunks.

        Yields:
            LLMResponse: One normalized chunk per Ollama streaming event.
        """
        for chunk in raw_stream:
            chunk_done = getattr(chunk, 'done', False)
            yield LLMResponse(
                content=chunk.message.content or "",
                role=chunk.message.role or "assistant",
                done=chunk_done,
                finish_reason=getattr(chunk, 'done_reason', None) if chunk_done else None,
            )

    def agentic_chat(self, prompt: str, model: str, system_prompt: str, assistant_prompt: str, tools: dict,
                     config: ProviderConfiguration = None) -> Union[LLMResponse, Iterator[LLMResponse]]:
        """
        Perform an agentic chat with tool calls.

        This method handles conversations where the LLM can call tools, process their results,
        and generate a final response.

        Args:
            prompt (str): The user prompt.
            model (str): The Ollama model to use.
            system_prompt (str): The system prompt.
            assistant_prompt (str): The assistant prompt.
            tools (dict): Dictionary of available tools.
            config (ProviderConfiguration, optional): Configuration for the chat.

        Returns:
            Union[LLMResponse, Iterator[LLMResponse]]: Normalized response or streaming generator.
        """
        think = True if config is not None and config.get_think() is not None and config.get_think() else False
        stream = True if config is not None and config.get_stream() is not None and config.get_stream() else False

        _messages = []

        if system_prompt is not None:
            _messages.append({'role': 'system', 'content': system_prompt})

        _messages.append({"role": "user", "content": prompt})

        if assistant_prompt is not None:
            _messages.append({'role': 'assistant', 'content': assistant_prompt})

        response = OllamaClient.chat(model=model, messages=_messages, tools=tools.values(),
                                     think=think)
        _messages.append(response.message)

        if response.message.tool_calls:
            for tc in response.message.tool_calls:
                if tc.function.name in tools:
                    result = tools[tc.function.name](**tc.function.arguments)
                    # add the tool result to the messages
                    _messages.append({'role': 'tool', 'tool_name': tc.function.name, 'content': str(result)})
                else:
                    print(f"No tool available for {tc.function.name}")

        # generate the final response
        raw_final = OllamaClient.chat(model=model, messages=_messages, stream=stream,
                                      think=False)
        if stream:
            return self._normalize_stream(raw_final)
        return self._normalize_response(raw_final)

    def simple_chat(self, prompt: str, model: str, system_prompt: str = None, config: ProviderConfiguration = None) -> \
            Union[LLMResponse, Iterator[LLMResponse]]:
        """
        Perform a simple chat without tools.

        Args:
            prompt (str): The user prompt.
            model (str): The Ollama model to use.
            system_prompt (str, optional): The system prompt.
            config (ProviderConfiguration, optional): Configuration for the chat.

        Returns:
            Union[LLMResponse, Iterator[LLMResponse]]: Normalized response or streaming generator.
        """
        _messages = []
        if system_prompt is not None:
            _messages.append({'role': 'system', 'content': system_prompt})
        _messages.append({'role': 'user', 'content': prompt})

        stream = config.get_stream() if config is not None else False
        raw = OllamaClient.chat(
            model=model,
            messages=_messages,
            stream=stream,
            think=config.get_think() if config is not None else False,
        )
        if stream:
            return self._normalize_stream(raw)
        return self._normalize_response(raw)

    def embed(self, text: str, embedding_model: str = env.get_embedding_model()) -> List[float]:
        """
        Generate embeddings for the given text.

        Args:
            text (str): The text to embed.
            embedding_model (str, optional): The embedding model to use. Defaults to the configured model.

        Returns:
            List[float]: The embedding vector.
        """
        return OllamaClient.embed(model=embedding_model, input=text)['embeddings'][0]
