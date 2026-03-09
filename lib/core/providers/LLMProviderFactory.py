"""
LLMProviderFactory Module

This module provides a factory class for creating instances of LLM providers based on
environment configuration. It supports different providers like Ollama and LiteLLM and returns
the appropriate singleton instance.
"""

from lib.commons.Constants import Constants
from lib.commons.EnvironmentVariables import EnvironmentVariables
from lib.core.providers.OllamaProvider import OllamaProvider
from lib.core.providers.LiteLLMProvider import LiteLLMProvider

env = EnvironmentVariables()
LLM_PROVIDER = env.get_llm_provider('ollama')

const = Constants.get_instance()
ollama_provider = OllamaProvider.get_instance()
litellm_provider = LiteLLMProvider.get_instance()

class LLMProviderFactory:
    """
    Factory class to get the appropriate LLM provider instance based on configuration.

    This class uses the LLM_PROVIDER environment variable to determine which provider
    to instantiate. Supports Ollama and LiteLLM providers.
    """
    @classmethod
    def get_instance(cls):
        """
        Get the LLM provider instance based on the configured provider.

        Returns the singleton instance of the appropriate LLM provider. If the provider
        is not recognized, returns None.

        Returns:
            OllamaProvider, LiteLLMProvider, or None: The LLM provider instance or None if unsupported.
        """
        if LLM_PROVIDER == const.llm_provider_ollama:
            return ollama_provider
        elif LLM_PROVIDER == const.llm_provider_litellm:
            return litellm_provider
        else:
            return None