from abc import ABC, abstractmethod


class PromptManager(ABC):
    """
    Abstract base class for managing prompts.
    Defines the interface for getting user and system prompts.
    """
    @abstractmethod
    def get_user_prompt(self, *args, **kwargs) -> str:
        """
        Get the user prompt based on provided arguments.
        :param args: Positional arguments for prompt formatting.
        :param kwargs: Keyword arguments for prompt formatting.
        :return: The formatted user prompt string.
        """
        pass

    @abstractmethod
    def get_system_prompt(self, *args, **kwargs) -> str:
        """
        Get the system prompt based on provided arguments.
        :param args: Positional arguments for prompt formatting.
        :param kwargs: Keyword arguments for prompt formatting.
        :return: The formatted system prompt string.
        """
        pass