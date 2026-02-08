from lib.use_case.prompts.PromptManager import PromptManager

class FilePromptManager(PromptManager):
    """
    Concrete implementation of PromptManager that loads prompts from files.

    This class loads system and user prompt templates from specified file paths
    and provides methods to format them with given arguments.
    """

    def __init__(self, system_prompt_path: str, user_prompt_path: str):
        """
        Initialize the FilePromptManager with file paths for prompts.

        Args:
            system_prompt_path (str): Path to the file containing the system prompt template.
            user_prompt_path (str): Path to the file containing the user prompt template.
        """
        self.system_prompt_template = self._load_prompt(system_prompt_path)
        self.user_prompt_template = self._load_prompt(user_prompt_path)

    def _load_prompt(self, file_path: str) -> str:
        """
        Load the content of a prompt file.

        Args:
            file_path (str): The path to the prompt file.

        Returns:
            str: The content of the file, or an empty string if the path is invalid.
        """
        if file_path is None or len(file_path) == 0:
            return ""
        with open(file_path, 'r') as f:
            return f.read()

    def get_system_prompt(self, *args, **kwargs) -> str:
        """
        Get the formatted system prompt.

        Args:
            *args: Positional arguments for formatting.
            **kwargs: Keyword arguments for formatting.

        Returns:
            str: The formatted system prompt.
        """
        return self.system_prompt_template.format(*args, **kwargs)

    def get_user_prompt(self, *args, **kwargs) -> str:
        """
        Get the formatted user prompt.

        Args:
            *args: Positional arguments for formatting.
            **kwargs: Keyword arguments for formatting.

        Returns:
            str: The formatted user prompt.
        """
        return self.user_prompt_template.format(*args, **kwargs)
