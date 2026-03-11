import pytest
from lib.use_case.prompts.PromptManager import PromptManager


class ConcretePromptManager(PromptManager):
    """Concrete implementation of PromptManager that calls super() to exercise abstract bodies."""

    def get_user_prompt(self, *args, **kwargs) -> str:
        return super().get_user_prompt(*args, **kwargs)

    def get_system_prompt(self, *args, **kwargs) -> str:
        return super().get_system_prompt(*args, **kwargs)


class TestPromptManager:
    def test_abstract_get_user_prompt_returns_none(self):
        """Test that the abstract get_user_prompt body (pass) returns None."""
        manager = ConcretePromptManager()
        result = manager.get_user_prompt()
        assert result is None

    def test_abstract_get_system_prompt_returns_none(self):
        """Test that the abstract get_system_prompt body (pass) returns None."""
        manager = ConcretePromptManager()
        result = manager.get_system_prompt()
        assert result is None
