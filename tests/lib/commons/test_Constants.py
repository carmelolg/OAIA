import pytest
from lib.commons.Constants import Constants


class TestConstants:
    def test_singleton_instance(self):
        """Test that Constants is a singleton."""
        instance1 = Constants.get_instance()
        instance2 = Constants.get_instance()
        assert instance1 is instance2

    def test_singleton_init_raises_exception_on_second_call(self):
        """Test that initializing Constants twice raises an exception."""
        Constants.get_instance()  # First instance
        with pytest.raises(Exception, match="This class is a singleton!"):
            Constants()

    def test_llm_provider_ollama_constant(self):
        """Test that llm_provider_ollama constant is accessible."""
        instance = Constants.get_instance()
        assert instance.llm_provider_ollama == "ollama"
