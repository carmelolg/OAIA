import pytest
from lib.core.providers.model.LLMProviderConfiguration import ProviderConfiguration, ProviderConfigurationBuilder


class TestProviderConfiguration:
    def test_init(self):
        """Test initialization of ProviderConfiguration."""
        config = ProviderConfiguration(stream=True, think=False)
        assert config.get_stream() == True
        assert config.get_think() == False

    def test_stream_setter(self):
        """Test stream setter method."""
        config = ProviderConfiguration(False, False)
        result = config.stream(True)
        assert result is config
        assert config.get_stream() == True

    def test_think_setter(self):
        """Test think setter method."""
        config = ProviderConfiguration(False, False)
        result = config.think(True)
        assert result is config
        assert config.get_think() == True

    def test_get_stream(self):
        """Test get_stream method."""
        config = ProviderConfiguration(True, False)
        assert config.get_stream() == True

    def test_get_think(self):
        """Test get_think method."""
        config = ProviderConfiguration(False, True)
        assert config.get_think() == True

    def test_build(self):
        """Test build method."""
        config = ProviderConfiguration(False, False)
        result = config.build()
        assert result is config

    def test_builder_function(self):
        """Test ProviderConfigurationBuilder function."""
        config = ProviderConfigurationBuilder()
        assert isinstance(config, ProviderConfiguration)
        assert config.get_stream() == False
        assert config.get_think() == False
