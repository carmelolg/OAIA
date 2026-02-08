import pytest
from unittest.mock import patch
from lib.commons.EnvironmentVariables import EnvironmentVariables


class TestEnvironmentVariables:
    def test_singleton_instance(self, mocker):
        """Test that EnvironmentVariables is a singleton."""
        mocker.patch('lib.commons.EnvironmentVariables.load_dotenv')
        instance1 = EnvironmentVariables()
        instance2 = EnvironmentVariables()
        assert instance1 is instance2

    @patch('lib.commons.EnvironmentVariables.load_dotenv')
    @patch('os.getenv')
    def test_get_embedding_model(self, mock_getenv, mock_load_dotenv):
        """Test get_embedding_model method."""
        mock_getenv.return_value = "test_model"
        result = EnvironmentVariables.get_embedding_model("default")
        mock_getenv.assert_called_with("EMBEDDING_MODEL", "default")
        assert result == "test_model"

    @patch('lib.commons.EnvironmentVariables.load_dotenv')
    @patch('os.getenv')
    def test_get_language_model(self, mock_getenv, mock_load_dotenv):
        """Test get_language_model method."""
        mock_getenv.return_value = "test_lang_model"
        result = EnvironmentVariables.get_language_model("default")
        mock_getenv.assert_called_with("LANGUAGE_MODEL", "default")
        assert result == "test_lang_model"

    @patch('lib.commons.EnvironmentVariables.load_dotenv')
    @patch('os.getenv')
    def test_get_thinking_mode(self, mock_getenv, mock_load_dotenv):
        """Test get_thinking_mode method."""
        mock_getenv.return_value = "test_mode"
        result = EnvironmentVariables.get_thinking_mode("default")
        mock_getenv.assert_called_with("THINKING_MODE", "default")
        assert result == "test_mode"

    @patch('lib.commons.EnvironmentVariables.load_dotenv')
    @patch('os.getenv')
    def test_get_kindergarten_api_host(self, mock_getenv, mock_load_dotenv):
        """Test get_kindergarten_api_host method."""
        mock_getenv.return_value = "test_host"
        result = EnvironmentVariables.get_kindergarten_api_host("default")
        mock_getenv.assert_called_with("KINDERGARTEN_API_HOST", "default")
        assert result == "test_host"

    @patch('lib.commons.EnvironmentVariables.load_dotenv')
    @patch('os.getenv')
    def test_get_home_kitchen_api_host(self, mock_getenv, mock_load_dotenv):
        """Test get_home_kitchen_api_host method."""
        mock_getenv.return_value = "test_host"
        result = EnvironmentVariables.get_home_kitchen_api_host("default")
        mock_getenv.assert_called_with("HOME_KITCHEN_API_HOST", "default")
        assert result == "test_host"

    @patch('lib.commons.EnvironmentVariables.load_dotenv')
    @patch('os.getenv')
    def test_get_kindergarten_api_path(self, mock_getenv, mock_load_dotenv):
        """Test get_kindergarten_api_path method."""
        mock_getenv.return_value = "test_path"
        result = EnvironmentVariables.get_kindergarten_api_path("default")
        mock_getenv.assert_called_with("KINDERGARTEN_API_PATH", "default")
        assert result == "test_path"

    @patch('lib.commons.EnvironmentVariables.load_dotenv')
    @patch('os.getenv')
    def test_get_home_kitchen_api_path(self, mock_getenv, mock_load_dotenv):
        """Test get_home_kitchen_api_path method."""
        mock_getenv.return_value = "test_path"
        result = EnvironmentVariables.get_home_kitchen_api_path("default")
        mock_getenv.assert_called_with("HOME_KITCHEN_API_PATH", "default")
        assert result == "test_path"

    @patch('lib.commons.EnvironmentVariables.load_dotenv')
    @patch('os.getenv')
    def test_get_llm_provider(self, mock_getenv, mock_load_dotenv):
        """Test get_llm_provider method."""
        mock_getenv.return_value = "test_provider"
        result = EnvironmentVariables.get_llm_provider("default")
        mock_getenv.assert_called_with("LLM_PROVIDER", "default")
        assert result == "test_provider"
