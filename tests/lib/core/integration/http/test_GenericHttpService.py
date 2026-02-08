import pytest
import requests
import json
from unittest.mock import patch, mock_open, MagicMock
from lib.core.integration.http.GenericHttpService import GenericHttpService


class TestGenericHttpService:
    @patch('lib.core.integration.http.GenericHttpService.requests.get')
    def test_get_success(self, mock_get):
        """Test get method with successful API response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "value"}
        mock_get.return_value = mock_response
        result = GenericHttpService.get("http://api.com", "path", "fallback.json")
        mock_get.assert_called_once_with("http://api.com/path")
        assert result == {"data": "value"}

    @patch('lib.core.integration.http.GenericHttpService.requests.get')
    @patch('builtins.open', new_callable=mock_open, read_data='{"fallback": "data"}')
    @patch('lib.core.integration.http.GenericHttpService.json.load')
    def test_get_api_failure_fallback_success(self, mock_json_load, mock_file, mock_get):
        """Test get method with API failure and successful fallback."""
        mock_get.side_effect = requests.exceptions.RequestException("API error")
        mock_json_load.return_value = {"fallback": "data"}
        result = GenericHttpService.get("http://api.com", "path", "fallback.json")
        mock_file.assert_called_once_with("fallback.json")
        assert result == {"fallback": "data"}

    @patch('lib.core.integration.http.GenericHttpService.requests.get')
    @patch('builtins.open')
    def test_get_api_failure_fallback_failure(self, mock_file, mock_get):
        """Test get method with API and fallback failure."""
        mock_get.side_effect = requests.exceptions.RequestException("API error")
        mock_file.side_effect = FileNotFoundError()
        result = GenericHttpService.get("http://api.com", "path", "fallback.json")
        assert result is None

    @patch('builtins.open', new_callable=mock_open, read_data='invalid json')
    @patch('lib.core.integration.http.GenericHttpService.json.load')
    def test_get_no_api_fallback_json_error(self, mock_json_load, mock_file):
        """Test get method with no API and JSON decode error in fallback."""
        mock_json_load.side_effect = json.JSONDecodeError("error", "doc", 0)
        result = GenericHttpService.get("", "", "fallback.json")
        assert result is None
