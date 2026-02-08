import pytest
from unittest.mock import patch, mock_open
from lib.use_case.prompts.FilePromptManager import FilePromptManager


class TestFilePromptManager:
    @patch('builtins.open', new_callable=mock_open)
    def test_init(self, mock_file):
        """Test initialization of FilePromptManager."""
        mock_file.side_effect = [mock_open(read_data="system template").return_value, mock_open(read_data="user template").return_value]
        manager = FilePromptManager("system.txt", "user.txt")
        assert manager.system_prompt_template == "system template"
        assert manager.user_prompt_template == "user template"

    @patch('builtins.open', new_callable=mock_open, read_data="template")
    def test_load_prompt_valid(self, mock_file):
        """Test _load_prompt with valid file."""
        manager = FilePromptManager.__new__(FilePromptManager)  # Avoid init
        result = manager._load_prompt("path.txt")
        mock_file.assert_called_once_with("path.txt", 'r')
        assert result == "template"

    def test_load_prompt_none_path(self):
        """Test _load_prompt with None path."""
        manager = FilePromptManager.__new__(FilePromptManager)
        result = manager._load_prompt(None)
        assert result == ""

    def test_load_prompt_empty_path(self):
        """Test _load_prompt with empty path."""
        manager = FilePromptManager.__new__(FilePromptManager)
        result = manager._load_prompt("")
        assert result == ""

    def test_get_system_prompt(self):
        """Test get_system_prompt method."""
        manager = FilePromptManager.__new__(FilePromptManager)
        manager.system_prompt_template = "Hello {name}"
        result = manager.get_system_prompt(name="World")
        assert result == "Hello World"

    def test_get_user_prompt(self):
        """Test get_user_prompt method."""
        manager = FilePromptManager.__new__(FilePromptManager)
        manager.user_prompt_template = "User {action}"
        result = manager.get_user_prompt(action="login")
        assert result == "User login"
