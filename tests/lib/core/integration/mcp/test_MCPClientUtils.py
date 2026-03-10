import asyncio
import pytest
from lib.core.integration.mcp.MCPClientUtils import MCPClientUtils


class TestMCPClientUtils:
    def setup_method(self):
        """Reset singleton before each test."""
        MCPClientUtils._instance = None

    def test_singleton_get_instance(self):
        """Test that get_instance returns the same instance on repeated calls."""
        instance1 = MCPClientUtils.get_instance()
        instance2 = MCPClientUtils.get_instance()
        assert instance1 is instance2

    def test_get_instance_creates_instance(self):
        """Test that get_instance creates an instance of MCPClientUtils."""
        instance = MCPClientUtils.get_instance()
        assert isinstance(instance, MCPClientUtils)

    def test_get_mcp_tools_returns_none(self):
        """Test that get_mcp_tools returns None (pass body)."""
        instance = MCPClientUtils.get_instance()
        result = asyncio.run(instance.get_mcp_tools())
        assert result is None

    def test_execute_mcp_tool_returns_none(self):
        """Test that execute_mcp_tool returns None (pass body)."""
        instance = MCPClientUtils.get_instance()
        result = asyncio.run(instance.execute_mcp_tool("tool_name", {"key": "value"}))
        assert result is None
