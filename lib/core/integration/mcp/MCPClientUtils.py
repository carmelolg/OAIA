from fastmcp import Client as MCPClient

class MCPClientUtils:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance



    async def get_mcp_tools(self):
        pass

    async def execute_mcp_tool(self, mcp_tool: str, args: dict):
        pass
