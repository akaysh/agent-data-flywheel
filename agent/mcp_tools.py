"""Exposes the in-process FastMCP server as LangChain tools."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import Client
from langchain_core.tools import BaseTool, StructuredTool

from tools.server import build_server


@asynccontextmanager
async def open_tools(app_dir: str | Path | None = None):
    async with Client(build_server(app_dir)) as client:
        yield [_as_langchain_tool(client, t) for t in await client.list_tools()]


def _as_langchain_tool(client: Client, mcp_tool) -> BaseTool:
    async def call(**kwargs):
        result = await client.call_tool(mcp_tool.name, kwargs)
        return result.data if result.data is not None else result.content[0].text

    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description or mcp_tool.name,
        args_schema=mcp_tool.input_schema,
        coroutine=call,
    )
