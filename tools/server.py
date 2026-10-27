"""FastMCP wrapper around RideCoTools.

In-process:  Client(build_server(app_dir))
Standalone:  python -m tools.server   (stdio; reads RIDECO_APP_DIR)
"""
from pathlib import Path

from fastmcp import FastMCP

from tools.mock_tools import RideCoTools

TOOL_NAMES = (
    "list_recent_rides",
    "get_ride",
    "get_charges",
    "issue_refund",
    "escalate_to_human",
    "create_lost_item_ticket",
    "contact_driver",
    "get_account",
    "update_account",
)


def build_server(app_dir: str | Path | None = None) -> FastMCP:
    tools = RideCoTools(app_dir)
    server = FastMCP("rideco-support")
    for name in TOOL_NAMES:
        server.tool(getattr(tools, name), name=name)
    return server


if __name__ == "__main__":
    build_server().run()
