"""Minimal stdio client for our Security MCP Server."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from multi_agent_code_reviewer.paths import PROJECT_ROOT, VULNERABLE_DEMO

DEMO_PROJECT = VULNERABLE_DEMO
SERVER_MODULE = "multi_agent_code_reviewer.mcp_servers.security"


async def main(project_path: Path) -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", SERVER_MODULE],
        cwd=PROJECT_ROOT,
    )
    async with (
        stdio_client(server) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
            await session.initialize()

            tools_response = await session.list_tools()
            print("=== list_tools ===")
            for tool in tools_response.tools:
                print(f"\nname: {tool.name}")
                print(f"description: {tool.description}")
                print("inputSchema:")
                print(json.dumps(tool.inputSchema, indent=2, ensure_ascii=False))

            arguments = {"project_path": str(project_path)}
            print("\n=== scan_security request ===")
            print(json.dumps(arguments, indent=2, ensure_ascii=False))

            scan_response = await session.call_tool("scan_security", arguments)
            print("\n=== scan_security structured result ===")
            print(
                json.dumps(
                    scan_response.structuredContent,
                    indent=2,
                    ensure_ascii=False,
                )
            )
            if scan_response.isError:
                print("\n=== scan_security MCP error response ===")
                print(
                    json.dumps(
                        scan_response.model_dump(mode="json", exclude_none=False),
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                raise RuntimeError("scan_security returned an MCP tool error")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call scan_security on the local Security MCP Server."
    )
    parser.add_argument(
        "project_path",
        nargs="?",
        type=Path,
        default=DEMO_PROJECT,
        help="Local project directory. Defaults to sample_projects/vulnerable_demo.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    project = arguments.project_path.expanduser().resolve()
    if not project.is_dir():
        raise SystemExit(f"Project directory does not exist: {project}")
    asyncio.run(main(project))
