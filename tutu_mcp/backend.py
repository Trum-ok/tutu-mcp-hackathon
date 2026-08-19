"""Shared contract between the live upstream client and the fixture-backed mock client.

`ToolBackend` is the seam the proxy server codes against, so it never knows
whether it's talking to the real `mcp.tutu.ru` or to recorded fixtures.
"""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCallResult:
    text: str
    is_error: bool


class ToolBackend(Protocol):
    async def list_tools(self) -> list[dict[str, Any]]:
        """Raw tool descriptors as returned by upstream `tools/list` (name/description/inputSchema/annotations)."""
        ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        """Call `name` with `arguments`, returning its single text content block."""
        ...
