"""Mock `ToolBackend` — serves recorded fixtures instead of calling Tutu live.

Used for reproducible evals and for demoing/developing without burning the
shared rate limit against `mcp.tutu.ru`.
"""

from typing import Any

from tutu_mcp.backend import ToolCallResult
from tutu_mcp.replay.store import FixtureStore


class MockUpstreamClient:
    def __init__(self, store: FixtureStore) -> None:
        self._store = store

    async def list_tools(self) -> list[dict[str, Any]]:
        return self._store.load_tools_list()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        return self._store.find_result(name, arguments)
