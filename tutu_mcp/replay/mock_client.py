"""Mock `ToolBackend` — serves recorded fixtures instead of calling Tutu live.

Used for reproducible evals and for demoing/developing without burning the
shared rate limit against `mcp.tutu.ru`.
"""

from typing import Any

from tutu_mcp.backend import BackendError, BackendUnknownToolError, ToolCallResult
from tutu_mcp.replay.store import FixtureStore


class UnknownToolError(BackendUnknownToolError):
    """Named tool is absent from the recorded catalog.

    Without this the store answered a misspelled or invented tool name with
    "no fixture recorded for it — run `tutu.py record`", sending the caller off
    to record a tool that does not exist. The catalog is right there on disk, so
    the mock can tell "you asked for the wrong thing" from "we never recorded
    this call" and say which one it is.
    """

    def __init__(self, tool: str, available: list[str]) -> None:
        self.tool = tool
        self.available = available
        super().__init__(
            f"No tool named {tool!r}. This server serves {len(available)}: {', '.join(available)}."
        )


class MockUpstreamClient:
    def __init__(self, store: FixtureStore) -> None:
        self._store = store

    async def list_tools(self) -> list[dict[str, Any]]:
        return self._store.load_tools_list()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        self._reject_unknown_tool(name)
        return self._store.find_result(name, arguments)

    def _reject_unknown_tool(self, name: str) -> None:
        """A fresh clone has no catalog recorded yet, and that is a normal state
        rather than a wrong tool name — stay quiet and let `find_result` report
        the missing recording, which is the gap that actually needs closing."""
        try:
            catalog = self._store.load_tools_list()
        except BackendError:
            return

        names = [tool["name"] for tool in catalog if isinstance(tool, dict) and "name" in tool]
        if names and name not in names:
            raise UnknownToolError(name, sorted(names))
