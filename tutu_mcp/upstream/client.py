"""Live `ToolBackend` — talks to the real mcp.tutu.ru over Streamable HTTP."""

from typing import Any

from mcp.client import Client as MCPClient

from tutu_mcp.backend import ToolCallResult


class UpstreamClient:
    """Thin wrapper around the official MCP client, scoped to one upstream server.

    One instance opens one long-lived connection (`connect`/`aclose`) so repeated
    calls reuse the same session instead of re-running the handshake every time.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._client: MCPClient | None = None

    async def connect(self) -> None:
        self._client = MCPClient(self._url)
        await self._client.__aenter__()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None

    async def __aenter__(self) -> "UpstreamClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _require_client(self) -> MCPClient:
        if self._client is None:
            raise RuntimeError(
                "UpstreamClient is not connected — use it as an async context manager"
            )
        return self._client

    async def list_tools(self) -> list[dict[str, Any]]:
        client = self._require_client()
        result = await client.list_tools()
        return [t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in result.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        client = self._require_client()
        result = await client.call_tool(name, arguments)
        text = "\n".join(block.text for block in result.content if block.type == "text")
        return ToolCallResult(text=text, is_error=result.is_error)

    def server_info(self) -> dict[str, Any]:
        """Upstream's `initialize` result: the always-on instructions plus identity.

        The instructions block is loaded by every client on every session, so it
        counts toward the tool surface the eval measures — not just `tools/list`.
        """
        client = self._require_client()
        result = client.session.initialize_result
        if result is None:
            return {"instructions": "", "name": None, "version": None}
        return {
            "instructions": result.instructions or "",
            "name": result.server_info.name if result.server_info else None,
            "version": result.server_info.version if result.server_info else None,
        }
