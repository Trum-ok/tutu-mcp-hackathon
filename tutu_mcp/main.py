"""Entrypoint: `uv run python -m tutu_mcp.main`.

TUTU_PROXY_MODE=mock (default) serves recorded fixtures — no network calls,
safe to hammer during a demo. TUTU_PROXY_MODE=live proxies the real
mcp.tutu.ru and is subject to its shared rate limit; use sparingly.
"""

import anyio

from tutu_mcp.backend import ToolBackend
from tutu_mcp.config import load_settings
from tutu_mcp.proxy.server import build_server
from tutu_mcp.replay.mock_client import MockUpstreamClient
from tutu_mcp.replay.store import FixtureStore
from tutu_mcp.upstream.client import UpstreamClient


async def _run() -> None:
    settings = load_settings()

    if settings.mode == "mock":
        backend: ToolBackend = MockUpstreamClient(FixtureStore(settings.fixtures_dir))
        server = build_server(backend, name="tutu-mcp-proxy-mock")
        print(f"[tutu-mcp-proxy] mock mode — fixtures={settings.fixtures_dir}")
        print(f"[tutu-mcp-proxy] listening on http://{settings.host}:{settings.port}/mcp")
        await server.run_streamable_http_async(host=settings.host, port=settings.port)
        return

    async with UpstreamClient(settings.upstream_url) as backend:
        server = build_server(backend, name="tutu-mcp-proxy")
        print(f"[tutu-mcp-proxy] live mode — upstream={settings.upstream_url}")
        print(f"[tutu-mcp-proxy] listening on http://{settings.host}:{settings.port}/mcp")
        await server.run_streamable_http_async(host=settings.host, port=settings.port)


def main() -> None:
    anyio.run(_run)


if __name__ == "__main__":
    main()
