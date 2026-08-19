"""Entrypoint: `uv run python tutu.py serve`.

TUTU_PROXY_MODE=mock (default) serves recorded fixtures — no network calls,
safe to hammer during a demo. TUTU_PROXY_MODE=live proxies the real
mcp.tutu.ru and is subject to its shared rate limit; use sparingly.
"""

import anyio

from tutu_mcp.backends import backend_for
from tutu_mcp.config import load_settings
from tutu_mcp.proxy.server import build_server


async def serve() -> None:
    settings = load_settings()
    live = settings.mode == "live"
    name = "tutu-mcp-proxy" if live else "tutu-mcp-proxy-mock"
    source = f"upstream={settings.upstream_url}" if live else f"fixtures={settings.fixtures_dir}"

    async with backend_for(settings, live=live) as wiring:
        server = build_server(wiring.backend, name=name)
        print(f"[tutu-mcp-proxy] {settings.mode} mode — {source}")
        print(f"[tutu-mcp-proxy] listening on http://{settings.host}:{settings.port}/mcp")
        await server.run_streamable_http_async(host=settings.host, port=settings.port)


def main() -> None:
    anyio.run(serve)


if __name__ == "__main__":
    main()
