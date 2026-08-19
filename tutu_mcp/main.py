"""Entrypoint: `uv run python tutu.py serve`.

TUTU_PROXY_MODE=mock (default) serves recorded fixtures — no network calls,
safe to hammer during a demo. TUTU_PROXY_MODE=live proxies the real
mcp.tutu.ru and is subject to its shared rate limit; use sparingly.

`main()` is also the container's entrypoint (`python -m tutu_mcp.main`, see the
Dockerfile), which is why the startup failures are reported here rather than in
`tutu.py`: both ways in must answer a bad setting or an unreachable upstream
with a line, not a traceback.
"""

import socket
import sys

import anyio

from tutu_mcp.backend import BackendError
from tutu_mcp.backends import backend_for
from tutu_mcp.config import SettingsError, load_settings
from tutu_mcp.proxy.server import build_server

BAD_SETTINGS = 2
UPSTREAM_UNREACHABLE = 1


def claim_address(host: str, port: int) -> None:
    """Binds `host:port` a moment before uvicorn does, and fails here if it can't.

    uvicorn binds only once `run_streamable_http_async` has control — long after
    the "listening" line below is printed. On a busy port that line stayed on
    screen as a lie, above uvicorn's bare errno and an exit code nobody asked
    for. The same bind done here turns that into a message naming the address
    and the variable to change, and leaves the line after it true.
    """
    try:
        family, socktype, proto, _canonname, sockaddr = socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM
        )[0]
    except OSError as exc:
        raise SettingsError(f"TUTU_PROXY_HOST={host!r} — адрес не разрешается ({exc}).") from None

    with socket.socket(family, socktype, proto) as probe:
        # The option uvicorn's own listener sets: without it a port uvicorn would
        # take happily could be refused here, which is the opposite of the point.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(sockaddr)
        except OSError as exc:
            raise SettingsError(
                f"Адрес {host}:{port} занять не удалось ({exc}) — "
                f"освободите порт или задайте другой в TUTU_PROXY_PORT."
            ) from None


async def serve() -> None:
    settings = load_settings()
    live = settings.mode == "live"
    name = "tutu-mcp-proxy" if live else "tutu-mcp-proxy-mock"
    source = f"upstream={settings.upstream_url}" if live else f"fixtures={settings.fixtures_dir}"

    # Before the upstream connection: an address we cannot take makes the whole
    # run pointless, and finding that out costs neither a request nor a wait.
    claim_address(settings.host, settings.port)

    async with backend_for(settings, live=live) as wiring:
        server = build_server(wiring.backend, name=name, catalog_ttl_s=settings.catalog_ttl_s)
        # flush: under a pipe (docker logs, a `tee` into a demo terminal) stdout is
        # block-buffered, and a server that then blocks forever never fills the
        # block — the startup lines would appear only when the proxy is killed.
        print(f"[tutu-mcp-proxy] {settings.mode} mode — {source}", flush=True)
        print(
            f"[tutu-mcp-proxy] listening on http://{settings.host}:{settings.port}/mcp", flush=True
        )
        await server.run_streamable_http_async(host=settings.host, port=settings.port)


def main() -> None:
    try:
        anyio.run(serve)
    except SettingsError as exc:
        print(f"[tutu-mcp-proxy] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(BAD_SETTINGS) from None
    except BackendError as exc:
        # Only the startup handshake reaches here: once serving, a backend failure
        # is classified per call by `proxy.dispatch` and never leaves the server.
        print(
            f"[tutu-mcp-proxy] upstream недоступен: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(UPSTREAM_UNREACHABLE) from None


if __name__ == "__main__":
    main()
