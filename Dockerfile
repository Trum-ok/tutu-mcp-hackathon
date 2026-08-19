# Container image for deploying the proxy — the demo needs a URL a judge can
# paste into their own MCP client, not a laptop on conference wifi.
#
# Defaults to mock mode on purpose: the hackathon's rate limit at mcp.tutu.ru is
# shared with every other team, and a public URL anyone may poke is exactly the
# thing that would burn it. Set TUTU_PROXY_MODE=live to proxy the real server.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Dependencies first, so a code change doesn't re-resolve the whole lockfile.
# --no-install-project: the app is copied in below, not installed as a package.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY tutu_mcp/ ./tutu_mcp/
COPY fixtures/ ./fixtures/

ENV TUTU_PROXY_MODE=mock \
    TUTU_FIXTURES_DIR=/app/fixtures \
    TUTU_PROXY_HOST=0.0.0.0 \
    PORT=8800 \
    PYTHONUNBUFFERED=1

EXPOSE 8800

# `tutu.py serve` is the documented entrypoint locally, but that CLI imports
# `evals` at module level and this image deliberately ships the proxy alone —
# so it runs the same server through the module the CLI itself calls.
CMD ["uv", "run", "--no-dev", "python", "-m", "tutu_mcp.main"]
