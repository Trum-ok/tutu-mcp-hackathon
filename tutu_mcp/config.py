"""Runtime settings for the proxy server itself.

Values come from the environment. A `.env` file in the repo root is loaded on
import (it never overrides variables already exported in the shell, so an
explicit `export` still wins). `.env` is gitignored — put the OpenAI key there,
never in tracked files; see `.env.example` for the full list.

Only `TUTU_*` lives here. The eval harness has its own credentials and model
defaults in `evals/config.py`; keeping them out means the proxy never has to
import the harness that measures it.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(REPO_ROOT / ".env", override=False)

DEFAULT_CATALOG_TTL_S = 900.0
"""How long a fetched `tools/list` stays good, in seconds.

Lives here rather than next to `proxy.server.build_server`, which takes it as an
argument, so that the number and the `TUTU_CATALOG_TTL_S` that overrides it are
declared in one place — the same rule every other setting in this file follows.
"""

DEFAULT_PORT = 8800
"""Port `serve` takes when neither `TUTU_PROXY_PORT` nor `PORT` says otherwise."""


class SettingsError(ValueError):
    """An environment variable holding a value these settings cannot take.

    Its own type rather than a bare `ValueError` because both entrypoints
    (`tutu.py` and `python -m tutu_mcp.main`) catch it to print a message
    instead of a traceback — and catching plain `ValueError` there would
    swallow real bugs along with a typo in `.env`.
    """


def _number(name: str, raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        raise SettingsError(f"{name}={raw!r} — ожидалось число.") from None


def _seconds(name: str, raw: str) -> float:
    value = _number(name, raw)
    if value <= 0:
        raise SettingsError(f"{name}={raw!r} — ожидалось положительное число секунд.")
    return value


def _port(name: str, raw: str) -> int:
    value = _number(name, raw)
    if value != int(value) or not (1 <= value <= 65535):
        raise SettingsError(f"{name}={raw!r} — ожидался номер порта от 1 до 65535.")
    return int(value)


@dataclass(frozen=True)
class Settings:
    upstream_url: str
    upstream_timeout_s: float
    mode: str  # "live" or "mock"
    fixtures_dir: Path
    catalog_ttl_s: float
    host: str
    port: int

    def __post_init__(self) -> None:
        if self.mode not in ("live", "mock"):
            raise SettingsError(
                f"TUTU_PROXY_MODE={self.mode!r} — допустимы только 'live' и 'mock'."
            )


def _configured_port() -> int:
    """`PORT` is the fallback because every PaaS (Render, Railway, Fly, Heroku)
    assigns the port that way and ignores anything else; `TUTU_PROXY_PORT` still
    wins so a local run is never at the mercy of a stray `PORT` in the shell.

    Whichever variable was actually read is the one named in the error: told that
    `PORT` is unusable, someone would go and edit `TUTU_PROXY_PORT` — the variable
    that is not in play.
    """
    for name in ("TUTU_PROXY_PORT", "PORT"):
        raw = os.environ.get(name)
        if raw:
            return _port(name, raw)
    return DEFAULT_PORT


def load_settings() -> Settings:
    return Settings(
        upstream_url=os.environ.get("TUTU_UPSTREAM_URL", "https://mcp.tutu.ru/mcp"),
        # A hung upstream must not hang `serve`/`tutu.py record` forever — it has
        # to fail loudly enough to turn into a proxy error instead of silence.
        upstream_timeout_s=_seconds(
            "TUTU_UPSTREAM_TIMEOUT_S", os.environ.get("TUTU_UPSTREAM_TIMEOUT_S") or "20"
        ),
        mode=os.environ.get("TUTU_PROXY_MODE", "mock"),
        fixtures_dir=Path(os.environ.get("TUTU_FIXTURES_DIR", str(REPO_ROOT / "fixtures"))),
        # Upstream can add a tool or change a schema at any time, and a catalog
        # cached for the life of the process looks exactly like a fresh one — so
        # a long-running proxy served a stale surface until someone restarted it.
        catalog_ttl_s=_seconds(
            "TUTU_CATALOG_TTL_S", os.environ.get("TUTU_CATALOG_TTL_S") or str(DEFAULT_CATALOG_TTL_S)
        ),
        # The default host stays 127.0.0.1 — a container that must listen publicly
        # says so itself (see the Dockerfile).
        host=os.environ.get("TUTU_PROXY_HOST", "127.0.0.1"),
        port=_configured_port(),
    )
