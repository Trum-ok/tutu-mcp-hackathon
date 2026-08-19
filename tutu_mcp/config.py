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


@dataclass(frozen=True)
class Settings:
    upstream_url: str
    upstream_timeout_s: float
    mode: str  # "live" or "mock"
    fixtures_dir: Path
    host: str
    port: int

    def __post_init__(self) -> None:
        if self.mode not in ("live", "mock"):
            raise ValueError(f"mode must be 'live' or 'mock', got {self.mode!r}")


def load_settings() -> Settings:
    return Settings(
        upstream_url=os.environ.get("TUTU_UPSTREAM_URL", "https://mcp.tutu.ru/mcp"),
        # A hung upstream must not hang `serve`/`tutu.py record` forever — it has
        # to fail loudly enough to turn into a proxy error instead of silence.
        upstream_timeout_s=float(os.environ.get("TUTU_UPSTREAM_TIMEOUT_S", "20")),
        mode=os.environ.get("TUTU_PROXY_MODE", "mock"),
        fixtures_dir=Path(os.environ.get("TUTU_FIXTURES_DIR", str(REPO_ROOT / "fixtures"))),
        host=os.environ.get("TUTU_PROXY_HOST", "127.0.0.1"),
        # `PORT` is the fallback because every PaaS (Render, Railway, Fly, Heroku)
        # assigns the port that way and ignores anything else; `TUTU_PROXY_PORT`
        # still wins so a local run is never at the mercy of a stray `PORT` in the
        # shell. Defaulting the host stays 127.0.0.1 — a container that must listen
        # publicly says so itself (see the Dockerfile).
        port=int(os.environ.get("TUTU_PROXY_PORT") or os.environ.get("PORT") or "8800"),
    )
