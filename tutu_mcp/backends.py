"""Which `ToolBackend` a run gets, decided in one place.

`backend.py` states the CONTRACT (the Protocol and the error hierarchy); this
module does the WIRING — reading `Settings` and handing back the live client,
the fixture-backed mock, or the record-on-miss decorator around both. It cannot
live in `backend.py` itself: `upstream/client.py` and `replay/store.py` already
import that module, so importing them back would be an import cycle.

Before this existed, `tutu_mcp/main.py` and `evals/run.py` each hand-wrote the
same branch, and a third backend would have meant editing both. The two
switches stay deliberately different and that difference is now stated as an
argument rather than implied by two lookalike branches: the server reads
`TUTU_PROXY_MODE`, while the harness passes its own `--live` flag and ignores
`settings.mode` entirely — a `.env` left on `live` must never drag an eval run
onto the shared hackathon rate limit.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from tutu_mcp.backend import ToolBackend
from tutu_mcp.config import Settings
from tutu_mcp.replay.mock_client import MockUpstreamClient
from tutu_mcp.replay.recording import RecordingBackend
from tutu_mcp.replay.store import FixtureStore
from tutu_mcp.upstream.client import UpstreamClient


@dataclass(frozen=True)
class Wiring:
    """A backend and the server `instructions` that belong with it.

    They travel together because they must come from the same source: measuring
    the mock's tool surface against the live server's instructions (or the other
    way round) would silently compare two things that never coexisted.

    `instructions` is a callable rather than a string so that reading them stays
    optional. The live client already holds them after `initialize`, but the mock
    has to read `_meta/server_info.json` — and `serve` in mock mode never needs
    them, so it must not fail to start over a file it will not use.
    """

    backend: ToolBackend
    instructions: Callable[[], str]


@asynccontextmanager
async def backend_for(
    settings: Settings, *, live: bool, record_missing: bool = False
) -> AsyncIterator[Wiring]:
    """The one place a `ToolBackend` is chosen. A third backend is a new branch
    here and nothing else."""
    store = FixtureStore(settings.fixtures_dir)

    if not live:
        if record_missing:
            # Recording is the one thing the mock cannot do — it has no upstream to
            # record from. Caught here so no caller can wire an impossible combination.
            raise ValueError("record_missing requires live=True")
        yield Wiring(MockUpstreamClient(store), store.instructions)
        return

    async with UpstreamClient(
        settings.upstream_url, timeout_s=settings.upstream_timeout_s
    ) as client:
        backend: ToolBackend = RecordingBackend(store, client) if record_missing else client
        yield Wiring(backend, lambda: client.server_info()["instructions"])
