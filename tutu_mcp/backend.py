"""Shared contract between the live upstream client and the fixture-backed mock client.

`ToolBackend` is the seam the proxy server codes against, so it never knows
whether it's talking to the real `mcp.tutu.ru` or to recorded fixtures.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCallResult:
    """One `tools/call` answer, reduced to the two fields this proxy acts on.

    A lossy view of `CallToolResult` on purpose, and the loss has to be stated
    because it is silent: `structuredContent` and every non-text content block
    (image / audio / embedded resource) are dropped by
    `UpstreamClient.call_tool`, which joins the text blocks and keeps nothing
    else. Today that costs nothing — Tutu answers every tool with a single JSON
    text block, which is also what `premises` and `groundedness` parse and what
    `FixtureStore` round-trips through `asdict()` — but the day upstream starts
    returning `structuredContent`, the proxy would forward an answer with a hole
    in it and no error anywhere to show for it.

    Adding a field here is only half the change: recorded fixtures carry exactly
    these keys (`ToolCallResult(**entry["result"])`), so every fixture predating
    the new field still loads, and both the server and `evals/variants.py` have
    to be taught to put it back on the wire.
    """

    text: str
    is_error: bool


class ToolBackend(Protocol):
    async def list_tools(self) -> list[dict[str, Any]]:
        """Raw tool descriptors as returned by upstream `tools/list` (name/description/inputSchema/annotations)."""
        ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        """Call `name` with `arguments`, returning its text content — everything
        else the answer carried is dropped here, see `ToolCallResult`."""
        ...


class BackendError(Exception):
    """A `ToolBackend` failure, already classified into the domain status
    `tutu_mcp.proxy.dispatch.backend_error` reports to the agent.

    The Protocol above only covers the happy path — nothing stops a
    `ToolBackend` from raising whatever its underlying transport happens to
    raise. Without this hierarchy, the caller (`dispatch.backend_error`)
    would have to know each implementation's exception types by name (a
    mock's `FixtureNotFoundError`, a live client's `MCPError`), which breaks
    the moment a third backend raises something neither branch expects.
    Each `ToolBackend` implementation instead translates its own failures
    into one of these subclasses at its own boundary, so the caller only
    ever needs to check `isinstance(exc, BackendError)`.
    """

    status = "upstream_unavailable"


class BackendMissError(BackendError):
    """The backend has no answer for this exact call (a mock's fixture gap)."""

    status = "fixture_not_found"


class BackendCorruptError(BackendError):
    """The backend's own recorded data is unreadable (a mock's malformed fixture).

    Distinct from `BackendMissError` because the fix is different: a miss is
    closed by recording the call, a corrupt fixture by re-recording or deleting
    the file that already exists.
    """

    status = "fixture_corrupt"


class BackendUnknownToolError(BackendError):
    """The call named a tool this backend does not serve at all.

    Kept apart from `BackendMissError` because the two send the caller in
    opposite directions: a miss is a hole in what we recorded and is closed by
    recording it, while this one says the name itself is wrong and no amount of
    recording will help. Reported to the agent as its own status so a model that
    invented a tool name gets told so, instead of being pointed at
    `tutu.py record` for a tool that does not exist.
    """

    status = "unknown_tool"


class BackendTimeoutError(BackendError):
    """The backend didn't answer within its configured timeout."""

    status = "upstream_timeout"


class BackendUnavailableError(BackendError):
    """The backend failed for any other reason (connection, protocol, ...)."""

    status = "upstream_unavailable"


ERROR_STATUSES = frozenset(
    {
        BackendMissError.status,
        BackendCorruptError.status,
        BackendUnknownToolError.status,
        BackendTimeoutError.status,
        BackendUnavailableError.status,
    }
)
"""Every status `dispatch.backend_error` can produce — the single source of
truth for anything downstream that needs to recognize one of our own error
payloads (e.g. `tutu_mcp.groundedness`, which must not treat an error
payload's echoed arguments as grounding evidence)."""


LOCAL_FIXTURE_STATUSES = frozenset({BackendMissError.status, BackendCorruptError.status})
"""The statuses that mean the gap is in OUR recorded fixtures rather than in
upstream. Callers that report "how often did Tutu fail" (`proxy.dispatch`'s
`fixture_miss`, and the eval transcript counters built on it) must not count
these, or a hole in our own recording reads as Tutu misbehaving."""


async def call_with_timeout_retry[T](call: Callable[[], Awaitable[T]]) -> T:
    """Retries `call` once, but only on a `BackendTimeoutError`.

    A timeout is often transient — a queued request, a cold cache on Tutu's
    side — so one retry recovers most of them before the agent ever sees
    `upstream_timeout`. Any other failure (a broken connection, a malformed
    request) isn't made more likely to succeed by trying again immediately,
    so only a timeout gets this second chance. Shared by `proxy.dispatch`
    (a `tools/call`) and `proxy.server` (a `tools/list` catalog load) so the
    policy lives in one place rather than two lookalikes.
    """
    try:
        return await call()
    except BackendTimeoutError:
        return await call()
