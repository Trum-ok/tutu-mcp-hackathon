"""Shared contract between the live upstream client and the fixture-backed mock client.

`ToolBackend` is the seam the proxy server codes against, so it never knows
whether it's talking to the real `mcp.tutu.ru` or to recorded fixtures.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCallResult:
    text: str
    is_error: bool


class ToolBackend(Protocol):
    async def list_tools(self) -> list[dict[str, Any]]:
        """Raw tool descriptors as returned by upstream `tools/list` (name/description/inputSchema/annotations)."""
        ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        """Call `name` with `arguments`, returning its single text content block."""
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
