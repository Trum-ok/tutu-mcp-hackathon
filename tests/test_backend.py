"""`call_with_timeout_retry` — the one retry policy shared by `proxy.dispatch`
(a `tools/call`) and `proxy.server` (a `tools/list` catalog load).
"""

import pytest

from tutu_mcp.backend import BackendTimeoutError, BackendUnavailableError, call_with_timeout_retry


async def test_retries_exactly_once_on_a_timeout_and_returns_the_recovered_value():
    attempts = 0

    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise BackendTimeoutError("slow")
        return "ok"

    assert await call_with_timeout_retry(flaky) == "ok"
    assert attempts == 2


async def test_a_second_consecutive_timeout_is_not_retried_again():
    attempts = 0

    async def always_slow():
        nonlocal attempts
        attempts += 1
        raise BackendTimeoutError("slow")

    with pytest.raises(BackendTimeoutError):
        await call_with_timeout_retry(always_slow)
    assert attempts == 2


async def test_a_non_timeout_failure_is_not_retried():
    attempts = 0

    async def broken():
        nonlocal attempts
        attempts += 1
        raise BackendUnavailableError("boom")

    with pytest.raises(BackendUnavailableError):
        await call_with_timeout_retry(broken)
    assert attempts == 1
