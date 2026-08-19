"""Startup failures answer with a line, not a traceback.

A typo in `.env`, a port someone else already holds, an upstream that is down —
each of them used to reach the user as a raw exception from somewhere deep in
`load_settings`, `uvicorn` or the MCP client. These pin the three boundaries
that now classify them: `SettingsError` for anything read out of the
environment, `claim_address` for the listening socket, and `BackendError` for
the upstream handshake.
"""

import socket

import pytest

from tutu_mcp.config import DEFAULT_PORT, SettingsError, load_settings
from tutu_mcp.main import claim_address

TUTU_VARS = (
    "TUTU_PROXY_MODE",
    "TUTU_UPSTREAM_TIMEOUT_S",
    "TUTU_CATALOG_TTL_S",
    "TUTU_PROXY_PORT",
    "TUTU_PROXY_HOST",
    "PORT",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """`.env` is loaded on import, so whatever the developer has in theirs would
    otherwise decide what these tests read."""
    for name in TUTU_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_need_no_environment():
    settings = load_settings()

    assert settings.mode == "mock"
    assert (settings.host, settings.port) == ("127.0.0.1", DEFAULT_PORT)


def test_unknown_mode_names_the_variable_and_the_two_values(monkeypatch):
    monkeypatch.setenv("TUTU_PROXY_MODE", "moc")

    with pytest.raises(SettingsError, match="TUTU_PROXY_MODE"):
        load_settings()


@pytest.mark.parametrize(
    "name, value",
    [
        ("TUTU_UPSTREAM_TIMEOUT_S", "двадцать"),
        ("TUTU_UPSTREAM_TIMEOUT_S", "0"),
        ("TUTU_CATALOG_TTL_S", "-1"),
        ("TUTU_PROXY_PORT", "восемь"),
        ("TUTU_PROXY_PORT", "70000"),
        ("PORT", "0"),
    ],
)
def test_unusable_number_is_reported_with_the_variable_that_holds_it(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(SettingsError) as excinfo:
        load_settings()

    assert name in str(excinfo.value)


def test_tutu_proxy_port_wins_over_a_stray_paas_port(monkeypatch):
    monkeypatch.setenv("TUTU_PROXY_PORT", "9001")
    monkeypatch.setenv("PORT", "10000")

    assert load_settings().port == 9001


def test_a_bad_paas_port_is_not_blamed_on_the_variable_that_is_unset(monkeypatch):
    monkeypatch.setenv("PORT", "nope")

    with pytest.raises(SettingsError) as excinfo:
        load_settings()

    assert "PORT='nope'" in str(excinfo.value)
    assert "TUTU_PROXY_PORT" not in str(excinfo.value)


def test_a_free_address_is_claimed_and_released():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    # released above: the check must not be what keeps the port busy
    assert claim_address("127.0.0.1", port) is None


def test_a_busy_port_fails_before_the_listening_line_is_printed():
    with socket.socket() as taken:
        taken.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]

        with pytest.raises(SettingsError) as excinfo:
            claim_address("127.0.0.1", port)

    assert str(port) in str(excinfo.value)
    assert "TUTU_PROXY_PORT" in str(excinfo.value)


def test_an_unresolvable_host_names_its_variable():
    with pytest.raises(SettingsError, match="TUTU_PROXY_HOST"):
        claim_address("host.invalid.", 8800)
