"""`SYNTHETIC`/`proxy_catalog` are the one place a synthetic tool is declared —
these guard against the three-copy drift that existed before `surface.py` did
(catalog assembly duplicated in `proxy/server.py` and `evals/variants.py`,
tool names re-hardcoded a third time in `proxy/dispatch.py`'s routing).
"""

from tutu_mcp.groundedness import CHECK_GROUNDEDNESS_TOOL
from tutu_mcp.premises import ASSESS_REQUEST_TOOL
from tutu_mcp.proxy.dispatch import dispatch
from tutu_mcp.proxy.surface import SYNTHETIC, proxy_catalog


def test_synthetic_registry_is_keyed_by_each_tools_own_name():
    assert set(SYNTHETIC) == {ASSESS_REQUEST_TOOL["name"], CHECK_GROUNDEDNESS_TOOL["name"]}
    for name, (spec, _run) in SYNTHETIC.items():
        assert spec["name"] == name


def test_proxy_catalog_appends_every_synthetic_tool_to_the_compacted_upstream_ones():
    compacted = [{"name": "search_rail", "description": "...", "inputSchema": {}}]

    catalog = proxy_catalog(compacted)

    assert catalog[0] is compacted[0]
    assert {t["name"] for t in catalog[1:]} == set(SYNTHETIC)
    assert len(catalog) == 1 + len(SYNTHETIC)


async def test_dispatch_reports_every_synthetic_tool_unknown_on_a_session_less_baseline():
    """The baseline (upstream-only) variant never gets a session, so every tool
    in `SYNTHETIC` must report itself unknown there instead of running —
    otherwise the baseline measurement would leak proxy-only behavior. A tool
    outside `SYNTHETIC` reaches the backend instead; see `test_dispatch.py`."""

    class _NoBackend:
        async def list_tools(self):
            return []

        async def call_tool(self, name, arguments):
            raise AssertionError("synthetic tools must not reach the backend")

    for name in SYNTHETIC:
        result = await dispatch(None, _NoBackend(), name, {}, {})
        assert result.text == f"unknown tool: {name}"
        assert result.is_error is True
