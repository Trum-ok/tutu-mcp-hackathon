"""Driver that records real mcp.tutu.ru responses as fixtures for `TUTU_PROXY_MODE=mock`.

Run against the live server (once, or after upstream changes):

    uv run python tutu.py record

WHAT to record is not here. The call list is a property of the eval scenarios
that replay it — same cities, same dates, same argument spelling — so it lives
in `evals/fixtures_recipe.py`, which may import `evals.scenarios` and therefore
cannot drift from it. This module only knows how to drive a list of
`RecordSpec`s: it is the mechanism, `evals/` is the policy. Nothing here may
import `evals` (see the layering note in `tutu_mcp/config.py`).

Deliberately sequential with a short delay between calls — the hackathon's
rate limit is shared across every team, so this driver is polite by design
rather than fast. Specs are recorded in order and a later one may read an
earlier one's result (a `details_ref` or `checkout_ref`), which is why
`arguments` can be a callable.

Known gap: no 429 fixture. Provoking one on purpose would burn the shared
rate limit for every other team on-site, so that fixture has to be
hand-authored (or captured incidentally) instead of recorded here.
"""

import asyncio
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from tutu_mcp.config import load_settings
from tutu_mcp.replay.store import FixtureStore
from tutu_mcp.upstream.client import UpstreamClient

DELAY_BETWEEN_CALLS_SECONDS = 0.4

Recorded = dict[str, Any]
ArgumentsFor = Callable[[Recorded], dict[str, Any] | None]


@dataclass(frozen=True)
class RecordSpec:
    """One call to record, keyed in `Recorded` as `"<tool>/<scenario>"`.

    `arguments` may be a callable when the call depends on an earlier result
    (a rail `details_ref`, a hotel `checkout_ref`). Returning `None` from it
    means "the dependency isn't there" — the spec is skipped with a note rather
    than failing the whole run, because a partial fixture set is still useful.
    """

    tool: str
    scenario: str
    arguments: dict[str, Any] | ArgumentsFor
    group: str = ""


async def record_one(
    store: FixtureStore, client: UpstreamClient, tool: str, scenario: str, arguments: dict[str, Any]
) -> Any | None:
    try:
        result = await client.call_tool(tool, arguments)
    except Exception as exc:  # this is a recording script — log and keep going
        print(f"  ! {tool}/{scenario} raised {type(exc).__name__}: {exc}")
        return None

    store.save_tool_result(tool, scenario, arguments, result)
    status = "ERROR" if result.is_error else "ok"
    print(f"  - {tool}/{scenario}: {status}, {len(result.text)} bytes")
    await asyncio.sleep(DELAY_BETWEEN_CALLS_SECONDS)

    if result.is_error:
        return None
    try:
        return json.loads(result.text)
    except json.JSONDecodeError:
        return None


async def record_catalog(store: FixtureStore, client: UpstreamClient) -> None:
    """`initialize` + `tools/list` — what every mock run needs before any call."""
    print("initialize")
    info = client.server_info()
    store.save_server_info(info)
    print(f"  - {info['name']} {info['version']}, instructions {len(info['instructions'])} bytes")

    print("tools/list")
    tools = await client.list_tools()
    store.save_tools_list(tools)
    print(f"  - {len(tools)} tools, {len(json.dumps(tools, ensure_ascii=False))} bytes")
    await asyncio.sleep(DELAY_BETWEEN_CALLS_SECONDS)


async def record_calls(
    store: FixtureStore, client: UpstreamClient, calls: Iterable[RecordSpec]
) -> Recorded:
    """Record each spec in order, returning what every successful call parsed to."""
    recorded: Recorded = {}
    group = ""
    for spec in calls:
        if spec.group and spec.group != group:
            group = spec.group
            print(group)

        spec_arguments = spec.arguments
        arguments = spec_arguments if isinstance(spec_arguments, dict) else spec_arguments(recorded)
        if arguments is None:
            print(f"  ! {spec.tool}/{spec.scenario}: dependency missing, skipped")
            continue

        payload = await record_one(store, client, spec.tool, spec.scenario, arguments)
        if payload is not None:
            recorded[f"{spec.tool}/{spec.scenario}"] = payload
    return recorded


async def record_fixtures(calls: Sequence[RecordSpec]) -> None:
    settings = load_settings()
    store = FixtureStore(settings.fixtures_dir)

    async with UpstreamClient(
        settings.upstream_url, timeout_s=settings.upstream_timeout_s
    ) as client:
        print(f"Recording from {settings.upstream_url} into {settings.fixtures_dir}")
        await record_catalog(store, client)
        await record_calls(store, client, calls)
        print("Done.")
