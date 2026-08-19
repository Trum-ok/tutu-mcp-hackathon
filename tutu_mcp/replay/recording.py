"""Record-on-miss backend: replay what we have, fetch and record what we don't.

A model-driven eval calls tools with arguments nobody predicted, so a purely
pre-recorded fixture set will always have holes on the first pass. This backend
closes them: run the eval once against live upstream with recording on, then every
later run is fully offline and reproducible — standard VCR "record new episodes".

It touches the network only on a miss, and never in plain mock mode, so it cannot
quietly start hammering the shared hackathon rate limit.
"""

from typing import Any

from tutu_mcp.backend import ToolBackend, ToolCallResult
from tutu_mcp.replay.store import (
    FixtureMissingError,
    FixtureNotFoundError,
    FixtureStore,
    scenario_slug,
)


class RecordingBackend:
    def __init__(self, store: FixtureStore, upstream: ToolBackend) -> None:
        self._store = store
        self._upstream = upstream
        self.recorded: list[tuple[str, str]] = []
        # errored calls we refused to record, so a run can say so instead of
        # reporting the miss as closed
        self.skipped_errors: list[tuple[str, str]] = []

    async def list_tools(self) -> list[dict[str, Any]]:
        try:
            return self._store.load_tools_list()
        except FixtureMissingError:
            tools = await self._upstream.list_tools()
            self._store.save_tools_list(tools)
            return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        try:
            return self._store.find_result(name, arguments)
        except FixtureNotFoundError:
            pass

        result = await self._upstream.call_tool(name, arguments)
        scenario = scenario_slug(name, arguments)
        if result.is_error:
            # A rate-limited, timed-out or rejected call is not a recording of what
            # Tutu answers for these arguments. Saved once, it would be replayed as
            # data by every later offline run — an outage frozen into the fixture
            # set, indistinguishable from a real response. Hand it to the caller,
            # who already classifies upstream failures, and record nothing.
            self.skipped_errors.append((name, scenario))
            return result

        self._store.save_tool_result(name, scenario, arguments, result)
        self.recorded.append((name, scenario))
        return result
