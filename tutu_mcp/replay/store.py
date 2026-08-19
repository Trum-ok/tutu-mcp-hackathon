"""On-disk store for recorded MCP responses (VCR-style: match by tool + exact arguments).

Layout under `fixtures_dir`:
    _meta/tools_list.json                 -- raw upstream `tools/list` response
    _meta/server_info.json                -- upstream `initialize` result (instructions, version)
    <tool_name>/<scenario>.json           -- one recorded `tools/call` response

Each scenario file keys playback on the exact (normalized) arguments it was
recorded with, so an agent that calls a tool the same way it was recorded
gets the same answer back — same idea as the mock flag described in the
Tutu MCP article: a correct request plays back real data, anything else is
a miss the developer has to go record.
"""

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tutu_mcp.backend import BackendMissError, ToolCallResult

TOOLS_LIST_FIXTURE = "_meta/tools_list.json"
SERVER_INFO_FIXTURE = "_meta/server_info.json"

_SLUG_UNSAFE = re.compile(r"[^a-z0-9]+")


def normalize_arguments(arguments: dict[str, Any], defaults: dict[str, Any] | None = None) -> str:
    """Key for fixture lookup: the same REQUEST must produce the same key.

    Two things a model does that a human recording fixtures does not:

    - spells out optional arguments as `null` (`from_city: None`);
    - spells out optional arguments at their schema default (`page: 1`,
      `sort: "price_asc"`, `view: "compact"`).

    Both are, to the upstream, identical to omitting the argument — so keying on
    the raw dict would miss a perfectly good fixture and report it as a gap in our
    recording. Pruning them is what makes mock-mode evals usable with a real model
    at all; without it nearly every call misses.

    A value that merely LOOKS default-ish is not pruned — only one equal to the
    default the tool's own schema declares. `page_size: 30` against a default of
    10 is a genuinely different request and stays a miss, which is correct.
    """
    defaults = defaults or {}
    pruned = {
        k: v
        for k, v in arguments.items()
        if v is not None and not (k in defaults and v == defaults[k])
    }
    return json.dumps(pruned, ensure_ascii=False, sort_keys=True)


def scenario_slug(tool: str, arguments: dict[str, Any]) -> str:
    """Stable, filesystem-safe name for an auto-recorded scenario.

    Uses sha1 rather than `hash()`: string hashing is salted per process, so
    `hash()` would give the same request a different filename on every run.
    """
    digest = hashlib.sha1(normalize_arguments(arguments).encode()).hexdigest()[:8]
    hint_parts = [str(v) for _, v in sorted(arguments.items()) if isinstance(v, str | int)][:3]
    hint = _SLUG_UNSAFE.sub("_", "_".join(hint_parts).lower()).strip("_")[:40]
    return f"auto_{hint}_{digest}" if hint else f"auto_{digest}"


class FixtureNotFoundError(BackendMissError):
    def __init__(self, tool: str, arguments: dict[str, Any], available: list[str]) -> None:
        self.tool = tool
        self.arguments = arguments
        self.available = available
        listed = ", ".join(available) if available else "(none recorded)"
        super().__init__(
            f"No fixture for {tool}({arguments!r}). Recorded scenarios for {tool}: {listed}. "
            f"Run `uv run python tutu.py record` against the live server to add it."
        )


class FixtureStore:
    def __init__(self, fixtures_dir: Path) -> None:
        self.fixtures_dir = fixtures_dir
        self._defaults: dict[str, dict[str, Any]] | None = None

    def _tool_dir(self, tool: str) -> Path:
        return self.fixtures_dir / tool

    def schema_defaults(self, tool: str) -> dict[str, Any]:
        """Per-argument defaults the tool's own `inputSchema` declares.

        Read from the recorded `tools/list`, so the source of truth is upstream's
        schema rather than a hand-maintained list that would drift the first time
        Tutu changes a default.
        """
        if self._defaults is None:
            self._defaults = {}
            try:
                for entry in self.load_tools_list():
                    props = (entry.get("inputSchema") or {}).get("properties") or {}
                    self._defaults[entry["name"]] = {
                        name: field["default"]
                        for name, field in props.items()
                        if isinstance(field, dict) and "default" in field
                    }
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                # No catalog recorded yet — fall back to exact matching rather
                # than refusing to serve fixtures at all.
                self._defaults = {}
        return self._defaults.get(tool, {})

    def save_tool_result(
        self, tool: str, scenario: str, arguments: dict[str, Any], result: ToolCallResult
    ) -> Path:
        tool_dir = self._tool_dir(tool)
        tool_dir.mkdir(parents=True, exist_ok=True)
        path = tool_dir / f"{scenario}.json"
        payload = {
            "tool": tool,
            "scenario": scenario,
            "arguments": arguments,
            "result": asdict(result),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def load_scenarios(self, tool: str) -> list[dict[str, Any]]:
        tool_dir = self._tool_dir(tool)
        if not tool_dir.is_dir():
            return []
        scenarios = []
        for path in sorted(tool_dir.glob("*.json")):
            scenarios.append(json.loads(path.read_text(encoding="utf-8")))
        return scenarios

    def find_result(self, tool: str, arguments: dict[str, Any]) -> ToolCallResult:
        defaults = self.schema_defaults(tool)
        target = normalize_arguments(arguments, defaults)
        scenarios = self.load_scenarios(tool)
        for entry in scenarios:
            if normalize_arguments(entry["arguments"], defaults) == target:
                return ToolCallResult(**entry["result"])
        raise FixtureNotFoundError(tool, arguments, [e["scenario"] for e in scenarios])

    def save_tools_list(self, tools: list[dict[str, Any]]) -> Path:
        path = self.fixtures_dir / TOOLS_LIST_FIXTURE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tools, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def load_tools_list(self) -> list[dict[str, Any]]:
        path = self.fixtures_dir / TOOLS_LIST_FIXTURE
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing — run `uv run python tutu.py record` first")
        return json.loads(path.read_text(encoding="utf-8"))

    def save_server_info(self, info: dict[str, Any]) -> Path:
        path = self.fixtures_dir / SERVER_INFO_FIXTURE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def load_server_info(self) -> dict[str, Any]:
        path = self.fixtures_dir / SERVER_INFO_FIXTURE
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing — run `uv run python tutu.py record` first")
        return json.loads(path.read_text(encoding="utf-8"))

    def instructions(self) -> str:
        """Upstream's always-on `initialize` instructions — part of what the
        baseline tool surface costs, so it belongs in the token measurement."""
        return self.load_server_info().get("instructions", "")
