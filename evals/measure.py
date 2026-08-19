"""Reproducible byte accounting for the trimmed catalog.

    uv run python tutu.py measure

Every "X -> Y bytes" claim in the README and the docs page traces back to this
module's output, not to the one-off live probe against mcp.tutu.ru that
`docs/findings.md` records (2026-08-19, server `0.38.0`) — that capture used
Tutu's own wire serialization, which nothing here can reproduce offline, and a
number nobody can regenerate from the committed fixtures doesn't belong in
prose that is supposed to prove itself. This instead re-serializes the parsed
fixture with `json.dumps(..., ensure_ascii=False)` — the exact method
`tests/test_compact_tools.py` already asserts against — so the figures here
are reproducible from `fixtures/` alone, on any machine, at any time, at the
cost of not being byte-identical to that one live capture (they land a couple
of percent apart; `docs/findings.md` keeps the original as a dated data point).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tutu_mcp.proxy.compact_tools import apply_compact_overrides, apply_result_appendix
from tutu_mcp.proxy.surface import PROXY_INSTRUCTIONS, proxy_catalog

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_META = REPO_ROOT / "fixtures" / "_meta"


def _bytes_of(text: str | None) -> int:
    return len((text or "").encode())


def _size(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False).encode())


def _schema_prose_bytes(node: Any) -> int:
    """Sum of every `description` string nested anywhere inside a schema."""
    if isinstance(node, dict):
        total = _bytes_of(node["description"]) if isinstance(node.get("description"), str) else 0
        return total + sum(_schema_prose_bytes(v) for k, v in node.items() if k != "description")
    if isinstance(node, list):
        return sum(_schema_prose_bytes(item) for item in node)
    return 0


@dataclass(frozen=True)
class CatalogMeasurement:
    n_tools_raw: int
    n_tools_proxy: int
    baseline_bytes: int
    proxy_bytes: int
    baseline_with_init_bytes: int
    proxy_with_init_bytes: int
    top_level_description_bytes: int
    schema_prose_bytes: int
    targeted_top_level_before: int
    targeted_top_level_after: int
    schema_prose_after: int
    rail_instructions_before: int
    rail_instructions_after: int

    @property
    def reduction(self) -> float:
        return (self.baseline_bytes - self.proxy_bytes) / self.baseline_bytes

    @property
    def reduction_with_init(self) -> float:
        return (
            self.baseline_with_init_bytes - self.proxy_with_init_bytes
        ) / self.baseline_with_init_bytes


def measure_catalog() -> CatalogMeasurement:
    raw = json.loads((FIXTURES_META / "tools_list.json").read_text(encoding="utf-8"))
    compacted, trimmed_originals = apply_compact_overrides(raw)
    proxy_tools = proxy_catalog(compacted)

    server_info = json.loads((FIXTURES_META / "server_info.json").read_text(encoding="utf-8"))
    tutu_instructions_bytes = _bytes_of(server_info["instructions"])
    proxy_instructions_bytes = _bytes_of(PROXY_INSTRUCTIONS)

    targeted = set(trimmed_originals)
    schema_prose_after = sum(_schema_prose_bytes(t.get("inputSchema")) for t in compacted)

    rail_fixture = json.loads(
        (REPO_ROOT / "fixtures" / "get_rail_instructions" / "default.json").read_text(
            encoding="utf-8"
        )
    )
    rail_text = rail_fixture["result"]["text"]
    rail_after = apply_result_appendix("get_rail_instructions", rail_text, trimmed_originals)

    baseline = _size(raw)
    proxy = _size(proxy_tools)
    return CatalogMeasurement(
        n_tools_raw=len(raw),
        n_tools_proxy=len(proxy_tools),
        baseline_bytes=baseline,
        proxy_bytes=proxy,
        baseline_with_init_bytes=baseline + tutu_instructions_bytes,
        proxy_with_init_bytes=proxy + proxy_instructions_bytes,
        top_level_description_bytes=sum(_bytes_of(t.get("description")) for t in raw),
        schema_prose_bytes=sum(_schema_prose_bytes(t.get("inputSchema")) for t in raw),
        targeted_top_level_before=sum(
            _bytes_of(t.get("description")) for t in raw if t["name"] in targeted
        ),
        targeted_top_level_after=sum(
            _bytes_of(t.get("description")) for t in compacted if t["name"] in targeted
        ),
        schema_prose_after=schema_prose_after,
        rail_instructions_before=_bytes_of(rail_text),
        rail_instructions_after=_bytes_of(rail_after),
    )


def print_report() -> int:
    m = measure_catalog()
    print("=== tools/list: baseline vs proxy (reproducible from fixtures/) ===")
    print(f"  tools            {m.n_tools_raw} -> {m.n_tools_proxy}  (2 synthetic added)")
    print(f"  catalog bytes    {m.baseline_bytes:,} -> {m.proxy_bytes:,}  (-{m.reduction:.1%})")
    print(
        f"  + initialize     {m.baseline_with_init_bytes:,} -> {m.proxy_with_init_bytes:,}"
        f"  (-{m.reduction_with_init:.1%})"
    )
    print()
    print("--- where the bytes are ---")
    print(
        f"  top-level description   {m.top_level_description_bytes:,} bytes total; "
        f"3 targeted tools {m.targeted_top_level_before:,} -> {m.targeted_top_level_after:,}"
    )
    print(
        f"  inputSchema prose        {m.schema_prose_bytes:,} bytes total"
        f" -> {m.schema_prose_after:,} after trimming"
    )
    print(
        f"  get_rail_instructions    {m.rail_instructions_before:,}"
        f" -> {m.rail_instructions_after:,} bytes (absorbs the trimmed appendix)"
    )
    return 0
