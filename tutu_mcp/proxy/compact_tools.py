"""Trims the top-level `description` of the biggest upstream tools.

`tools/list` is always loaded by the client — Tutu's own article measured it
compressing *responses* by 74%, but the always-on tool catalog itself still
runs ~110 KB / ~30K tokens (measured live on 2026-08-19: search_rail alone
carries a 9.2 KB description, create_checkout_link a 5.9 KB one, 14.7 KB of
per-field schema). That's paid on every single session before the agent has
searched anything.

This module rewrites PROSE and nothing else. Measured on the same catalog, the
prose splits in two: ~14 KB of top-level `description`, and **31.7 KB — 35% of
the entire catalog — inside `inputSchema.properties[].description`**. The second
number is the bigger one, so `trim_schema_prose` gives that nested prose the same
treatment: the field keeps a short hint, the full sentence moves on-demand.

The authoritative half of a schema still goes out byte-identical — `type`,
`enum`, `required`, `format`, field names, nothing renamed or retyped
(`test_schema_types_are_never_touched` pins exactly that). Prose is not
authoritative. A field carrying an `enum` loses its description outright: the
permitted values already state what the field accepts, so a sentence restating
them is duplication paid on every session. The trimmed prose isn't deleted: `apply_result_appendix` splices
it into the *call result* of the matching `get_<domain>_instructions` tool —
not into that tool's tools/list entry, which must stay short, since a
tool's description is always-on regardless of whether it's ever called.
Only the RESULT of calling an instructions tool is genuinely on-demand (see
the server's own "progressive disclosure" note in its `initialize`
instructions) — this applies that same on-demand principle one level
further, to the search tools' edge-case prose.

Only tools with a measured, disproportionate description size get an
override here; everything else passes through unchanged from upstream. In
particular `create_checkout_link` is deliberately NOT compacted yet — it is
the purchase-dispatch tool (wrong fields there can mean the wrong party or
wrong fare gets charged), and safely trimming it needs eval coverage this
scaffold doesn't have yet.
"""

import hashlib
import logging
from copy import deepcopy
from typing import Any

log = logging.getLogger(__name__)

COMPACT_DESCRIPTIONS: dict[str, str] = {
    "search_rail": (
        "Search Russian Railways (РЖД) tickets between two cities. Each offer carries `price`, "
        "a `fares` summary (`{count, price_from, price_to, refundable_count, seat_categories, "
        "uncategorized_fares?}` — a count is a LOWER bound: `refundable_count: 0` means 'none "
        "confirmed refundable', not 'no refundable fare exists', and a `seat_categories` entry "
        "missing for сидячий/плацкарт/купе/СВ means not on sale ONLY when `uncategorized_fares` is "
        "absent), `legs[].segments[]` with carrier + train number + `vehicle_meta`, "
        "`search_results_url`, `checkout_url`, `checkout_ref`, and `details_ref` "
        "(→ `get_offer_details` for the full per-class ladder, → `get_rail_seatmap` for exact seats). "
        "Paginated: a page is a WINDOW over matched trains (`meta.total_matched`, `meta.has_more`) — "
        "never say a numbered train doesn't run from one page while `has_more` is true; pass "
        "`train_numbers` instead of paging blind. An empty `offers` with `meta.interchange_routes` "
        "populated means no DIRECT train that day; transfer plans are shown instead. Full field "
        "semantics, filter-interaction caveats and grounding rules: `get_rail_instructions`."
    ),
    "get_rail_seatmap": (
        "Read-only per-car seat layout for a rail offer. Authoritative for any seat/car/gender/WC-"
        "distance question — if the offer has `details_ref`, CALL this rather than guessing; check "
        "`seatmap_status` (`ok` vs the rare `no_layout_for_carrier`). Prefer `task=` "
        "(`far_from_wc` / `female` / `together` / `summary`) for a short ranked answer over the "
        "whole train instead of paging every seat. Seat `type` is open vocabulary — match by PREFIX "
        "(`LOWER*`, `SIDE_LOWER*`), not equality. Join fares via `group_index` → "
        "`cars[].seat_groups[].fares[]`, not via `seat.type`. Seatmap prices are pre-cart and run "
        "below the final total (checkout adds ~6-8%) — quote the `search_rail` listing price as the "
        "bookable total. No window/aisle data exists in any car type. Gender: only "
        '`gender="FEMALE"`/`"MALE"` are proof of a gendered seat. Full pagination/`task=` '
        "reference and checkout handoff: `get_rail_instructions`."
    ),
    "search_hotels": (
        "Search Tutu hotel listings for a city and date range (`city_name` or `geo_id`). Each hotel "
        "carries `name`, `stars`, `rating`, `review_summary`, and `best_offer` — the cheapest rate of "
        "ONE room only (`price`, `offerpack_hash`, `checkout_url`); compare room categories via "
        "`get_offer_details`. `best_offer.price` is the TOTAL for the whole stay+guests "
        '(`price_basis="stay_total"`) — render as-is, never multiply by `stay.nights`. Grounding: '
        "ratings/amenities/quotes must come from `review_summary` / `get_offer_details`, never "
        "invented or substituted with generic praise. Before searching a broad request, ask 2-4 short "
        "clarifying questions (bed, breakfast, free-cancel, view) UNLESS the user already gave a "
        "deterministic pick rule ('дешевле', 'без уточнений') or explicit checkout intent — then "
        "search now and hand over the link in the same turn. Checkout: `best_offer.checkout_url` "
        "directly, or `create_checkout_link` with the row's `checkout_ref`; only a ROOM-level "
        "`offerpack_hash` (from `get_offer_details`) mints a cart, the listing one doesn't. Full "
        "field semantics: `get_hotels_instructions`."
    ),
}

# Source tool -> the instructions tool whose CALL RESULT absorbs its trimmed prose
# (never that tool's tools/list description, which must stay short). Governs both
# the top-level description and the per-field schema prose.
#
# `get_offer_details` and `create_checkout_link` are absent on purpose: neither has
# a paired instructions tool, so their prose has nowhere on-demand to go — and
# `create_checkout_link` is the purchase dispatcher, which we do not touch at all.
APPENDIX_TARGETS: dict[str, str] = {
    "search_rail": "get_rail_instructions",
    "get_rail_seatmap": "get_rail_instructions",
    "search_hotels": "get_hotels_instructions",
    "search_avia": "get_avia_instructions",
    "search_bus": "get_bus_instructions",
    "search_etrain": "get_etrain_instructions",
    "search_multitransport": "get_multitransport_instructions",
}

# sha1 of the upstream description each override in `COMPACT_DESCRIPTIONS` was
# written against, as recorded in `fixtures/_meta/tools_list.json`. An override is
# a hand-written SUMMARY of a specific text: once Tutu rewrites that text, the
# summary may be describing behavior the tool no longer has, and shipping it would
# be worse than shipping the long original. `test_compact_tools.py` recomputes
# these from the fixture, so updating the catalog forces a deliberate re-read
# rather than a silent mismatch.
SOURCE_DIGESTS: dict[str, str] = {
    "search_rail": "b0700e946958",
    "get_rail_seatmap": "45901d053cbc",
    "search_hotels": "6565067fe5c2",
}


def description_digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def compactable(tools: list[dict[str, Any]]) -> set[str]:
    """Which tools may be compacted against THIS catalog.

    Two ways a tool drops out, both of which used to pass unnoticed:

    * its appendix target is gone (Tutu renamed or dropped
      `get_<domain>_instructions`) — the trimmed prose would have nowhere
      on-demand to go, so `apply_result_appendix` would splice it into nothing
      and several KB of meaning would simply disappear, with no error and no log;
    * upstream rewrote the description our override summarizes, so the override
      now describes a tool that may have changed underneath it.

    In both cases the tool passes through exactly as upstream sent it. That costs
    bytes and keeps the catalog honest, which is the right way round: the whole
    claim of this proxy is that nothing meaningful is lost.
    """
    present = {tool["name"] for tool in tools}
    allowed = set()
    for tool in tools:
        name = tool["name"]
        if name not in APPENDIX_TARGETS:
            continue
        target = APPENDIX_TARGETS[name]
        if target not in present:
            log.warning(
                "compaction skipped for %s: appendix target %s is not in this catalog", name, target
            )
            continue
        expected = SOURCE_DIGESTS.get(name)
        actual = description_digest(tool.get("description", ""))
        if expected is not None and expected != actual:
            log.warning(
                "compaction skipped for %s: upstream description changed (%s != %s)",
                name,
                actual,
                expected,
            )
            continue
        allowed.add(name)
    return allowed


# Enough to say what the field is; the sentence that says how it behaves moves to
# the instructions tool. Cut on a word boundary — a hint truncated mid-word reads
# as corruption and invites the agent to guess at the rest.
SCHEMA_HINT_CHARS = 80


def _has_enum(field: dict[str, Any]) -> bool:
    items = field.get("items")
    return "enum" in field or (isinstance(items, dict) and "enum" in items)


def _shorten(text: str, limit: int = SCHEMA_HINT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = head.rfind(" ")
    return (head[:cut] if cut > limit // 2 else head).rstrip(" ,.;:") + "…"


def trim_schema_prose(
    tools: list[dict[str, Any]], allowed: set[str] | None = None
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Shorten `properties[].description` on the tools listed in
    `APPENDIX_TARGETS`, returning `(tools, removed)` where `removed` maps
    tool -> field -> the full original sentence.

    `allowed` narrows that to the tools this catalog can still absorb the prose
    for (see `compactable`); omitted, every appendix target is trimmed, which is
    what the schema-level tests want to exercise.

    Only prose moves. Types, enums, `required` and field names are untouched.
    """
    tools = deepcopy(tools)
    removed: dict[str, dict[str, str]] = {}
    for tool in tools:
        if tool["name"] not in APPENDIX_TARGETS:
            continue
        if allowed is not None and tool["name"] not in allowed:
            continue
        properties = (tool.get("inputSchema") or {}).get("properties") or {}
        for name, field in properties.items():
            if not isinstance(field, dict):
                continue
            full = field.get("description")
            if not full:
                continue
            short = "" if _has_enum(field) else _shorten(full)
            if short == full:
                continue
            removed.setdefault(tool["name"], {})[name] = full
            if short:
                field["description"] = short
            else:
                del field["description"]
    return tools, removed


def apply_compact_overrides(
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return `(compacted_tools, trimmed_originals)`. `inputSchema` and every other
    tool's `description` (including instructions tools) are left untouched.
    `trimmed_originals` holds the pre-trim description of each overridden tool, keyed
    by its name — feed it to `apply_result_appendix` so the removed prose still
    reaches the agent when it actually calls the matching instructions tool.

    A tool `compactable` rules out passes through byte-identical to upstream: the
    proxy gets bigger, and nothing the agent needs goes missing."""
    tools = deepcopy(tools)
    allowed = compactable(tools)
    trimmed_originals: dict[str, str] = {}
    for tool in tools:
        override = COMPACT_DESCRIPTIONS.get(tool["name"])
        if override is not None and tool["name"] in allowed:
            trimmed_originals[tool["name"]] = tool["description"]
            tool["description"] = override

    tools, schema_prose = trim_schema_prose(tools, allowed)
    for name, fields in schema_prose.items():
        block = "\n".join(f"- `{field}` — {text}" for field, text in fields.items())
        existing = trimmed_originals.get(name, "")
        trimmed_originals[name] = (
            f"{existing}\n\n### Field reference (trimmed from inputSchema)\n{block}"
            if existing
            else f"### Field reference (trimmed from inputSchema)\n{block}"
        )
    return tools, trimmed_originals


def apply_result_appendix(
    tool_name: str, result_text: str, trimmed_originals: dict[str, str]
) -> str:
    """If `tool_name` is an appendix target, append the prose trimmed off its source
    tool(s)' tools/list description (`trimmed_originals`, from `apply_compact_overrides`).
    No-op for every other tool."""
    sources = [name for name, target in APPENDIX_TARGETS.items() if target == tool_name]
    blocks = [
        f"## Full reference for `{source}` (trimmed from tools/list)\n{trimmed_originals[source]}"
        for source in sources
        if source in trimmed_originals
    ]
    if not blocks:
        return result_text
    return f"{result_text}\n\n" + "\n\n".join(blocks)
