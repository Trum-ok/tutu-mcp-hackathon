import json
from copy import deepcopy

from tutu_mcp.proxy.compact_tools import (
    APPENDIX_TARGETS,
    SCHEMA_HINT_CHARS,
    SOURCE_DIGESTS,
    apply_compact_overrides,
    apply_result_appendix,
    description_digest,
)

from .conftest import REPO_FIXTURES_DIR

RAW_TOOLS = json.loads(
    (REPO_FIXTURES_DIR / "_meta" / "tools_list.json").read_text(encoding="utf-8")
)


def test_compaction_meaningfully_shrinks_the_always_on_catalog():
    compacted, _ = apply_compact_overrides(RAW_TOOLS)

    raw_size = len(json.dumps(RAW_TOOLS, ensure_ascii=False))
    compacted_size = len(json.dumps(compacted, ensure_ascii=False))

    assert compacted_size < raw_size * 0.9, (
        f"expected a real reduction, got {raw_size} -> {compacted_size} bytes"
    )


def test_instructions_tools_stay_short_in_tools_list():
    compacted, _ = apply_compact_overrides(RAW_TOOLS)
    by_name = {t["name"]: t for t in compacted}

    for target in set(APPENDIX_TARGETS.values()):
        raw_len = len(next(t["description"] for t in RAW_TOOLS if t["name"] == target))
        compacted_len = len(by_name[target]["description"])
        assert compacted_len == raw_len, (
            f"{target}'s tools/list description must stay untouched, not carry the appendix"
        )


def _without_prose(node):
    """A schema stripped of every `description`, i.e. only its authoritative half."""
    if isinstance(node, dict):
        return {k: _without_prose(v) for k, v in node.items() if k != "description"}
    if isinstance(node, list):
        return [_without_prose(v) for v in node]
    return node


def test_schema_types_are_never_touched():
    """The line this module must not cross.

    Prose inside a schema is fair game; `type`, `enum`, `required`, `format` and
    field names are not — an agent fills fields from those, and a proxy that
    quietly retyped one would corrupt calls rather than shrink them.
    """
    compacted, _ = apply_compact_overrides(RAW_TOOLS)
    raw_by_name = {t["name"]: t.get("inputSchema") for t in RAW_TOOLS}

    for tool in compacted:
        assert _without_prose(tool.get("inputSchema")) == _without_prose(
            raw_by_name[tool["name"]]
        ), f"{tool['name']}: изменена авторитетная часть схемы, а не только проза"


def test_untargeted_tools_keep_their_schema_prose_verbatim():
    """`create_checkout_link` is the purchase dispatcher and stays fully untouched."""
    compacted, _ = apply_compact_overrides(RAW_TOOLS)
    raw_by_name = {t["name"]: t for t in RAW_TOOLS}

    for name in ("create_checkout_link", "get_offer_details", "fetch_resource"):
        assert next(t for t in compacted if t["name"] == name) == raw_by_name[name]


def test_enum_fields_lose_their_description_entirely():
    """The permitted values already say what the field accepts."""
    compacted, _ = apply_compact_overrides(RAW_TOOLS)
    trimmed = {t["name"]: t for t in compacted}

    checked = 0
    for name in APPENDIX_TARGETS:
        for field in (trimmed[name].get("inputSchema") or {}).get("properties", {}).values():
            if isinstance(field, dict) and "enum" in field:
                assert "description" not in field, f"{name}: enum-поле сохранило прозу"
                checked += 1
    assert checked, "в выборке не оказалось ни одного enum-поля — тест ничего не проверил"


def test_remaining_field_hints_are_short_and_not_cut_mid_word():
    compacted, _ = apply_compact_overrides(RAW_TOOLS)

    for tool in compacted:
        if tool["name"] not in APPENDIX_TARGETS:
            continue
        for field in (tool.get("inputSchema") or {}).get("properties", {}).values():
            hint = field.get("description") if isinstance(field, dict) else None
            if not hint:
                continue
            assert len(hint) <= SCHEMA_HINT_CHARS + 1  # +1 for the ellipsis
            if hint.endswith("…"):
                assert not hint[:-1].endswith(" ")


def test_schema_prose_is_not_lost_only_moved():
    _, trimmed = apply_compact_overrides(RAW_TOOLS)
    raw_by_name = {t["name"]: t for t in RAW_TOOLS}

    appendix = apply_result_appendix("get_rail_instructions", "RESULT", trimmed)
    original_fields = (raw_by_name["search_rail"]["inputSchema"]).get("properties", {})
    long_prose = [
        d
        for f in original_fields.values()
        if isinstance(f, dict) and len(d := f.get("description", "")) > SCHEMA_HINT_CHARS
    ]

    assert long_prose, "фикстура изменилась: у search_rail не осталось длинных описаний полей"
    for text in long_prose:
        assert text in appendix, "проза выброшена, а не перенесена в результат instructions-тула"


def test_trimmed_prose_reaches_the_instructions_tool_call_result():
    _, trimmed_originals = apply_compact_overrides(RAW_TOOLS)

    for source, target in APPENDIX_TARGETS.items():
        result = apply_result_appendix(target, "ORIGINAL_RESULT_TEXT", trimmed_originals)
        assert "ORIGINAL_RESULT_TEXT" in result
        assert trimmed_originals[source] in result


def test_appendix_is_a_no_op_for_unrelated_tools():
    _, trimmed_originals = apply_compact_overrides(RAW_TOOLS)

    result = apply_result_appendix("create_checkout_link", "UNRELATED_RESULT", trimmed_originals)

    assert result == "UNRELATED_RESULT"


def _renamed(old: str, new: str) -> list[dict]:
    tools = deepcopy(RAW_TOOLS)
    for tool in tools:
        if tool["name"] == old:
            tool["name"] = new
    return tools


def test_source_digests_still_match_the_recorded_catalog():
    """Each override summarizes a specific upstream text. When the fixture is
    re-recorded this test fails, which is the point: someone has to re-read the
    new description and decide the summary still holds, instead of the mismatch
    passing unnoticed."""
    for tool in RAW_TOOLS:
        expected = SOURCE_DIGESTS.get(tool["name"])
        if expected is not None:
            assert description_digest(tool["description"]) == expected, (
                f"{tool['name']}'s upstream description changed — re-read it, then "
                f"update COMPACT_DESCRIPTIONS and SOURCE_DIGESTS together"
            )


def test_a_renamed_instructions_tool_stops_compaction_instead_of_losing_the_prose():
    """The failure this guards against was silent: with the appendix target gone,
    `apply_result_appendix` had nowhere to splice the trimmed prose, so ~9 KB of
    `search_rail` semantics vanished from both tools/list and every call result."""
    tools = _renamed("get_rail_instructions", "get_rail_playbook")

    compacted, trimmed = apply_compact_overrides(tools)
    by_name = {t["name"]: t for t in compacted}
    original = {t["name"]: t for t in tools}

    assert by_name["search_rail"]["description"] == original["search_rail"]["description"]
    assert by_name["get_rail_seatmap"]["description"] == original["get_rail_seatmap"]["description"]
    assert "search_rail" not in trimmed
    # The unaffected domain keeps its saving: the degradation is per-tool, not global.
    assert len(by_name["search_hotels"]["description"]) < len(
        original["search_hotels"]["description"]
    )


def test_a_rewritten_upstream_description_is_passed_through_untouched():
    """An override is a hand-written summary of a text we read once. Once that text
    changes, the summary may describe behavior the tool no longer has."""
    tools = deepcopy(RAW_TOOLS)
    for tool in tools:
        if tool["name"] == "search_hotels":
            tool["description"] += " Новое поведение, которого мы не читали."

    compacted, trimmed = apply_compact_overrides(tools)
    by_name = {t["name"]: t for t in compacted}

    assert by_name["search_hotels"]["description"].endswith("которого мы не читали.")
    assert "search_hotels" not in trimmed
    assert len(by_name["search_rail"]["description"]) < len(
        next(t["description"] for t in RAW_TOOLS if t["name"] == "search_rail")
    )
