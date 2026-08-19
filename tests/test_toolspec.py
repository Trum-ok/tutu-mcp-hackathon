"""Our own two tools are declared from Pydantic models — these pin what that must
produce: a flat schema every client can read, and argument errors an agent can act on.
"""

import json

from pydantic import BaseModel, Field

from tutu_mcp.groundedness import CHECK_GROUNDEDNESS_TOOL, run_check_groundedness_tool
from tutu_mcp.premises import ASSESS_REQUEST_TOOL, SessionPremises, run_assess_request_tool
from tutu_mcp.toolspec import parse_args, tool_schema

OUR_TOOLS = [ASSESS_REQUEST_TOOL, CHECK_GROUNDEDNESS_TOOL]


class Nested(BaseModel):
    tool: str


class Outer(BaseModel):
    name: str = Field(description="a name")
    items: list[Nested] = Field(default_factory=list)


def test_nested_models_are_inlined_not_referenced():
    """A `$ref` is valid JSON Schema but crosses into whichever client the user
    runs; a flat object is the shape all of them handle."""
    schema = json.dumps(tool_schema(Outer), ensure_ascii=False)

    assert "$ref" not in schema
    assert "$defs" not in schema
    assert tool_schema(Outer)["properties"]["items"]["items"]["properties"]["tool"]


def test_titles_are_dropped():
    """Pydantic titles restate the key name — dead weight in an always-on catalog."""
    assert "title" not in json.dumps(tool_schema(Outer), ensure_ascii=False)


def test_field_descriptions_survive():
    assert tool_schema(Outer)["properties"]["name"]["description"] == "a name"


def test_required_is_derived_from_the_model():
    assert tool_schema(Outer)["required"] == ["name"]


def test_our_tools_match_the_upstream_descriptor_shape():
    for tool in OUR_TOOLS:
        assert set(tool) >= {"name", "description", "inputSchema"}
        assert tool["inputSchema"]["type"] == "object"
        assert "title" not in json.dumps(tool, ensure_ascii=False)


def test_bad_arguments_name_the_offending_field():
    message = parse_args(Outer, {"items": "не список"})

    assert isinstance(message, str)
    assert "name" in message and "items" in message


def test_assess_request_rejects_a_malformed_plan_with_a_readable_error():
    text, is_error = run_assess_request_tool(
        {"user_request": "поезда в Питер", "planned_calls": [{"arguments": {}}]},
        SessionPremises(),
    )

    assert is_error
    assert "tool" in text


def test_check_groundedness_still_separates_bad_json_from_a_bad_shape():
    """`list[str]` of non-JSON passes validation but cannot be parsed — the two
    failures must read differently, or a debugging agent chases the wrong one."""
    shape_error, _ = run_check_groundedness_tool({"answer_text": "x"})
    json_error, _ = run_check_groundedness_tool(
        {"answer_text": "x", "tool_result_json": ["{не json}"]}
    )

    assert "tool_result_json" in shape_error
    assert "не-JSON" in json_error


def test_a_declared_price_assumption_is_amber_not_red():
    """A price arrives as float and was recorded as int, so "4000.0" never matched
    "4000" — an openly declared price read as an invention. The trace viewer paints
    those two differently, so the distinction has to survive the round trip.
    """
    from tutu_mcp.groundedness import check_groundedness

    report = check_groundedness(
        "Ориентировочно 4000 ₽",
        [{"offers": []}],
        assumed_values={"4000"},
        assumptions=["price_max = 4000 (типовой бюджет)"],
    )

    assert [c.status for c in report.checks] == ["assumed"]
    assert report.rate == 0.0, "допущение не должно засчитываться как подтверждённое"
