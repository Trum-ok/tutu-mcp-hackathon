"""Which endpoint's serialization the surface is measured against.

The two endpoints nest tool definitions differently, so this choice decides the
headline number. It used to be a string compared with `== "responses"`, where
every value that was not that exact literal fell through to the Chat shape —
including a typo, which would have produced a plausible wrong number in silence.
"""

import pytest

from evals.options import Api
from evals.tokens import render_tools, surface_bytes

TOOLS = [{"name": "search_rail", "description": "x", "inputSchema": {"type": "object"}}]


def test_the_two_endpoints_render_differently():
    assert render_tools(TOOLS, Api.RESPONSES) != render_tools(TOOLS, Api.CHAT)


def test_a_string_is_accepted_and_means_the_same_thing():
    """`Api` is a StrEnum, so a plain string reaching this from older code must not
    quietly select the other endpoint."""
    assert render_tools(TOOLS, "responses") == render_tools(TOOLS, Api.RESPONSES)
    assert render_tools(TOOLS, "chat") == render_tools(TOOLS, Api.CHAT)


def test_an_unknown_endpoint_fails_loudly():
    with pytest.raises(ValueError):
        render_tools(TOOLS, "responsez")
    with pytest.raises(ValueError):
        surface_bytes(TOOLS, "system", "responsez")
