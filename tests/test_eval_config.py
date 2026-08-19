"""`OPENAI_EFFORT` has to behave like `OPENAI_MODEL`: a .env default the CLI flag
overrides, validated before a run starts spending tokens."""

import pytest

from evals.config import InvalidEffortError, openai_effort_default
from evals.options import AgentKind, Api, Effort, EvalOptions
from evals.run import build_agent


def test_unset_effort_leaves_the_choice_to_the_model(monkeypatch):
    monkeypatch.delenv("OPENAI_EFFORT", raising=False)
    assert openai_effort_default() is None


@pytest.mark.parametrize(
    "raw, expected",
    [("minimal", Effort.MINIMAL), (" HIGH ", Effort.HIGH), ("xhigh", Effort.XHIGH)],
)
def test_effort_is_read_case_and_space_insensitively(monkeypatch, raw, expected):
    monkeypatch.setenv("OPENAI_EFFORT", raw)
    assert openai_effort_default() is expected


def test_blank_effort_reads_as_unset(monkeypatch):
    monkeypatch.setenv("OPENAI_EFFORT", "   ")
    assert openai_effort_default() is None


def test_unusable_effort_is_rejected_with_the_allowed_values(monkeypatch):
    monkeypatch.setenv("OPENAI_EFFORT", "maximal")
    with pytest.raises(InvalidEffortError, match="minimal"):
        openai_effort_default()


@pytest.mark.parametrize("api", [Api.RESPONSES, Api.CHAT])
def test_effort_reaches_the_agent_on_both_endpoints(api):
    opts = EvalOptions(agent=AgentKind.OPENAI, api=api)
    agent = build_agent(opts, "gpt-5", Effort.MINIMAL)
    assert agent.effort == "minimal"
    assert agent.label == "gpt-5/minimal"


def test_no_effort_means_no_reasoning_field():
    agent = build_agent(EvalOptions(agent=AgentKind.OPENAI), "gpt-5", None)
    assert agent.effort is None
    assert "reasoning" not in agent._request_kwargs()
    assert agent.label == "gpt-5"


def test_the_whole_effort_scale_is_accepted(monkeypatch):
    """Rungs differ by model — gpt-5 has `minimal`, gpt-5.6 has `xhigh`/`max` — so
    the harness carries the union and lets the API reject what its model lacks."""
    for effort in Effort:
        monkeypatch.setenv("OPENAI_EFFORT", effort.value)
        assert openai_effort_default() is effort
