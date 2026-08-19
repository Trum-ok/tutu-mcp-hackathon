"""The two OpenAI loops, driven by a stub client through `client_factory`.

Every branch here is the MODEL misbehaving — refusing, running out of output
budget, emitting unparsable arguments, never stopping. No fixture can produce
them, and against the real API they are exactly the cases that must end as a
recorded `failure` on the transcript rather than as an exception that takes the
whole run down with it.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from evals.agent import ChatCompletionsAgent, ResponsesAgent
from evals.transcript import ToolCallRecord

CALLS: list[tuple[str, dict[str, Any]]] = []


async def _execute(name: str, arguments: dict[str, Any]) -> ToolCallRecord:
    CALLS.append((name, arguments))
    return ToolCallRecord(name=name, arguments=arguments, result_text="{}", is_error=False)


@pytest.fixture(autouse=True)
def _clear_calls():
    CALLS.clear()


class Dumpable(SimpleNamespace):
    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not exclude_none or v is not None}


def _tool_call(name: str, arguments: str) -> Dumpable:
    return Dumpable(
        id="call_1", type="function", function=SimpleNamespace(name=name, arguments=arguments)
    )


class StubChat:
    """`client.chat.completions.create` replaying a fixed list of responses."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.requests = 0

    async def create(self, **kwargs: Any) -> Any:
        self.requests += 1
        # a turn cap test needs more turns than scripted responses
        return self._responses[min(self.requests - 1, len(self._responses) - 1)]

    def client(self) -> Any:
        return SimpleNamespace(chat=SimpleNamespace(completions=self))


def _chat_response(message: Dumpable, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
    )


async def _run(agent: Any) -> Any:
    return await agent.run(
        scenario_id="s",
        variant="proxy",
        request="Поезда из Питера в Мск",
        tools=[],
        server_instructions="",
        execute=_execute,
    )


async def test_a_refusal_ends_the_run_as_a_recorded_failure():
    stub = StubChat([_chat_response(Dumpable(refusal="не могу", content=None, tool_calls=None))])
    transcript = await _run(ChatCompletionsAgent(client_factory=stub.client))

    assert transcript.failure == "refusal:не могу"
    assert transcript.answer_text == ""


async def test_hitting_the_output_cap_is_reported_next_to_the_partial_answer():
    """The text is kept: a truncated answer is still evidence about the surface,
    as long as nothing downstream mistakes it for a completed one."""
    stub = StubChat(
        [_chat_response(Dumpable(refusal=None, content="Поезд ", tool_calls=None), "length")]
    )
    transcript = await _run(ChatCompletionsAgent(client_factory=stub.client))

    assert transcript.failure == "max_completion_tokens"
    assert transcript.answer_text == "Поезд "


async def test_unparsable_tool_arguments_call_the_tool_with_an_empty_dict():
    """Arguments arrive as a JSON *string* the model wrote. Broken JSON must reach
    the tool as `{}` — which the tool then rejects on its own terms — instead of
    raising inside the loop."""
    calling = _chat_response(
        Dumpable(refusal=None, content=None, tool_calls=[_tool_call("search_rail", "{not json")])
    )
    answering = _chat_response(Dumpable(refusal=None, content="готово", tool_calls=None))
    stub = StubChat([calling, answering])

    transcript = await _run(ChatCompletionsAgent(client_factory=stub.client))

    assert CALLS == [("search_rail", {})]
    assert transcript.failure is None
    assert transcript.answer_text == "готово"


async def test_a_model_that_never_stops_calling_tools_hits_the_turn_cap():
    forever = _chat_response(
        Dumpable(refusal=None, content=None, tool_calls=[_tool_call("search_rail", "{}")])
    )
    stub = StubChat([forever])

    transcript = await _run(ChatCompletionsAgent(client_factory=stub.client, max_turns=3))

    assert transcript.failure == "turn_cap:3"
    assert stub.requests == 3
    assert transcript.turns == 3


async def test_an_api_error_becomes_the_transcript_failure_not_an_exception():
    """One bad scenario must not kill the other scenarios in the run."""

    class Exploding:
        async def create(self, **kwargs: Any) -> Any:
            raise RuntimeError("connection reset")

        def client(self) -> Any:
            return SimpleNamespace(chat=SimpleNamespace(completions=self))

    transcript = await _run(ChatCompletionsAgent(client_factory=Exploding().client))

    assert transcript.failure == "RuntimeError: connection reset"
    assert transcript.duration_s >= 0


# --- Responses API loop -------------------------------------------------------


class StubResponses:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.requests = 0

    async def create(self, **kwargs: Any) -> Any:
        self.requests += 1
        return self._responses[min(self.requests - 1, len(self._responses) - 1)]

    def client(self) -> Any:
        return SimpleNamespace(responses=self)


def _responses_reply(
    output: list[Any], output_text: str = "", status: str = "completed", reason: str = ""
) -> SimpleNamespace:
    return SimpleNamespace(
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        output=output,
        output_text=output_text,
        status=status,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
    )


def _function_call(name: str, arguments: str) -> Dumpable:
    return Dumpable(
        type="function_call", name=name, arguments=arguments, call_id="fc_1", id="item_1"
    )


async def test_an_incomplete_response_names_the_reason():
    stub = StubResponses(
        [_responses_reply([], "частичный ответ", status="incomplete", reason="max_output_tokens")]
    )
    transcript = await _run(ResponsesAgent(client_factory=stub.client))

    assert transcript.failure == "incomplete:max_output_tokens"
    assert transcript.answer_text == "частичный ответ"


async def test_the_responses_loop_also_survives_unparsable_arguments():
    calling = _responses_reply([_function_call("search_rail", "")])
    answering = _responses_reply([], "готово")
    stub = StubResponses([calling, answering])

    transcript = await _run(ResponsesAgent(client_factory=stub.client))

    assert CALLS == [("search_rail", {})]
    assert transcript.answer_text == "готово"
    assert transcript.tool_names() == ["search_rail"]


async def test_the_responses_loop_has_the_same_turn_cap():
    stub = StubResponses([_responses_reply([_function_call("search_rail", "{}")])])

    transcript = await _run(ResponsesAgent(client_factory=stub.client, max_turns=2))

    assert transcript.failure == "turn_cap:2"
    assert transcript.input_tokens == 20
    assert transcript.output_tokens == 10
