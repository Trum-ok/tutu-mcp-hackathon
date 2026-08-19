"""Shared record of one agent run — what it called, what came back, what it answered.

Every downstream consumer (checks, metrics, report, the future trace viewer) reads
this and nothing else, so the agent implementation stays swappable.
"""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result_text: str
    is_error: bool
    # True when the mock backend had no fixture for these arguments. Distinct from
    # `is_error`: it means OUR eval set is incomplete, not that Tutu returned an error.
    # Conflating the two would let harness gaps read as upstream failures.
    fixture_miss: bool = False
    duration_s: float = 0.0

    def parsed_result(self) -> Any | None:
        try:
            return json.loads(self.result_text)
        except (json.JSONDecodeError, TypeError):
            return None


@dataclass
class Transcript:
    scenario_id: str
    variant: str
    answer_text: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0
    # set when the run could not complete: refusal, API error, turn-cap hit
    failure: str | None = None

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def tool_error_count(self) -> int:
        return sum(1 for c in self.tool_calls if c.is_error and not c.fixture_miss)

    @property
    def fixture_miss_count(self) -> int:
        return sum(1 for c in self.tool_calls if c.fixture_miss)

    def tool_names(self) -> list[str]:
        return [c.name for c in self.tool_calls]

    def result_payloads(self) -> list[Any]:
        """Every successfully parsed tool_result payload — the grounding evidence set."""
        payloads = []
        for call in self.tool_calls:
            parsed = call.parsed_result()
            if parsed is not None:
                payloads.append(parsed)
        return payloads
