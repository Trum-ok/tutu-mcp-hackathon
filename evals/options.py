"""What one eval run is asked to do: the enums the CLI exposes, the variant
names, and the options record that carries them into `evals.run`.

Deliberately a leaf — it imports nothing but the stdlib. `tutu.py` has to name
these types at module level (Typer reads them off the command signature at
import time), so anything reachable from here is loaded on *every* invocation,
`--help` included. Keeping it dependency-free is what lets the commands import
their real work lazily: `tutu.py serve` never pays for the eval harness, and
`tutu.py viewer` never pays for `mcp`.

Put anything that needs `evals.agent`, `evals.variants` or `tutu_mcp` in
`evals/run.py` instead — a single import here quietly costs ~0.4 s on every
command.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# The two tool surfaces being compared. They live here rather than in
# `evals/variants.py` (which re-exports them) only because `EvalOptions` needs
# them for its default and that module pulls in the whole proxy.
BASELINE = "baseline"
PROXY = "proxy"


class AgentKind(StrEnum):
    OPENAI = "openai"
    SCRIPTED = "scripted"


class Api(StrEnum):
    RESPONSES = "responses"
    CHAT = "chat"


class Effort(StrEnum):
    """Every reasoning effort any current model accepts — the union, not an
    intersection: the scale differs BY MODEL and only the API knows which rung a
    given one has. gpt-5 takes `minimal` and has no `xhigh`/`max`; gpt-5.6 dropped
    `minimal` and added both. So an unsupported value is rejected by the API, not
    here, and its 400 names the supported set for that model — a better answer
    than a hardcoded table that goes stale on the next release."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


@dataclass(frozen=True)
class EvalOptions:
    """Everything one run needs. Defaults match what the CLI advertises."""

    agent: AgentKind = AgentKind.OPENAI
    model: str | None = None
    effort: Effort | None = None
    api: Api = Api.RESPONSES
    live: bool = False
    record_missing: bool = False
    variants: tuple[str, ...] = (BASELINE, PROXY)
    scenarios: tuple[str, ...] | None = None
    domains: tuple[str, ...] | None = None
    concurrency: int = 1
    out: Path = Path("out/eval-results.json")
    estimate_tokens: bool = False
