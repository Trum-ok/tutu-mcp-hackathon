"""Single entrypoint for everything in this repo.

    uv run python tutu.py --help

Lives at the repo root rather than inside a package, and that placement is the
point: it needs to reach both `tutu_mcp` (the proxy) and `evals` (the harness
that measures it). Putting it in `tutu_mcp/` would make the proxy import its own
harness — the exact dependency direction `evals/config.py` exists to avoid.

    tutu.py ──> tutu_mcp/       (serve, record)
            ├─> evals/          (evals, demo) ──> tutu_mcp/
            └─> evals/options.py  — the only import taken eagerly, see below
"""

import re
import sys
from pathlib import Path

import anyio
import typer

from evals.options import AgentKind, Api, Effort, EvalOptions

# Every command imports its own work inside the function body. That is not
# style: `evals.variants` and `tutu_mcp.main` both pull in `mcp`, which costs
# ~0.4 s, and Typer imports this module for `--help` alone. The types above are
# the one thing that has to be here — Typer reads them off the signatures — and
# `evals/options.py` exists so that they cost nothing.

app = typer.Typer(
    help="Compacting MCP proxy in front of mcp.tutu.ru, plus the eval harness that measures it.",
    no_args_is_help=True,
    add_completion=False,
)

# argparse took these as `nargs="+"` (space-separated). Typer's native list form
# is a repeated flag, which would have rewritten every example in the README, so
# one string is parsed here instead — commas or spaces, both work.
_SEPARATORS = re.compile(r"[,\s]+")


def _split(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    parts = tuple(p for p in _SEPARATORS.split(value.strip()) if p)
    return parts or None


@app.command()
def serve() -> None:
    """Run the proxy. TUTU_PROXY_MODE=mock (default) serves fixtures; live proxies Tutu."""
    from tutu_mcp.main import serve as _serve

    anyio.run(_serve)


@app.command()
def evals(
    agent: AgentKind = AgentKind.OPENAI,
    model: str | None = typer.Option(None, help="default: $OPENAI_MODEL, else gpt-5"),
    effort: Effort | None = typer.Option(
        None,
        help="reasoning effort; default: $OPENAI_EFFORT, else the model's own default",
    ),
    api: Api = typer.Option(
        Api.RESPONSES,
        help=(
            "OpenAI endpoint. 'responses' (default) is the only one that takes "
            "function tools together with reasoning; 'chat' is for OpenAI-compatible "
            "gateways that do not implement /v1/responses"
        ),
    ),
    live: bool = typer.Option(False, "--live", help="call the real mcp.tutu.ru"),
    record_missing: bool = typer.Option(
        False,
        "--record-missing",
        help="with --live: record any fixture the agent asks for and we don't have",
    ),
    variants: str = typer.Option("baseline,proxy", help="tool surfaces to compare"),
    scenarios: str | None = typer.Option(None, help="scenario ids to run"),
    domains: str | None = typer.Option(None, help="restrict to these domains"),
    concurrency: int = 1,
    out: Path = Path("out/eval-results.json"),
    estimate_tokens: bool = typer.Option(
        False,
        "--estimate-tokens",
        help="use the offline tiktoken estimate instead of one real probe request",
    ),
    list_models: bool = typer.Option(
        False, "--list-models", help="print the models this key can reach, then exit"
    ),
) -> None:
    """Baseline vs proxy on recorded fixtures (or --live against the real server)."""
    from evals.config import MISSING_CREDENTIALS_HELP, openai_credentials_source
    from evals.run import list_models as _list_models
    from evals.run import run_evals

    if list_models:
        if openai_credentials_source() is None:
            print(MISSING_CREDENTIALS_HELP, file=sys.stderr)
            raise typer.Exit(2)
        raise typer.Exit(anyio.run(_list_models))

    opts = EvalOptions(
        agent=agent,
        model=model,
        effort=effort,
        api=api,
        live=live,
        record_missing=record_missing,
        variants=_split(variants) or (),
        scenarios=_split(scenarios),
        domains=_split(domains),
        concurrency=concurrency,
        out=out,
        estimate_tokens=estimate_tokens,
    )
    raise typer.Exit(anyio.run(run_evals, opts))


@app.command()
def record() -> None:
    """Record fixtures off the live mcp.tutu.ru (sequential and deliberately polite)."""
    from evals.fixtures_recipe import FIXTURE_CALLS
    from tutu_mcp.replay.bootstrap import record_fixtures

    anyio.run(record_fixtures, FIXTURE_CALLS)


@app.command()
def demo() -> None:
    """Hand-written traces over real fixtures — no model, no key. NOT a measurement."""
    from evals.demo import run_demo

    raise typer.Exit(anyio.run(run_demo))


@app.command()
def viewer(
    data: Path | None = typer.Option(
        None, help="eval run to bake in [default: out/eval-results.json]"
    ),
    out: Path | None = typer.Option(None, help="[default: viewer/trace-viewer.html]"),
) -> None:
    """Bake an eval run into a self-contained trace-viewer.html."""
    from viewer.build import DEFAULT_DATA, DEFAULT_OUT, build_viewer

    raise typer.Exit(build_viewer(data or DEFAULT_DATA, out or DEFAULT_OUT))


@app.command()
def docs(out: Path | None = typer.Option(None, help="[default: site/index.html]")) -> None:
    """Bake the user-docs page (pages/template.html) into a self-contained index.html."""
    from pages.build import DEFAULT_OUT, build_docs

    raise typer.Exit(build_docs(out or DEFAULT_OUT))


@app.command()
def measure() -> None:
    """Reproducible tools/list byte accounting (baseline vs proxy), from fixtures/."""
    from tutu_mcp.proxy.measure import print_report

    raise typer.Exit(print_report())


if __name__ == "__main__":
    app()
