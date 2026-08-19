# tutu-mcp-proxy

Compacting/grounding MCP proxy in front of [`mcp.tutu.ru`](https://mcp.tutu.ru/mcp), built for
Tutu's hackathon (track 2 — "оптимизация инструментов"). Same 16 tools, same behavior, plus:

- **Trimmed always-on tool catalog.** `tools/list` on the real server runs ~110 KB / ~30K tokens
  before an agent has searched anything (measured live 2026-08-19). `search_rail`,
  `get_rail_seatmap` and `search_hotels` — the biggest offenders — get a short top-level
  `description` here; the trimmed edge-case prose isn't lost, it's spliced into the *call result*
  of the matching `get_<domain>_instructions` tool (see `tutu_mcp/proxy/compact_tools.py`), so it's
  paid only when actually read, the same on-demand principle upstream already applies to those
  instructions tools. `inputSchema` is left untouched everywhere.
- **`check_groundedness` tool.** Deterministically checks a drafted answer against the
  `tool_result` payload(s) it's based on — extracts prices/times/train-or-flight-codes/URLs from
  the answer text and verifies each is actually present in the JSON, no LLM judge involved (see
  `tutu_mcp/groundedness.py`).
- **Premise gate + `assess_request`.** `check_groundedness` checks the *output* of a turn;
  this checks the *input*. Every value that NARROWS a search must come from the user or from an
  earlier `tool_result` — there is no third source. A filter the model invented (the classic:
  silently assuming what time an event ends, then filtering return trains by it) makes the call
  return `clarification_required` INSTEAD of data, so the confident comparison table cannot be
  built on it. The agent resolves it by asking the user, by declaring `_sources`, or by declaring
  `_assume` — and an openly assumed value forces a preamble the answer must OPEN with, which
  `check_groundedness` then verifies (see `tutu_mcp/premises.py`). Also cross-checks the user's own
  words against each other: "в субботу 11 октября" when 11.10.2026 is a Sunday is flagged as a
  likely typo — shown, never silently corrected.
- **Mock mode.** Replays fixtures recorded from the live server instead of calling it — safe to
  hammer during eval runs or a demo without touching the hackathon's shared rate limit.

## Layout

```
tutu_mcp/                  the proxy server itself — this is what `make run-mock` starts
  backend.py               ToolBackend protocol shared by the live and mock clients
  upstream/client.py       live backend — wraps the official `mcp` SDK client
  replay/store.py          on-disk fixture store (VCR-style: match by tool + exact arguments)
  replay/mock_client.py    mock backend — serves fixtures/ instead of calling Tutu
  replay/recording.py      records fixtures/ off the live server while a run proceeds
  proxy/compact_tools.py   description trimming + appendix splicing
  proxy/server.py          builds the proxy MCPServer (tools/list, tools/call, check_groundedness)
  groundedness.py          claim extraction + grounding check, used by proxy/server.py
  premises.py              premise gate: input-side provenance, typo detection, assess_request
  toolspec.py              Pydantic -> tools/list descriptors for the tools we add ourselves
  config.py                TUTU_* runtime settings; loads .env on import
  main.py                  entrypoint
evals/                     the harness that measures the proxy — imports it, never the reverse
  scenarios.py             the scenarios both surfaces are run through
  variants.py              the two tool surfaces under comparison: baseline vs proxy
  agent.py                 the agent under test (OpenAI, or a scripted stand-in for CI)
  runner.py                drives one scenario, collects a transcript
  checks.py, report.py     per-scenario metrics and the aggregate report
  tokens.py                token accounting (exact via the API, or offline via tiktoken)
  config.py                OPENAI_* credentials and model default — harness-only
scripts/record_fixtures.py   records fixtures/ from the live server
scripts/run_evals.py          runs the baseline-vs-proxy comparison
scripts/demo_traces.py        hand-written traces over real fixtures (no model needed)
viewer/template.html          trace viewer UI; build.py bakes an eval run into it
fixtures/                     recorded responses (tools/list, instructions, searches, ...)
tests/                         pytest suite, runs entirely against recorded fixtures (no network)
```

## Configuration

All settings come from environment variables. `tutu_mcp/config.py` loads a `.env` file from the repo
root on import — it never overrides what the shell already exports, so an explicit `export` still
wins. `.env` is gitignored; `.env.example` documents every variable and is the file to copy:

```bash
cp .env.example .env      # then fill in OPENAI_API_KEY
```

| Variable | Needed for | Default |
|---|---|---|
| `OPENAI_API_KEY` | **evals only** — the agent under test + exact token counts | — |
| `OPENAI_MODEL` | default model for evals (`--model` overrides per run) | `gpt-5` |
| `OPENAI_BASE_URL` | set for an OpenAI-compatible gateway | `https://api.openai.com/v1` |
| `TUTU_PROXY_MODE` | `mock` (fixtures, no network) or `live` | `mock` |
| `TUTU_UPSTREAM_URL` | the upstream MCP server | `https://mcp.tutu.ru/mcp` |
| `TUTU_FIXTURES_DIR` | where recorded fixtures live | `./fixtures` |
| `TUTU_PROXY_HOST` / `TUTU_PROXY_PORT` | where the proxy listens | `127.0.0.1` / `8800` |

**The proxy itself needs no OpenAI key** — only the eval harness does, because that's the part
that runs a model. `make run-mock`, `make run-live` and the whole test suite work without one.

The runner checks for the key before the first request and refuses up front with instructions,
rather than dying on an auth error several scenarios into a run that has already spent time and
(in `--live`) upstream rate limit.

Not sure which model ids the key can reach? `uv run python scripts/run_evals.py --list-models`
prints them straight from the API — don't guess from documentation.

## Running

```bash
uv sync
uv run python -m tutu_mcp.main            # mock mode (default) — http://127.0.0.1:8800/mcp
TUTU_PROXY_MODE=live uv run python -m tutu_mcp.main   # proxies the real mcp.tutu.ru
```

Point any MCP client at `http://127.0.0.1:8800/mcp` (Streamable HTTP, no auth — matches upstream).

## Fixtures

```bash
uv run python scripts/record_fixtures.py
```

Sequential, rate-limit-polite, records `tools/list` + all 6 instructions tools + representative
search/detail/checkout scenarios per domain (including a couple of edge cases: an unmatched
`train_numbers` filter, an invalid date). Mock-mode lookups match on the tool name **and** exact
(normalized) arguments — call a tool with arguments that weren't recorded and you get a clear
`FixtureNotFoundError` naming what scenarios *are* available, not a silent wrong answer.

Known gap: no `429` fixture — deliberately provoking one would burn the shared hackathon rate
limit for every other team, so that one needs to be hand-authored instead.

## Evals — baseline vs proxy

```bash
uv run python scripts/run_evals.py                     # fixtures + real model
uv run python scripts/run_evals.py --agent scripted    # no model: harness self-check
uv run python scripts/run_evals.py --live --record-missing  # fill fixture gaps, once
uv run python scripts/run_evals.py --model <id>        # per-model breakdown
uv run python scripts/run_evals.py --list-models       # what this key can reach
```

Runs every scenario in `evals/scenarios.py` through two tool surfaces — `baseline` (raw
upstream) and `proxy` (compacted + `check_groundedness`) — against the **same** backend, so tool
results are identical between variants and the only independent variable is the surface itself.
The proxy variant calls the same functions the live MCP server calls, so this measures the real
proxy, not a re-implementation of it.

Reported per variant: premise metrics (how often the gate fired, how many runs proceeded on an
assumption, what share disclosed it up front, and — the guard against turning the assistant into a
questionnaire — how many clarifying questions were asked on scenarios that had nothing to clarify),
tool-surface cost in tokens (the number paid on every session before any
search), task success against per-scenario deterministic checks, pooled groundedness rate, tool
calls, tool errors, and p50/p95 latency — plus a per-scenario matrix showing exactly where the two
variants diverge. Full detail lands in `eval-results.json` for the trace viewer to render.

Two honesty notes built into the output:

- **Fixture misses are counted separately from tool errors.** A gap in our recording must never
  read as upstream misbehaving; the report says so explicitly and tells you how to fill it.
- **Token counts are labelled `~` when estimated.** OpenAI has no token-counting endpoint, so the
  exact figure comes from one real probe request per variant, reading back `usage.prompt_tokens` —
  the only way to capture the provider's own tool-serialization overhead. `--estimate-tokens`
  swaps in a tiktoken count instead; it is a good approximation but still an estimate, so it
  prints with a `~` rather than a precise-looking number nobody can back up.

Model choice is per run (`--model`, or `OPENAI_MODEL` in `.env`). Reasoning effort is sent only
when `--effort` is passed, so the model otherwise uses its own default. Sampling parameters are
never sent: reasoning models reject them, and an eval wants the model's own default behavior.

**Endpoint.** The runner defaults to `/v1/responses`. Chat Completions refuses function tools
together with reasoning on current reasoning models — it answers *"use /v1/responses or set
reasoning_effort to 'none'"* — and the second option would quietly measure a reasoning model with
its reasoning switched off, then attribute the numbers to the model by name. Pass `--api chat`
for OpenAI-compatible gateways that don't implement `/v1/responses`. The surface-token counter
follows the same flag, because the two endpoints serialize tool definitions differently and
counting on one while running on the other would report a cost the run never paid.

**Fixture matching ignores schema defaults.** A model spells out every optional argument —
`page: 1`, `sort: "price_asc"`, `view: "compact"`, `from_city: null` — where a human recording a
fixture writes none of them. Upstream treats those as identical requests, so the lookup prunes any
argument that is `null` or equal to the default declared in the tool's own `inputSchema` (read from
the recorded `tools/list`, so it can't drift from Tutu's schema). Without this nearly every call in
a model-driven mock run misses. A value that merely looks default-ish is not pruned: `page_size: 30`
against a default of `10` is a genuinely different request and correctly stays a miss.

## Trace viewer

```bash
make viewer        # from the last real eval run (eval-results.json)
make viewer-demo   # from hand-written demo traces — no model, no key needed
```

Both write `viewer/trace-viewer.html`, a **single self-contained file**: double-click it, no
server, no network. That is deliberate — the pitch runs on someone's laptop on conference wifi,
and a page that fetches its data over `file://` would hit CORS and show nothing.

What it shows: the user's request, the agent's answer with every price / time / train-or-flight
code / URL highlighted by status — green `confirmed`, amber `assumed`, red `unavailable` — the
per-scenario checks, and every tool call with its arguments and raw response. **Click any
highlighted value** and a drawer opens showing the exact fragment of the server payload it came
from, or stating plainly that it appears in none of them.

Three things the UI is careful about:

- **Synthetic runs are labelled.** A `demo:` or `scripted:` agent gets an amber `НЕ ЗАМЕР` badge in
  the header, so hand-written demo traces can never be mistaken on screen for a measurement.
- **An empty highlight is not a pass.** When an answer contains no typed claims there is nothing to
  colour, and the panel says so explicitly rather than looking clean — the verdict for those
  answers comes from the scenario checks, not the highlighting.
- **`assumed` is not green.** A declared assumption is disclosed, not proven, so it gets its own
  colour and never counts toward the groundedness rate.

`make demo-traces` writes `eval-results.demo.json`, deliberately *not* `eval-results.json`: real
runs own that path, and two people generating different things into one file overwrite each
other's work. The demo answers are hand-written over the real recorded fixtures, so every
"confirmed" claim is genuinely confirmed against real Tutu data and every fabricated one is
genuinely absent — the highlighting is exercised for real, not faked with hardcoded colours.

## Status

`tools/list` goes from 108 539 to 78 334 bytes — **27.8%** — and 33.3% counting each side's
always-on `initialize` instructions, since the proxy replaces Tutu's 11 KB block with its own 1.6 KB one, and that is measured AFTER adding our own two
tools (`assess_request` 1 309 bytes, `check_groundedness` 755), not before.

Most of that came from the second seam, not the first: top-level tool descriptions were ~14 KB of
the catalog, while prose inside `inputSchema.properties[].description` was 31.7 KB — 35% of the
whole thing. Fields keep a short hint (or, when they carry an `enum`, nothing — the permitted
values already say what the field accepts) and the full sentence moves into the paired
instructions tool's call result. The authoritative half of every schema — `type`, `enum`,
`required`, `format`, field names — still goes out byte-identical, which
`test_schema_types_are_never_touched` pins directly.

The trade-off is honest and measured: `get_rail_instructions` grows from 26 KB to 49 KB. That is
paid only by a session that actually calls it, instead of by every session before it has searched
anything. `create_checkout_link` and `get_offer_details` keep their schema prose in full — neither
has a paired instructions tool to park it in, and the former is the purchase dispatcher.

Not done, but measured: deferring the tools themselves behind a `find_tools` lookup
(`notifications/tools/list_changed`) would put the pre-search catalog at 11 702 bytes / 89%. It is
left out deliberately — a client that ignores the notification would never see the search tools at
all, which is the first place this proxy would actually break compatibility.

This is a scaffold: the mechanism (compaction, appendix splicing, groundedness, premise gate,
fixtures) is real and tested, but only 3 of 16 tools have hand-tuned compact descriptions so far
(`search_rail`, `get_rail_seatmap`, `search_hotels` — chosen because they're the biggest and the
ones the source article calls out by name). `create_checkout_link` is deliberately left alone —
see the docstring in `tutu_mcp/proxy/compact_tools.py` for why. Extending coverage to the rest of the
domains, and building the trace-viewer UI on top of `check_groundedness`, are the next steps.

## Commands

```bash
make lint          # ruff check + format --check + ty
make format        # ruff check --fix + format
make test          # pytest
make fixtures      # re-record fixtures/ from the live server
make run-mock      # run the proxy against recorded fixtures
make run-live      # run the proxy against the real mcp.tutu.ru
make evals         # baseline vs proxy (needs OPENAI_API_KEY)
make evals-dry     # same wiring, scripted agent, no credentials needed
make evals-record  # one live pass that records missing fixtures
make demo-traces   # hand-written traces over real fixtures, no model
make viewer        # trace-viewer.html from the last real run
make viewer-demo   # trace-viewer.html from the demo traces
```
