.PHONY: lint format test fixtures run-mock run-live evals evals-dry evals-record \
        demo-traces viewer viewer-demo

package ?= tutu_mcp evals scripts tests

lint:
	uv run ruff check $(package)
	uv run ruff format --check $(package)
	uv run ty check tutu_mcp evals

format:
	uv run ruff check --fix $(package)
	uv run ruff format $(package)

test:
	uv run pytest

fixtures:
	uv run python scripts/record_fixtures.py

run-mock:
	TUTU_PROXY_MODE=mock uv run python -m tutu_mcp.main

run-live:
	TUTU_PROXY_MODE=live uv run python -m tutu_mcp.main

# baseline vs proxy on recorded fixtures — needs OPENAI_API_KEY for the agent
evals:
	uv run python scripts/run_evals.py

# no model, no credentials: proves the harness itself is wired end to end
evals-dry:
	uv run python scripts/run_evals.py --agent scripted

# one live pass that records whatever fixtures the agent asks for and we lack
evals-record:
	uv run python scripts/run_evals.py --live --record-missing

# hand-written traces over real fixtures — no model, no key; for viewer development
demo-traces:
	uv run python scripts/demo_traces.py

# self-contained trace-viewer.html from the last real eval run
viewer:
	uv run python viewer/build.py

# ...and from the demo traces, so the viewer works before any key exists
viewer-demo: demo-traces
	uv run python viewer/build.py --data eval-results.demo.json
