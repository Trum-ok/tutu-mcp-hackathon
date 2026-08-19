.PHONY: lint format test fixtures run-mock run-live evals evals-dry evals-record \
        demo-traces viewer viewer-demo

package ?= tutu.py tutu_mcp evals viewer tests

lint:
	uv run ruff check $(package)
	uv run ruff format --check $(package)
	uv run ty check tutu.py tutu_mcp evals viewer

format:
	uv run ruff check --fix $(package)
	uv run ruff format $(package)

test:
	uv run pytest

fixtures:
	uv run python tutu.py record

run-mock:
	TUTU_PROXY_MODE=mock uv run python tutu.py serve

run-live:
	TUTU_PROXY_MODE=live uv run python tutu.py serve

# baseline vs proxy on recorded fixtures — needs OPENAI_API_KEY for the agent
evals:
	uv run python tutu.py evals

# no model, no credentials: proves the harness itself is wired end to end
evals-dry:
	uv run python tutu.py evals --agent scripted

# one live pass that records whatever fixtures the agent asks for and we lack
evals-record:
	uv run python tutu.py evals --live --record-missing

# hand-written traces over real fixtures — no model, no key; for viewer development
demo-traces:
	uv run python tutu.py demo

# self-contained trace-viewer.html from the last real eval run
viewer:
	uv run python tutu.py viewer

# ...and from the demo traces, so the viewer works before any key exists
viewer-demo: demo-traces
	uv run python tutu.py viewer --data out/eval-results.demo.json
