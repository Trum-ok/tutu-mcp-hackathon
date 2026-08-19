.PHONY: lint format lint-front format-front test fixtures run-mock run-live evals \
        evals-dry evals-record demo-traces viewer viewer-demo docs site

package ?= tutu.py tutu_mcp evals viewer pages tests

lint:
	uv run ruff check $(package)
	uv run ruff format --check $(package)
	uv run ty check tutu.py tutu_mcp evals viewer pages

format:
	uv run ruff check --fix $(package)
	uv run ruff format $(package)

# фронт вьювера и доки отдельно от питона: своя тулза, свой цикл правок
lint-front:
	npx --no biome ci viewer pages

format-front:
	npx --no biome check --write viewer pages

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

# self-contained site/index.html — the user docs page
docs:
	uv run python tutu.py docs

# docs + trace viewer, both landing in site/ — mirrors the Pages build, so the
# "Открыть трейс-вьювер" link on the docs page resolves locally too
site: demo-traces
	uv run python tutu.py docs
	uv run python tutu.py viewer --data out/eval-results.demo.json --out site/trace-viewer.html
