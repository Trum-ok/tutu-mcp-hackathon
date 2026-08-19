"""Bakes an eval run into a single self-contained HTML file.

    uv run python tutu.py viewer [--data out/eval-results.json] [--out viewer/trace-viewer.html]

Self-contained on purpose: the pitch happens on someone's laptop, possibly on
conference wifi. A page that fetches its data over `file://` would hit CORS and
show nothing, and one that needs a dev server is one more thing to fail on stage.
Double-clicking the output must be enough.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "template.html"
PLACEHOLDER = "__TRACE_DATA__"

DEFAULT_DATA = REPO_ROOT / "out" / "eval-results.json"
DEFAULT_OUT = REPO_ROOT / "viewer" / "trace-viewer.html"


def bake(data: dict, template: str) -> str:
    # `</script>` inside a tool result would close the data block early; escaping
    # `<` at the JSON level keeps the payload inert wherever it lands in the page.
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    if PLACEHOLDER not in template:
        raise ValueError(f"{PLACEHOLDER} not found in {TEMPLATE}")
    return template.replace(PLACEHOLDER, payload)


def build_viewer(data_path: Path, out: Path) -> int:
    if not data_path.is_file():
        print(
            f"{data_path} не найден.\n"
            "Сначала соберите данные:\n"
            "  make evals        — реальный прогон (нужен OPENAI_API_KEY)\n"
            "  make demo-traces  — демо-трейсы на фикстурах, без модели",
            file=sys.stderr,
        )
        return 2

    data = json.loads(data_path.read_text(encoding="utf-8"))
    html = bake(data, TEMPLATE.read_text(encoding="utf-8"))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    scenarios = sum(len(v.get("scenarios", [])) for v in data.get("variants", []))
    print(f"{out}  ({len(html) / 1024:.0f} КБ, {scenarios} трейсов, агент: {data.get('agent')})")
    return 0
