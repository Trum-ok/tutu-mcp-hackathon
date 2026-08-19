"""Bakes the user-docs page from `template.html` + `styles.css` + `app.js`.

    uv run python tutu.py docs [--out site/index.html]

Живёт рядом с `viewer/` и переиспользует его: шрифты — те же локальные woff2
из `viewer/fonts/`, цветовые токены — общий `viewer/tokens.css`. Так вёрстка
доки и трейс-вьювера не расходится по двум скопированным друг у друга палитрам,
а собранная страница, как и trace-viewer.html, не ходит в сеть за шрифтами.

`__DEPLOY_URL__` in the template is resolved HERE, not by a separate CI-only
`sed` step — a local `make docs` used to ship the literal placeholder because
only the workflow knew how to fill it in. `MCP_PUBLIC_URL` is the same env var
the workflow already exports; unset (the local default), the page points at
the address `tutu.py serve` itself listens on.
"""

import os
from pathlib import Path

from viewer.build import VIEWER_DIR, inline_assets, inline_fonts, page_assets

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = Path(__file__).resolve().parent
TEMPLATE = PAGES_DIR / "template.html"
STYLES = PAGES_DIR / "styles.css"
SCRIPT = PAGES_DIR / "app.js"
TOKENS = VIEWER_DIR / "tokens.css"

DEFAULT_OUT = REPO_ROOT / "site" / "index.html"
LOCAL_DEPLOY_URL = "http://127.0.0.1:8800/mcp"


def load_styles() -> str:
    css = TOKENS.read_text(encoding="utf-8") + "\n" + STYLES.read_text(encoding="utf-8")
    return inline_fonts(css.rstrip("\n"))


def bake(template: str) -> str:
    template = inline_assets(template, page_assets(load_styles(), SCRIPT), TEMPLATE)

    if "__DEPLOY_URL__" not in template:
        raise ValueError(f"__DEPLOY_URL__ not found in {TEMPLATE}")
    deploy_url = os.environ.get("MCP_PUBLIC_URL") or LOCAL_DEPLOY_URL
    return template.replace("__DEPLOY_URL__", deploy_url)


def build_docs(out: Path) -> int:
    html = bake(TEMPLATE.read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"{out}  ({len(html) / 1024:.0f} КБ)")
    return 0
