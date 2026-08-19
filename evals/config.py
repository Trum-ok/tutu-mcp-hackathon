"""Credentials and model defaults for the eval harness.

Split out of `tutu_mcp/config.py` so the dependency arrow points one way only:
the harness imports the proxy it measures, never the reverse. Importing
`tutu_mcp.config` here is also what loads `.env` — that happens as an import
side effect there, and this module relies on it.
"""

import os

from evals.options import Effort
from tutu_mcp.config import REPO_ROOT as REPO_ROOT

# Override with --model or OPENAI_MODEL; `--list-models` prints what the key can reach.
DEFAULT_MODEL = "gpt-5"


def openai_credentials_source() -> str | None:
    """Which credential the OpenAI SDK will pick up, or `None` if it will find
    nothing. Used to fail fast with a useful message instead of letting the run
    die on an auth error several scenarios in."""
    if os.environ.get("OPENAI_API_KEY"):
        base = os.environ.get("OPENAI_BASE_URL")
        return f"OPENAI_API_KEY → {base}" if base else "OPENAI_API_KEY"
    return None


def openai_model_default() -> str:
    """`OPENAI_MODEL` lets the whole team pin one model in .env without passing
    --model on every invocation."""
    return os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL


class InvalidEffortError(ValueError):
    """`OPENAI_EFFORT` holds a value no model will accept."""


def openai_effort_default() -> Effort | None:
    """Reasoning effort pinned in .env, the same way `OPENAI_MODEL` pins the model.

    `None` — the variable is unset — is not the same as `Effort.NONE`: unset means
    the harness sends no `reasoning` field at all and the model applies its own
    default, while `none` is an explicit instruction to skip reasoning that only
    some models accept.

    Validated here rather than at the first API call: a typo in .env would
    otherwise surface as a 400 on every scenario, after the run has already
    started spending tokens.
    """
    raw = os.environ.get("OPENAI_EFFORT", "").strip()
    if not raw:
        return None
    try:
        return Effort(raw.lower())
    except ValueError:
        allowed = ", ".join(e.value for e in Effort)
        raise InvalidEffortError(
            f"OPENAI_EFFORT={raw!r} — недопустимое значение. Допустимы: {allowed} "
            f"(или уберите переменную, чтобы модель выбрала усилие сама)."
        ) from None


MISSING_CREDENTIALS_HELP = """\
Нет ключа OpenAI — агент в эвалах не запустится.

Что сделать:
  1. Создайте .env в корне репозитория (он в .gitignore):
         cp .env.example .env
     и впишите туда OPENAI_API_KEY=sk-...
  2. Либо экспортируйте ключ в текущей оболочке:
         export OPENAI_API_KEY=sk-...

Если используете свой шлюз, совместимый с OpenAI, задайте ещё OPENAI_BASE_URL.

Без ключа доступны только:
  make evals-dry     — самопроверка харнесса на рукописных планах
  --estimate-tokens  — оценка токенов через tiktoken вместо точного замера\
"""
