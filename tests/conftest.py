import json
from pathlib import Path

import pytest

from tutu_mcp.replay.store import FixtureStore

REPO_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def repo_fixtures() -> FixtureStore:
    """The real fixtures recorded from mcp.tutu.ru (`uv run python tutu.py record`)."""
    return FixtureStore(REPO_FIXTURES_DIR)


def load_result_payload(tool: str, scenario: str) -> dict:
    raw = json.loads((REPO_FIXTURES_DIR / tool / f"{scenario}.json").read_text(encoding="utf-8"))
    return json.loads(raw["result"]["text"])
