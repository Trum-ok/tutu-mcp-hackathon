"""The committed eval run is what the published trace viewer shows.

It is data, not code, so nothing else fails when it goes stale — hence this file.
A snapshot that no longer parses, covers half the set, or disagrees with the
numbers quoted in the docs is a broken showcase that still deploys cleanly.
"""

import json
from pathlib import Path

import pytest

from evals.scenarios import SCENARIOS

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "evals" / "runs"
# the one the Makefile and the Pages workflow bake in
SNAPSHOT = RUNS_DIR / "2026-08-21-gpt-5.6-luna.json"


@pytest.fixture(scope="module")
def snapshot() -> dict:
    assert SNAPSHOT.is_file(), f"{SNAPSHOT.name} отсутствует — `make site` соберёт пустую витрину"
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_the_snapshot_covers_the_whole_scenario_set(snapshot):
    """Three of twenty-two was exactly the problem the snapshot exists to fix."""
    for variant in snapshot["variants"]:
        ids = {s["id"] for s in variant["scenarios"]}
        missing = {s.id for s in SCENARIOS} - ids
        assert not missing, f"{variant['variant']}: не хватает сценариев {sorted(missing)}"


def test_the_snapshot_compares_both_variants(snapshot):
    assert [v["variant"] for v in snapshot["variants"]] == ["baseline", "proxy"]


def test_the_snapshot_carries_a_real_measurement(snapshot):
    """An estimated surface would put a `~` on the headline number the docs quote
    without one, and a scripted agent would mean no model ever ran."""
    assert snapshot["surface_tokens_exact"] is True
    assert snapshot["agent"] != "scripted"


def test_the_surface_numbers_match_the_ones_the_docs_quote(snapshot):
    """README, docs/findings.md and the docs page all print these two figures. If
    a rerun moves them, the docs are wrong until someone notices — this notices."""
    tokens = {v["variant"]: v["surface"]["tokens"] for v in snapshot["variants"]}
    assert tokens == {"baseline": 25269, "proxy": 15364}

    quoted = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "25 269" in quoted and "15 364" in quoted


def test_every_trace_has_something_to_show(snapshot):
    """A viewer page full of empty transcripts is worse than no page."""
    for variant in snapshot["variants"]:
        for scenario in variant["scenarios"]:
            assert scenario["answer"] or scenario["failure"], (
                f"{variant['variant']}/{scenario['id']}: ни ответа, ни причины сбоя"
            )


def test_the_snapshot_carries_the_metrics_the_viewer_reads(snapshot):
    """A snapshot written by an older report renders as "Выдумано 0" — the viewer
    falls back rather than crashing, which is worse: a wrong number looks fine."""
    for variant in snapshot["variants"]:
        metrics = variant["metrics"]
        claims = [c for s in variant["scenarios"] for c in s["groundedness"]["claims"]]
        assert metrics["fabricated_claims"] == sum(
            1 for c in claims if c["status"] == "unavailable"
        )
        assert metrics["checkable_claims"] == sum(1 for c in claims if c["status"] != "user_stated")
