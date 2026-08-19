"""Pins the numbers quoted in README.md / docs/findings.md / pages/template.html.

Not a correctness test — `test_compact_tools.py` already covers that. This one
exists so that a change to `compact_tools.py` or `surface.py` that shifts the
byte count fails LOUDLY here instead of leaving the docs quietly wrong. When it
fails: rerun `uv run python tutu.py measure`, update the pinned numbers below,
and update every place in the docs that quotes the old ones.
"""

from evals.measure import measure_catalog


def test_catalog_byte_counts_match_the_docs(repo_fixtures):
    m = measure_catalog(repo_fixtures)
    assert (m.n_tools_raw, m.n_tools_proxy) == (16, 18)
    assert (m.baseline_bytes, m.proxy_bytes) == (110164, 79411)
    assert (m.baseline_with_init_bytes, m.proxy_with_init_bytes) == (121646, 81428)
    assert (m.targeted_top_level_before, m.targeted_top_level_after) == (33321, 16616)
    assert m.top_level_description_bytes == 45728
    assert (m.schema_prose_bytes, m.schema_prose_after) == (32195, 15968)
    assert (m.rail_instructions_before, m.rail_instructions_after) == (27484, 51489)
