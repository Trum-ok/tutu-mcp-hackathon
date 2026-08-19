"""Recording and playback of upstream responses.

Named `replay`, not `fixtures`, because `fixtures/` at the repo root is the
*data* — the recorded JSON. This package is the *mechanism* that reads, writes
and serves it. The two used to share a name, which made every
`from tutu_mcp.fixtures.store import FixtureStore` a small puzzle: the store
lives here, the payloads it stores live there.
"""
