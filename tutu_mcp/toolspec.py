"""Tool descriptors for the tools this proxy adds itself, declared as Pydantic models.

Everything Tutu exposes passes through with its own schema untouched; these are
the two tools that are ours (`assess_request`, `check_groundedness`), and hand-
written dicts were the wrong way to carry them — the argument shape lived in a
literal, the validation lived in a `try/except KeyError` somewhere else, and
nothing tied the two together.

Two things this does beyond calling `model_json_schema()`:

* **Inlines `$defs`/`$ref`.** Pydantic factors nested models into `$defs` and
  points at them. That is valid JSON Schema, but a tool schema crosses into
  whichever client the user runs, and a flat object is the shape every one of
  them handles. The nesting here is one level deep, so inlining costs nothing.
* **Drops `title`.** Pydantic adds a `title` to the schema and to every field —
  pure restatement of the key name. On a proxy whose entire premise is that the
  always-on catalog is too big, shipping that would be self-defeating.
"""

from typing import Any

from pydantic import BaseModel, ValidationError


def _inline(node: Any, defs: dict[str, Any]) -> Any:
    """Resolve `$ref`s against `defs` and strip `title` keys, recursively."""
    if isinstance(node, list):
        return [_inline(item, defs) for item in node]
    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        target = defs.get(ref.removeprefix("#/$defs/"), {})
        # keep any siblings of the $ref (e.g. a field-level description)
        merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
        return _inline(merged, defs)

    return {k: _inline(v, defs) for k, v in node.items() if k not in ("title", "$defs")}


def tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The `inputSchema` for a tool whose arguments are `model`."""
    schema = model.model_json_schema()
    return _inline(schema, schema.get("$defs", {}))


def tool_spec(
    name: str,
    description: str,
    model: type[BaseModel],
    *,
    annotations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One entry for `tools/list`, in the same shape upstream's own entries take."""
    spec: dict[str, Any] = {
        "name": name,
        "description": description,
        "inputSchema": tool_schema(model),
    }
    if annotations:
        spec["annotations"] = annotations
    return spec


def parse_args[T: BaseModel](model: type[T], arguments: dict[str, Any]) -> T | str:
    """Validated arguments, or a message naming what was wrong.

    Returning the message instead of raising keeps the tool bodies' `(text,
    is_error)` contract intact: a malformed call is an error the agent can read
    and correct, not a stack trace that kills the request.
    """
    try:
        return model.model_validate(arguments)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<корень>'}: {e['msg']}" for e in exc.errors()
        )
        return f"invalid arguments: {problems}"
