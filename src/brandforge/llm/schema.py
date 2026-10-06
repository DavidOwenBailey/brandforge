"""Provider-neutral check for response schemas.

Every provider constrains structured output to a subset of JSON Schema, and the
subsets differ. The one thing none of them can express safely is a free-form object
(for example `dict[str, int]`): the keys are unknown, so strict modes either reject
it or force an object with no keys. We refuse such schemas up front, for every
provider, so agents write schemas that stay portable.
"""

from typing import Any

from pydantic import BaseModel


def _collect_free_form_objects(node: Any, path: str, found: list[str]) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            extra = node.get("additionalProperties")
            if not node.get("properties") or (extra is not None and extra is not False):
                found.append(path or "<root>")
        for key, value in node.items():
            _collect_free_form_objects(value, f"{path}.{key}" if path else str(key), found)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _collect_free_form_objects(item, f"{path}[{index}]", found)


def schema_problems(schema: type[BaseModel]) -> list[str]:
    """Paths in `schema` that are free-form objects, so not portable across providers.

    Use a list of `{key, value}` items instead and convert it to a dict in code.
    An empty result means the schema is safe to send.
    """
    found: list[str] = []
    _collect_free_form_objects(schema.model_json_schema(), "", found)
    return found
