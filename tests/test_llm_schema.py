"""Portable response-schema check: provider-neutral, no SDK involved."""

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from brandforge.llm.schema import schema_problems
from brandforge.models import Critique, Variant


class _ScoreItem(BaseModel):
    criterion: str
    score: int


class _ListReply(BaseModel):
    scores: list[_ScoreItem]
    note: str | None = None


class _Inner(BaseModel):
    labels: dict[str, str]


class _Outer(BaseModel):
    items: list[_Inner]


class _AnyDict(BaseModel):
    payload: dict[str, Any]


class _OpenModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str


class _Empty(BaseModel):
    pass


def test_flags_free_form_dict() -> None:
    assert schema_problems(Critique) == ["properties.scores"]


def test_flags_dict_of_any() -> None:
    assert schema_problems(_AnyDict) == ["properties.payload"]


def test_flags_dict_nested_in_definitions() -> None:
    (path,) = schema_problems(_Outer)
    assert "_Inner" in path
    assert path.endswith("properties.labels")


def test_flags_model_that_allows_extra_keys() -> None:
    assert schema_problems(_OpenModel) == ["<root>"]


def test_flags_object_with_no_fields() -> None:
    assert schema_problems(_Empty) == ["<root>"]


@pytest.mark.parametrize("schema", [Variant, _ListReply])
def test_accepts_fixed_shape_schemas(schema: type[BaseModel]) -> None:
    assert schema_problems(schema) == []
