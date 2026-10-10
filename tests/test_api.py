"""HTTP API tests. The graph is replaced, so no test calls a model."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict
from typer.testing import CliRunner

from brandforge import __version__
from brandforge.config import Settings
from brandforge.interfaces import api, cli
from brandforge.llm.base import GatewayError
from brandforge.models import (
    BrandProfile,
    Brief,
    FinalStatus,
    RunResult,
    RunState,
    Usage,
    Variant,
    VariantResult,
    new_run_state,
)

runner = CliRunner()

BRIEF: dict[str, Any] = {
    "product": "Voltride commuter e-bike",
    "audience": "city commuters",
    "objective": "conversion",
    "channels": ["search", "social"],
    "constraints": ["no discounts"],
}


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


def _body(brand: str = "voltride") -> dict[str, Any]:
    return {"brand": brand, "brief": BRIEF}


@pytest.fixture(autouse=True)
def checkpoint_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Checkpoints go to a temp file, never `.brandforge/` in the working tree."""
    path = tmp_path / "checkpoints.sqlite"
    settings = IsolatedSettings(checkpoint_db=path)
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    return path


@pytest.fixture(autouse=True)
def forbid_real_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that should run the graph replaces `run_graph`. The real one calls a model."""

    def forbidden(*_args: object, **_kwargs: object) -> RunState:
        raise AssertionError("POST /generate called the graph without a fake")

    monkeypatch.setattr(api, "run_graph", forbidden)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app=api.app, raise_server_exceptions=True) as http:
        yield http


def _install_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: FinalStatus = "complete",
    trace_id: str | None = "trace-abc",
) -> dict[str, Any]:
    """Replace the graph with one that returns a finished run and records how it was called."""
    seen: dict[str, Any] = {}

    def fake_run(
        brief: Brief,
        brand: BrandProfile,
        *,
        checkpointer: object = None,
        **kwargs: object,
    ) -> RunState:
        assert kwargs == {}
        seen["checkpointer"] = checkpointer
        seen["brand_id"] = brand.id
        seen["brief"] = brief
        state = new_run_state(brief, brand, run_id="run-1", trace_id=trace_id)
        state["result"] = RunResult(
            run_id="run-1",
            status=status,
            brand_id=brand.id,
            brand_version=brand.version,
            rubric_version=brand.rubric.version,
            variants=[
                VariantResult(
                    variant=Variant(
                        id="search-1",
                        channel="search",
                        headline="Ride in",
                        body="A calm commute.",
                        cta="Book a test ride",
                    ),
                    critique=None,
                    flagged=status != "complete",
                )
            ],
            revision_count=0,
            errors=[],
            usage=Usage(input_tokens=12, output_tokens=4, cost_usd=0.25),
            trace_id=trace_id,
        )
        seen["state"] = state
        return state

    monkeypatch.setattr(api, "run_graph", fake_run)
    return seen


@pytest.mark.parametrize("status", ["complete", "partial", "failed"])
def test_a_finished_run_returns_the_result_and_trace_id(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    checkpoint_db: Path,
    status: FinalStatus,
) -> None:
    seen = _install_run(monkeypatch, status=status, trace_id="trace-abc")

    response = client.post("/generate", json=_body(" voltride "))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    state = seen["state"]
    result = state["result"]
    assert isinstance(result, RunResult)
    body = response.json()
    assert body == result.model_dump(mode="json")
    assert body["status"] == status
    assert body["trace_id"] == "trace-abc"
    assert body["run_id"] == "run-1"
    assert "flagged_count" not in body
    assert "total_tokens" not in body["usage"]
    assert seen["brand_id"] == "voltride"
    assert seen["brief"] == Brief.model_validate(BRIEF)
    assert seen["checkpointer"] is not None
    assert checkpoint_db.is_file()


def test_trace_id_is_null_when_the_run_was_not_traced(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    _install_run(monkeypatch, trace_id=None)

    response = client.post("/generate", json=_body())

    assert response.status_code == 200
    assert "trace_id" in response.json()
    assert response.json()["trace_id"] is None


def test_unknown_brand_is_404(client: TestClient, checkpoint_db: Path) -> None:
    response = client.post("/generate", json=_body("no-such-brand"))

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "no-such-brand" in detail
    assert "voltride" in detail
    assert not checkpoint_db.exists()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"brand": "voltride"},
        {"brand": "   ", "brief": BRIEF},
        {"brand": "voltride", "brief": {**BRIEF, "channels": ["sms"]}},
        {"brand": "voltride", "brief": BRIEF, "note": "extra"},
        {"brand": "voltride", "brief": {**BRIEF, "slogan": "nope"}},
    ],
)
def test_a_body_that_is_not_a_brief_is_422(
    client: TestClient, checkpoint_db: Path, payload: dict[str, Any]
) -> None:
    response = client.post("/generate", json=payload)

    assert response.status_code == 422
    assert not checkpoint_db.exists()


def test_a_gateway_error_is_500_and_has_no_result(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> RunState:
        raise GatewayError("provider down")

    monkeypatch.setattr(api, "run_graph", fake_run)

    response = client.post("/generate", json=_body())

    assert response.status_code == 500
    assert response.json() == {"detail": "generation failed: provider down"}
    assert "traceback" not in response.text.lower()


@pytest.mark.parametrize(
    "exc",
    [OSError("disk full"), sqlite3.OperationalError("locked")],
)
def test_a_checkpoint_store_error_is_500(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, exc: Exception
) -> None:
    def broken(*_args: object, **_kwargs: object) -> NoReturn:
        raise exc

    monkeypatch.setattr(api, "open_checkpointer", broken)

    response = client.post("/generate", json=_body())

    assert response.status_code == 500
    assert response.json()["detail"] == f"generation failed: {exc}"


def test_a_run_with_no_result_is_500(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    def fake_run(
        brief: Brief,
        brand: BrandProfile,
        *,
        checkpointer: object = None,
        **_kwargs: object,
    ) -> RunState:
        del checkpointer
        return new_run_state(brief, brand, run_id="run-1", trace_id="trace-abc")

    monkeypatch.setattr(api, "run_graph", fake_run)

    response = client.post("/generate", json=_body())

    assert response.status_code == 500
    assert response.json() == {"detail": "the run finished without a result"}


def test_get_generate_is_not_allowed(client: TestClient) -> None:
    response = client.get("/generate")

    assert response.status_code == 405


def test_openapi_documents_generate_and_the_trace_id(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    schema = response.json()
    assert schema["info"]["title"] == "BrandForge"
    assert schema["info"]["version"] == __version__
    operation = schema["paths"]["/generate"]["post"]
    assert "200" in operation["responses"]
    assert "404" in operation["responses"]
    assert "500" in operation["responses"]
    components = schema["components"]["schemas"]
    assert "trace_id" in components["RunResult"]["properties"]
    trace = components["RunResult"]["properties"]["trace_id"]
    kinds = {item.get("type") for item in trace.get("anyOf", [])}
    assert {"string", "null"} <= kinds
    request = components["GenerateRequest"]["properties"]
    assert "brand" in request
    assert "brief" in request
    example = operation["requestBody"]["content"]["application/json"]["examples"]["voltride"]
    assert example["value"]["brand"] == "voltride"


def test_openapi_docs_render(client: TestClient) -> None:
    docs = client.get("/docs")
    redoc = client.get("/redoc")

    assert docs.status_code == 200
    assert docs.headers["content-type"].startswith("text/html")
    assert "swagger" in docs.text.lower()
    assert "/openapi.json" in docs.text
    assert redoc.status_code == 200
    assert redoc.headers["content-type"].startswith("text/html")
    assert "redoc" in redoc.text.lower()


def test_serve_help_points_at_the_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: IsolatedSettings())

    result = runner.invoke(cli.app, ["serve", "--help"])

    assert result.exit_code == 0
    assert "/docs" in result.stdout


def test_serve_binds_the_configured_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: IsolatedSettings(api_host="127.0.0.1", api_port=8123),
    )
    seen: dict[str, Any] = {}

    def fake_run(app: str, **kwargs: object) -> None:
        seen["app"] = app
        seen["host"] = kwargs["host"]
        seen["port"] = kwargs["port"]

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    result = runner.invoke(cli.app, ["serve"])

    assert result.exit_code == 0
    assert seen == {
        "app": "brandforge.interfaces.api:app",
        "host": "127.0.0.1",
        "port": 8123,
    }


def test_serve_flags_override_the_configured_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: IsolatedSettings(api_host="127.0.0.1", api_port=8000),
    )
    seen: dict[str, Any] = {}

    def fake_run(app: str, **kwargs: object) -> None:
        del app
        seen["host"] = kwargs["host"]
        seen["port"] = kwargs["port"]

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    result = runner.invoke(cli.app, ["serve", "--host", "0.0.0.0", "--port", "9000"])

    assert result.exit_code == 0
    assert seen == {"host": "0.0.0.0", "port": 9000}


def test_serve_rejects_a_port_outside_1_to_65535(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: IsolatedSettings())
    called: list[object] = []

    def fake_run(*_args: object, **_kwargs: object) -> None:
        called.append(True)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    result = runner.invoke(cli.app, ["serve", "--port", "0"])

    assert result.exit_code != 0
    assert called == []
