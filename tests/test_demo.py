"""Demo page tests. The API is faked, so no test calls a model or opens a port."""

import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic_settings import SettingsConfigDict
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

from brandforge.config import Budgets, Settings
from brandforge.interfaces import cli, demo
from brandforge.models import (
    Brief,
    Critique,
    FinalStatus,
    RunError,
    RunResult,
    Usage,
    Variant,
    VariantResult,
)

runner = CliRunner()

SAMPLE_BRIEF = Path("src/brandforge/briefs/voltride_01_commuter_ebike.yaml")

PASTED = """\
product: Gift tin
audience: hosts
objective: awareness
channels: [email]
constraints:
  - no discounts
"""


class IsolatedSettings(Settings):
    """Settings that ignore any local .env, so tests are hermetic."""

    model_config = SettingsConfigDict(env_file=None)


def _critique(variant_id: str, *, passed: bool) -> Critique:
    return Critique(
        variant_id=variant_id,
        scores={
            "voice": 5 if passed else 2,
            "clarity": 4,
            "call_to_action": 4 if passed else 2,
        },
        overall=4.3 if passed else 2.7,
        passed=passed,
        fixes=[] if passed else ["Name the bike", "Drop the hype"],
    )


def _result(*, status: FinalStatus = "partial", trace_id: str | None = "trace-abc") -> RunResult:
    passed = VariantResult(
        variant=Variant(
            id="search-1",
            channel="search",
            headline="Ride in",
            body="A calm commute.",
            cta="Book a test ride",
        ),
        critique=_critique("search-1", passed=True),
        flagged=False,
    )
    flagged = VariantResult(
        variant=Variant(
            id="social-1",
            channel="social",
            headline="Go faster",
            body="The city is waiting.",
            cta="Ride today",
        ),
        critique=_critique("social-1", passed=False),
        flagged=True,
    )
    unscored = VariantResult(
        variant=Variant(
            id="email-1",
            channel="email",
            headline="A note",
            body="Short.",
            cta="Read it",
        ),
        critique=None,
        flagged=True,
    )
    return RunResult(
        run_id="run-1",
        status=status,
        brand_id="voltride",
        brand_version="1.0",
        rubric_version="1.0",
        variants=[passed, flagged, unscored],
        revision_count=1,
        errors=[RunError(node="writer", message="social came back short")],
        usage=Usage(input_tokens=12, output_tokens=4, cost_usd=0.25),
        trace_id=trace_id,
    )


def _ok(request: httpx.Request) -> httpx.Response:
    del request
    return httpx.Response(200, json=_result().model_dump(mode="json"))


def test_example_brief_matches_the_voltride_sample() -> None:
    sample = SAMPLE_BRIEF.read_text(encoding="utf-8")
    assert demo.EXAMPLE_BRIEF.strip() == sample.strip()
    assert demo.parse_brief(demo.EXAMPLE_BRIEF) == demo.parse_brief(sample)


def test_parse_brief_accepts_json() -> None:
    text = json.dumps(
        {
            "product": "Gift tin",
            "audience": "hosts",
            "objective": "awareness",
            "channels": ["email"],
            "constraints": ["no discounts"],
        }
    )

    brief = demo.parse_brief(text)

    assert brief == Brief(
        product="Gift tin",
        audience="hosts",
        objective="awareness",
        channels=["email"],
        constraints=["no discounts"],
    )


def test_parse_brief_rejects_a_blank_paste() -> None:
    with pytest.raises(demo.DemoError, match="Paste a brief"):
        demo.parse_brief("  \n")


def test_parse_brief_rejects_a_list() -> None:
    with pytest.raises(demo.DemoError, match="mapping"):
        demo.parse_brief("- just a list\n")


def test_parse_brief_rejects_broken_yaml() -> None:
    with pytest.raises(demo.DemoError, match="not valid YAML or JSON"):
        demo.parse_brief("product: [\n")


def test_parse_brief_reports_the_field_that_failed() -> None:
    with pytest.raises(demo.DemoError, match="channels"):
        demo.parse_brief(
            "product: Tea\naudience: neighbours\nobjective: awareness\nchannels: [search, search]\n"
        )


def test_generation_timeout_covers_the_last_model_call() -> None:
    settings = IsolatedSettings(
        budgets=Budgets(max_wall_clock_seconds=90, request_timeout_seconds=60)
    )

    assert demo.generation_timeout(settings) == 165


def test_request_posts_the_brief_and_returns_the_result() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200, json=_result(status="failed", trace_id=None).model_dump(mode="json")
        )

    brief = demo.parse_brief(demo.EXAMPLE_BRIEF)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = demo.request_generation(
        "voltride",
        brief,
        api_base_url="http://api.test:8000/",
        timeout=5,
        client=client,
    )

    assert seen["url"] == "http://api.test:8000/generate"
    assert seen["body"] == {"brand": "voltride", "brief": brief.model_dump(mode="json")}
    assert result.status == "failed"
    assert result.trace_id is None
    assert result.flagged_count == 2


def test_a_passed_in_client_stays_open() -> None:
    client = httpx.Client(transport=httpx.MockTransport(_ok))
    brief = demo.parse_brief(demo.EXAMPLE_BRIEF)

    demo.request_generation(
        "voltride", brief, api_base_url="http://api.test", timeout=5, client=client
    )
    demo.request_generation(
        "voltride", brief, api_base_url="http://api.test", timeout=5, client=client
    )

    client.close()


def test_the_owned_client_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class RecordingClient(httpx.Client):
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            events.append("open")
            super().__init__(transport=httpx.MockTransport(_ok))

        def close(self) -> None:
            events.append("close")
            super().close()

    monkeypatch.setattr(httpx, "Client", RecordingClient)

    result = demo.request_generation(
        "voltride",
        demo.parse_brief(demo.EXAMPLE_BRIEF),
        api_base_url="http://api.test",
        timeout=5,
    )

    assert result.run_id == "run-1"
    assert events == ["open", "close"]


def test_unknown_brand_uses_the_api_detail() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Unknown brand 'nope'. Known brands: voltride"})

    with pytest.raises(demo.DemoError, match="Unknown brand 'nope'"):
        demo.request_generation(
            "nope",
            demo.parse_brief(demo.EXAMPLE_BRIEF),
            api_base_url="http://api.test",
            timeout=5,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )


def test_a_rejected_brief_names_the_field() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "detail": [
                    {
                        "loc": ["body", "brief", "channels"],
                        "msg": "Field required",
                        "type": "missing",
                    }
                ]
            },
        )

    with pytest.raises(demo.DemoError, match=r"brief\.channels: Field required"):
        demo.request_generation(
            "voltride",
            demo.parse_brief(demo.EXAMPLE_BRIEF),
            api_base_url="http://api.test",
            timeout=5,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )


def test_a_failed_generation_uses_the_api_sentence() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "generation failed: boom"})

    with pytest.raises(demo.DemoError, match="generation failed: boom"):
        demo.request_generation(
            "voltride",
            demo.parse_brief(demo.EXAMPLE_BRIEF),
            api_base_url="http://api.test",
            timeout=5,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )


def test_a_non_json_success_is_an_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"nope")

    with pytest.raises(demo.DemoError, match="not JSON"):
        demo.request_generation(
            "voltride",
            demo.parse_brief(demo.EXAMPLE_BRIEF),
            api_base_url="http://api.test",
            timeout=5,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )


def test_a_success_that_is_not_a_result_is_an_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "complete"})

    with pytest.raises(demo.DemoError, match="not a run result"):
        demo.request_generation(
            "voltride",
            demo.parse_brief(demo.EXAMPLE_BRIEF),
            api_base_url="http://api.test",
            timeout=5,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )


def test_a_timeout_says_the_run_can_take_a_while() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow")

    with pytest.raises(demo.DemoError, match="did not respond in time"):
        demo.request_generation(
            "voltride",
            demo.parse_brief(demo.EXAMPLE_BRIEF),
            api_base_url="http://api.test",
            timeout=5,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )


def test_a_connection_error_names_the_serve_command() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(demo.DemoError, match="brandforge serve"):
        demo.request_generation(
            "voltride",
            demo.parse_brief(demo.EXAMPLE_BRIEF),
            api_base_url="http://127.0.0.1:8000",
            timeout=5,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )


def test_variant_views_carry_scores_and_flags() -> None:
    views = demo.variant_views(_result())

    assert [view.verdict for view in views] == ["passed", "flagged", "flagged (not scored)"]
    assert views[0].headline == "Ride in"
    assert views[0].scores == (("voice", "5"), ("clarity", "4"), ("call_to_action", "4"))
    assert views[0].overall == "4.3"
    assert views[0].fixes == ()
    assert views[1].fixes == ("Name the bike", "Drop the hype")
    assert views[2].overall == "-"
    assert views[2].scores == ()

    rows = demo.score_rows(views)
    table = demo.format_score_table(rows)
    assert rows[1]["Result"] == "flagged"
    assert rows[1]["call_to_action"] == "2"
    assert rows[2]["voice"] == "-"
    assert "| Variant | Channel | voice | clarity | call_to_action | Overall | Result |" in table
    assert "flagged (not scored)" in table
    assert "Name the bike" in demo.format_variant(views[1])
    assert demo.status_text(_result()) == "Status: partial. 3 variants, 2 flagged. Revisions: 1."
    assert "trace-abc" in demo.run_caption(_result())
    assert "tracing is off" in demo.run_caption(_result(trace_id=None))
    assert "$0.2500" in demo.run_caption(_result())
    assert "[writer] social came back short" in demo.format_errors(_result())
    assert demo.format_errors(_result().model_copy(update={"errors": []})) == ""
    assert demo.format_score_table([]) == ""


def test_streamlit_command_stays_headless_and_sends_no_usage_stats() -> None:
    command = demo.streamlit_command(demo.APP_PATH, host="127.0.0.1", port=8501)

    assert command[:4] == [command[0], "-m", "streamlit", "run"]
    assert command[4] == str(demo.APP_PATH)
    assert "--server.address" in command
    assert "127.0.0.1" in command
    assert "--server.port" in command
    assert "8501" in command
    assert "--server.headless" in command
    assert "true" in command
    assert "--browser.gatherUsageStats" in command
    assert "false" in command


def test_launch_demo_points_the_child_at_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRANDFORGE_DEMO_SENTINEL", "kept")
    seen: dict[str, Any] = {}

    def fake_call(command: list[str], *, env: dict[str, str]) -> int:
        seen["command"] = command
        seen["env"] = env
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)

    code = demo.launch_demo(host="0.0.0.0", port=8600, api_base_url="http://api.test:8000")

    assert code == 0
    assert seen["command"][4] == str(demo.APP_PATH)
    assert seen["env"]["BRANDFORGE_API_BASE_URL"] == "http://api.test:8000"
    assert seen["env"]["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] == "false"
    assert seen["env"]["BRANDFORGE_DEMO_SENTINEL"] == "kept"


def test_demo_help_points_at_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: IsolatedSettings())

    result = runner.invoke(cli.app, ["demo", "--help"])

    assert result.exit_code == 0
    assert "POST /generate" in result.stdout
    assert "brandforge serve" in result.stdout


def test_demo_binds_the_configured_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: IsolatedSettings(
            demo_host="127.0.0.1",
            demo_port=8510,
            api_base_url="http://127.0.0.1:9000",
        ),
    )
    seen: dict[str, Any] = {}

    def fake_launch(*, host: str, port: int, api_base_url: str) -> int:
        seen["host"] = host
        seen["port"] = port
        seen["api_base_url"] = api_base_url
        return 0

    monkeypatch.setattr(demo, "launch_demo", fake_launch)

    result = runner.invoke(cli.app, ["demo"])

    assert result.exit_code == 0
    assert seen == {
        "host": "127.0.0.1",
        "port": 8510,
        "api_base_url": "http://127.0.0.1:9000",
    }


def test_demo_flags_override_the_configured_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: IsolatedSettings())
    seen: dict[str, Any] = {}

    def fake_launch(*, host: str, port: int, api_base_url: str) -> int:
        seen["host"] = host
        seen["port"] = port
        seen["api_base_url"] = api_base_url
        return 0

    monkeypatch.setattr(demo, "launch_demo", fake_launch)

    result = runner.invoke(
        cli.app,
        ["demo", "--host", "0.0.0.0", "--port", "8600", "--api-url", "http://api.internal:8000/"],
    )

    assert result.exit_code == 0
    assert seen == {
        "host": "0.0.0.0",
        "port": 8600,
        "api_base_url": "http://api.internal:8000",
    }


def test_demo_rejects_a_port_outside_1_to_65535(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: IsolatedSettings())
    called: list[object] = []
    monkeypatch.setattr(demo, "launch_demo", lambda **_kwargs: called.append(True))

    result = runner.invoke(cli.app, ["demo", "--port", "0"])

    assert result.exit_code != 0
    assert called == []


def test_demo_rejects_an_api_url_with_a_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: IsolatedSettings())
    called: list[object] = []
    monkeypatch.setattr(demo, "launch_demo", lambda **_kwargs: called.append(True))

    result = runner.invoke(cli.app, ["demo", "--api-url", "http://127.0.0.1:8000/generate"])

    assert result.exit_code != 0
    assert "path" in result.output
    assert called == []


def test_demo_returns_the_launcher_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: IsolatedSettings())
    monkeypatch.setattr(demo, "launch_demo", lambda **_kwargs: 3)

    result = runner.invoke(cli.app, ["demo"])

    assert result.exit_code == 3


def _page(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setattr(
        "brandforge.config.get_settings",
        lambda: IsolatedSettings(),
    )
    app = AppTest.from_file(str(demo.APP_PATH), default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    return app


def test_page_offers_a_brand_and_a_brief_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(demo, "request_generation", lambda *_args, **_kwargs: calls.append(True))

    app = _page(monkeypatch)

    assert app.selectbox[0].label == "Brand"
    assert app.selectbox[0].options == ["brightleaf", "ledgerly", "voltride"]
    assert app.selectbox[0].value == "voltride"
    assert app.text_area[0].label == "Brief"
    brief_text = app.text_area[0].value
    assert isinstance(brief_text, str)
    assert "product:" in brief_text
    assert app.button[0].label == "Generate"
    assert "/generate" in app.caption[1].value
    assert calls == []
    assert not app.success
    assert not app.warning
    assert not app.error


def test_page_shows_variants_scores_and_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_request(
        brand_id: str,
        brief: Brief,
        *,
        api_base_url: str,
        timeout: float,
        client: httpx.Client | None = None,
    ) -> RunResult:
        del client
        seen["brand_id"] = brand_id
        seen["brief"] = brief
        seen["api_base_url"] = api_base_url
        seen["timeout"] = timeout
        return _result()

    monkeypatch.setattr(demo, "request_generation", fake_request)
    app = _page(monkeypatch)
    app.selectbox[0].set_value("ledgerly")
    app.text_area[0].set_value(PASTED).run()
    assert "brand_id" not in seen

    app.button[0].click().run()

    assert not app.exception, app.exception
    assert seen["brand_id"] == "ledgerly"
    assert seen["brief"] == demo.parse_brief(PASTED)
    assert seen["api_base_url"] == "http://127.0.0.1:8000"
    assert seen["timeout"] == 165
    assert app.warning[0].value.startswith("Status: partial.")
    assert "2 flagged" in app.warning[0].value
    rendered = "\n".join(item.value for item in app.markdown)
    assert "Ride in" in rendered
    assert "Go faster" in rendered
    assert "call_to_action" in rendered
    assert "flagged (not scored)" in rendered
    assert "Name the bike" in rendered
    headings = [item.value for item in app.subheader]
    assert "search-1 · search · passed" in headings
    assert "social-1 · social · flagged" in headings
    assert "trace-abc" in app.caption[-1].value

    app.selectbox[0].set_value("brightleaf").run()
    assert "Ride in" in "\n".join(item.value for item in app.markdown)
    assert seen["brand_id"] == "ledgerly"


def test_page_shows_a_finished_failed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(*_args: object, **_kwargs: object) -> RunResult:
        return _result(status="failed", trace_id=None)

    monkeypatch.setattr(demo, "request_generation", fake_request)
    app = _page(monkeypatch)

    app.button[0].click().run()

    assert not app.exception, app.exception
    assert app.error[0].value.startswith("Status: failed.")
    assert "Ride in" in "\n".join(item.value for item in app.markdown)
    assert "tracing is off" in app.caption[-1].value


def test_page_shows_a_brief_error_without_calling_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(demo, "request_generation", lambda *_args, **_kwargs: calls.append(True))
    app = _page(monkeypatch)
    app.text_area[0].set_value("product: Tea\n").run()

    app.button[0].click().run()

    assert not app.exception, app.exception
    assert calls == []
    assert app.error
    assert "not valid" in app.error[0].value


def test_page_shows_an_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(*_args: object, **_kwargs: object) -> RunResult:
        raise demo.DemoError(
            "Cannot reach the API at http://127.0.0.1:8000. Start it with `brandforge serve`."
        )

    monkeypatch.setattr(demo, "request_generation", fake_request)
    app = _page(monkeypatch)

    app.button[0].click().run()

    assert not app.exception, app.exception
    assert "brandforge serve" in app.error[0].value
    assert not app.success


def test_page_says_when_there_are_no_brands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(demo, "brand_choices", lambda: [])

    app = _page(monkeypatch)

    assert app.error[0].value == "No brand profiles found."
    assert not app.button
