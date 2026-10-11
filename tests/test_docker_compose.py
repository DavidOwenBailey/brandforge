"""The app Compose file (BF-41): one image, the API and the demo, one API key."""

from pathlib import Path
from typing import Any, cast

import yaml

REPO = Path(__file__).parents[1]
COMPOSE_PATH = REPO / "docker-compose.yml"
DOCKERFILE_PATH = REPO / "Dockerfile"
DOCKERIGNORE_PATH = REPO / ".dockerignore"

_API_COMMAND = ["brandforge", "serve", "--host", "0.0.0.0", "--port", "8000"]
_DEMO_COMMAND = [
    "brandforge",
    "demo",
    "--host",
    "0.0.0.0",
    "--port",
    "8501",
    "--api-url",
    "http://api:8000",
]


def _compose() -> dict[str, Any]:
    loaded = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return cast(dict[str, Any], loaded)


def _services(compose: dict[str, Any]) -> dict[str, Any]:
    services = compose["services"]
    assert isinstance(services, dict)
    return cast(dict[str, Any], services)


def _service(services: dict[str, Any], name: str) -> dict[str, Any]:
    service = services[name]
    assert isinstance(service, dict)
    return cast(dict[str, Any], service)


def _env(service: dict[str, Any]) -> dict[str, Any]:
    environment = service["environment"]
    assert isinstance(environment, dict)
    return cast(dict[str, Any], environment)


def _command_text(service: dict[str, Any]) -> str:
    command = service["command"]
    assert isinstance(command, list)
    return " ".join(str(part) for part in command)


def _health_text(service: dict[str, Any]) -> str:
    healthcheck = service["healthcheck"]
    assert isinstance(healthcheck, dict)
    return " ".join(str(part) for part in cast(list[object], healthcheck["test"]))


def _dockerignore_lines() -> set[str]:
    lines: set[str] = set()
    for raw in DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.add(line)
    return lines


def test_compose_starts_the_api_and_the_demo_from_one_local_image() -> None:
    compose = _compose()
    services = _services(compose)
    assert compose["name"] == "brandforge"
    assert set(services) == {"api", "demo"}

    for name in ("api", "demo"):
        service = _service(services, name)
        assert service["image"] == "brandforge:local"
        assert service["build"] == "."
        assert service["pull_policy"] == "build"
        assert ":latest" not in str(service["image"])

    api = _service(services, "api")
    demo = _service(services, "demo")
    assert api["command"] == _API_COMMAND
    assert demo["command"] == _DEMO_COMMAND

    depends = demo["depends_on"]
    assert isinstance(depends, dict)
    api_dep = depends["api"]
    assert isinstance(api_dep, dict)
    assert api_dep["condition"] == "service_healthy"


def test_published_ports_stay_on_loopback() -> None:
    services = _services(_compose())
    api = _service(services, "api")
    demo = _service(services, "demo")
    assert api["ports"] == ["127.0.0.1:8000:8000"]
    assert demo["ports"] == ["127.0.0.1:8501:8501"]

    published = [str(port) for port in [*api["ports"], *demo["ports"]]]
    assert all(port.startswith("127.0.0.1:") for port in published)
    assert all("0.0.0.0:" not in port for port in published)

    assert _env(api)["BRANDFORGE_API_HOST"] == "0.0.0.0"
    assert _env(api)["BRANDFORGE_API_PORT"] == "8000"
    assert _env(demo)["BRANDFORGE_DEMO_HOST"] == "0.0.0.0"
    assert _env(demo)["BRANDFORGE_DEMO_PORT"] == "8501"
    assert _env(demo)["BRANDFORGE_API_BASE_URL"] == "http://api:8000"


def test_one_env_file_supplies_the_only_required_key() -> None:
    services = _services(_compose())
    text = COMPOSE_PATH.read_text(encoding="utf-8")
    for name in ("api", "demo"):
        env_file = _service(services, name)["env_file"]
        assert isinstance(env_file, list)
        assert {"path": ".env", "required": True} in env_file

    # The key is injected from .env at run time. It is not a Compose default.
    assert "ANTHROPIC_API_KEY=" not in text
    assert "GEMINI_API_KEY" not in text
    assert "OPENAI_API_KEY" not in text
    assert "sk-" not in text
    assert "langfuse-web" not in text
    assert "deploy/langfuse" in text


def test_api_data_is_a_named_volume_the_demo_does_not_mount() -> None:
    services = _services(_compose())
    api = _service(services, "api")
    demo = _service(services, "demo")
    assert api["volumes"] == ["brandforge_data:/app/.brandforge"]
    assert "volumes" not in demo
    assert _compose()["volumes"] == {"brandforge_data": None}
    assert _env(api)["BRANDFORGE_CHECKPOINT_DB"] == "/app/.brandforge/checkpoints.sqlite"
    assert _env(api)["BRANDFORGE_CHROMA_DIR"] == "/app/.brandforge/chroma"

    for name in ("api", "demo"):
        extra = _service(services, name)["extra_hosts"]
        assert extra == ["host.docker.internal:host-gateway"]


def test_healthchecks_do_not_call_the_model() -> None:
    services = _services(_compose())
    api_check = _health_text(_service(services, "api"))
    demo_check = _health_text(_service(services, "demo"))
    assert "openapi.json" in api_check
    assert "/generate" not in api_check
    assert "_stcore/health" in demo_check
    assert "/generate" not in demo_check
    assert "/generate" not in _command_text(_service(services, "api"))
    assert "/generate" not in _command_text(_service(services, "demo"))


def test_dockerfile_pins_uv_and_keeps_secrets_out_of_the_image() -> None:
    text = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "FROM ghcr.io/astral-sh/uv:0.12.23-python3.12-trixie-slim" in text
    assert ":latest" not in text
    assert "libgomp1" in text
    assert "UV_PYTHON_DOWNLOADS=0" in text
    assert "uv sync --locked --no-install-project" in text
    assert "uv sync --locked --no-editable" in text
    assert "ENTRYPOINT []" in text
    assert "USER brandforge" in text
    assert text.index("uv sync --locked --no-editable") < text.index("USER brandforge")
    assert "ANTHROPIC_API_KEY" not in text
    assert "COPY .env" not in text
    assert 'CMD ["brandforge", "serve", "--host", "0.0.0.0", "--port", "8000"]' in text


def test_dockerignore_excludes_secrets_and_local_data() -> None:
    ignored = _dockerignore_lines()
    assert ".env" in ignored
    assert ".venv" in ignored
    assert ".brandforge" in ignored
    assert ".git" in ignored
    assert "src" not in ignored
    assert "pyproject.toml" not in ignored
    assert "uv.lock" not in ignored


def test_docker_quickstart_is_documented() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "docker compose up" in readme
    assert "ANTHROPIC_API_KEY" in readme
    assert "127.0.0.1:8501" in readme

    example = (REPO / ".env.example").read_text(encoding="utf-8")
    assert "docker compose up" in example
    assert "http://api:8000" in example

    adr = (REPO / "docs" / "adr" / "0034_docker_compose.md").read_text(encoding="utf-8")
    assert "**Status:** Accepted" in adr
    assert "127.0.0.1" in adr
    assert "ANTHROPIC_API_KEY" in adr

    architecture = (REPO / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert "docker-compose.yml" in architecture
    assert "ADR 0034" in architecture

    langfuse = (REPO / "docs" / "langfuse.md").read_text(encoding="utf-8")
    assert "host.docker.internal" in langfuse
