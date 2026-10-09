"""The optional Langfuse stack (BF-28): Cloud is the default, Compose is local-only."""

import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from pydantic_settings import SettingsConfigDict

from brandforge.config import LANGFUSE_CLOUD_HOST, LANGFUSE_LOCAL_HOST, Settings

REPO = Path(__file__).parents[1]
COMPOSE_PATH = REPO / "deploy" / "langfuse" / "docker-compose.yml"

_LANGFUSE_IMAGE = re.compile(r"docker\.langfuse\.com/langfuse/langfuse:\d+\.\d+\.\d+")
_LANGFUSE_WORKER_IMAGE = re.compile(r"docker\.langfuse\.com/langfuse/langfuse-worker:\d+\.\d+\.\d+")
_PINNED_DEPENDENCIES = {
    "clickhouse": re.compile(r"docker\.io/clickhouse/clickhouse-server:\d+\.\d+"),
    "redis": re.compile(r"docker\.io/redis:\d+\.\d+"),
    "postgres": re.compile(r"docker\.io/postgres:\d+"),
}
_DATASTORES = ("postgres", "clickhouse", "redis")


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


class _SettingsNoEnv(Settings):
    """Settings that ignore a developer's .env, so the default host is what we assert."""

    model_config = SettingsConfigDict(env_file=None)


def test_cloud_host_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    assert Settings.model_fields["langfuse_host"].default == LANGFUSE_CLOUD_HOST
    assert _SettingsNoEnv().langfuse_host == LANGFUSE_CLOUD_HOST
    assert LANGFUSE_CLOUD_HOST == "https://cloud.langfuse.com"
    assert LANGFUSE_LOCAL_HOST == "http://localhost:3000"

    example = (REPO / ".env.example").read_text(encoding="utf-8")
    assert f"LANGFUSE_HOST={LANGFUSE_CLOUD_HOST}" in example
    assert f"LANGFUSE_HOST={LANGFUSE_LOCAL_HOST}" not in example


def test_compose_is_a_local_only_langfuse_stack() -> None:
    compose = _compose()
    services = _services(compose)
    assert compose["name"] == "brandforge-langfuse"
    expected = {"langfuse-web", "langfuse-worker", "postgres", "clickhouse", "redis", "minio"}
    assert expected <= set(services)

    web = _service(services, "langfuse-web")
    worker = _service(services, "langfuse-worker")
    web_image = web["image"]
    worker_image = worker["image"]
    assert isinstance(web_image, str)
    assert isinstance(worker_image, str)
    assert _LANGFUSE_IMAGE.fullmatch(web_image)
    assert _LANGFUSE_WORKER_IMAGE.fullmatch(worker_image)
    assert web_image.rsplit(":", 1)[-1] == worker_image.rsplit(":", 1)[-1]

    for name, pattern in _PINNED_DEPENDENCIES.items():
        image = _service(services, name)["image"]
        assert isinstance(image, str)
        assert pattern.fullmatch(image), image
    # Chainguard's MinIO image has no semver tag; the compose file says why.
    assert _service(services, "minio")["image"] == "cgr.dev/chainguard/minio"
    assert "Chainguard" in COMPOSE_PATH.read_text(encoding="utf-8")

    ports = web["ports"]
    assert isinstance(ports, list)
    assert any("127.0.0.1" in str(port) and "3000" in str(port) for port in ports)
    for name in _DATASTORES:
        assert "ports" not in _service(services, name)
    for name in (*_DATASTORES, "minio"):
        assert "healthcheck" in _service(services, name)

    environment = worker["environment"]
    assert isinstance(environment, dict)
    telemetry = environment["TELEMETRY_ENABLED"]
    assert isinstance(telemetry, str)
    assert "false" in telemetry

    volumes = compose["volumes"]
    assert isinstance(volumes, dict)
    for name in (
        "langfuse_postgres_data",
        "langfuse_clickhouse_data",
        "langfuse_minio_data",
        "langfuse_redis_data",
    ):
        assert name in volumes

    text = COMPOSE_PATH.read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY" not in text
    assert ":latest" not in text


def test_self_hosting_is_documented_as_optional() -> None:
    doc = (REPO / "docs" / "langfuse.md").read_text(encoding="utf-8")
    assert LANGFUSE_CLOUD_HOST in doc
    assert LANGFUSE_LOCAL_HOST in doc
    assert "deploy/langfuse/docker-compose.yml" in doc

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "deploy/langfuse/docker-compose.yml" in readme
    assert LANGFUSE_CLOUD_HOST in readme

    adr = (REPO / "docs" / "adr" / "0022_self_hosted_langfuse.md").read_text(encoding="utf-8")
    assert "**Status:** Accepted" in adr
    assert "deploy/langfuse/docker-compose.yml" in adr
