"""Versioned prompt files: prompts/<name>_<version>.md, filled with {{placeholders}}."""

import re
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
_SAFE_PART = re.compile(r"[A-Za-z0-9_.-]+")


def load_prompt(name: str, version: str, prompts_dir: Path = PROMPTS_DIR) -> str:
    """Read prompts/<name>_<version>.md. The version comes from config, so it is checked."""
    if not _SAFE_PART.fullmatch(name) or not _SAFE_PART.fullmatch(version):
        raise ValueError(f"Invalid prompt name or version: {name!r}, {version!r}")
    path = prompts_dir / f"{name}_{version}.md"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path.name}")
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, **values: str) -> str:
    """Fill {{placeholders}} in one pass, so a substituted value is never re-scanned."""

    def _fill(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"Missing prompt value: {key}")
        return values[key]

    return _PLACEHOLDER.sub(_fill, template)
