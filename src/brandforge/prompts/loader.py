"""Versioned prompt files: prompts/<name>_<version>.md, filled with {{placeholders}}.

A template may contain one cache break, the line `<!-- cache -->`. Everything above it is the
static prefix (the instructions and the brand profile) and everything below it changes per call.
The break is stripped before the text is sent. It is applied to the template, not to the
rendered text, so a brief that happens to contain the marker cannot move the boundary.
"""

import re
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent

# The cache break. Exact text, on its own where the author puts it. Not sent to the model.
CACHE_MARKER = "<!-- cache -->"

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


@dataclass(frozen=True, slots=True)
class PromptParts:
    """A rendered prompt split into the cached system prefix and the per-call user text.

    `system` is None when the template has no cache break. Neither side contains the marker.
    """

    system: str | None
    user: str

    @property
    def text(self) -> str:
        """Both sides joined, for a caller that still wants one string. No marker."""
        if self.system is None:
            return self.user
        return f"{self.system}\n\n{self.user}"


def render_prompt_parts(template: str, **values: str) -> PromptParts:
    """Render `template`, splitting on the cache break when it has one.

    A template with no break is entirely the user text, unchanged, including its trailing
    newline. A break must have text on both sides. More than one break is an error: the
    author marks exactly one boundary.
    """
    if template.count(CACHE_MARKER) > 1:
        raise ValueError("A prompt may contain at most one cache break.")
    if CACHE_MARKER not in template:
        return PromptParts(system=None, user=render_prompt(template, **values))
    system_template, _, user_template = template.partition(CACHE_MARKER)
    system = render_prompt(system_template, **values).strip()
    user = render_prompt(user_template, **values).strip()
    if not system or not user:
        raise ValueError("A cache break must have text on both sides.")
    return PromptParts(system=system, user=user)
