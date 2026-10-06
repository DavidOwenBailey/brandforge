"""Maps a provider name (the prefix in `provider:model`) to its adapter.

Adapters are imported only when their provider is used, so a provider's SDK is
never imported, or even required, unless a tier points at it.
"""

from brandforge.config import Settings
from brandforge.llm.base import GatewayConfigError, ProviderAdapter

AVAILABLE_PROVIDERS = ("anthropic", "gemini")


def get_adapter(provider: str, settings: Settings) -> ProviderAdapter:
    if provider == "anthropic":
        from brandforge.llm.adapters.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter.from_settings(settings)
    if provider == "gemini":
        from brandforge.llm.adapters.gemini_adapter import GeminiAdapter

        return GeminiAdapter.from_settings(settings)
    raise GatewayConfigError(
        f"No adapter for provider {provider!r}. Available: {', '.join(AVAILABLE_PROVIDERS)}."
    )
