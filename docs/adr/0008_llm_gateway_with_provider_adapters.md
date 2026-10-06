# 0008. Call models through our own gateway, with one adapter per provider

- **Status:** Accepted
- **Date:** 2026-10-06

## Context

Every agent needs a model, and the architecture says agents never call a provider
directly: the router owns control flow, and every model call must be retried,
budgeted, costed and traced in one place. BrandForge must support at least Claude
and Gemini, and possibly OpenAI later, so the gateway cannot be coupled to one
provider's SDK or response shapes.

Anthropic offers several layers: the Messages API (through the `anthropic` client
SDK), the Claude Agent SDK, and Managed Agents. MCP is also in the ecosystem, but it
lets a model reach tools and data; it is not a way to call a model. Every
BrandForge node is a single text-in, structured-out call. None needs file, shell or
web tools.

## Decision

Agents call `brandforge.llm.gateway.complete_structured(prompt, schema, tier)`,
which returns a validated Pydantic instance and its `Usage`. Behind it are two layers.

**Gateway core (provider-neutral, imports no provider SDK).** It does everything
that must behave identically for every provider:

- resolves the tier to `provider:model` from config and picks the adapter;
- computes cost from per-tier prices in config;
- validates the reply with Pydantic and raises a typed `StructuredOutputError`
  carrying the raw text, usage and validation error, which the schema-repair retry
  (BF-21) needs;
- refuses response schemas that are not portable across providers (free-form
  `dict` fields);
- will own retries, budgets and tracing (BF-20, BF-23, BF-25), written once.

**Provider adapters (one per provider).** An adapter implements `ProviderAdapter`:
it builds that provider's request, including its structured-output setting, calls
its SDK and returns a `RawCompletion` (text, token counts and a normalised
outcome: complete, truncated or refused). It does not validate, cost, retry or
trace. Its SDK is imported only inside the adapter, and only when a tier points at
that provider.

Two adapters exist. Anthropic uses the Messages API with a JSON-schema output config
and the SDK's own retries switched off (`max_retries=0`). Gemini uses
`generate_content` with `response_json_schema`; the SDK does not retry unless asked,
so none is configured. In both, retry policy lives in the core. OpenAI is added only
if time allows. Adding a provider means an adapter module, a branch in
`brandforge.llm.registry`, an API-key setting and per-tier prices.

## Alternatives considered

- **Claude Agent SDK:** brings its own agent loop, tools and permissions. That is a
  second orchestrator inside every node and conflicts with LangGraph owning control
  flow (0001) and with bounded, testable loops. It is also Claude-only.
- **Managed Agents:** server-hosted agents with a managed sandbox. Same conflict,
  plus less visibility into per-call tokens and cost.
- **`messages.parse(output_format=...)`:** convenient, but validation happens inside
  the SDK and only for Claude, which makes it harder to hand the raw output to a
  repair retry in a provider-neutral way.
- **MCP:** solves a different problem (exposing tools to a model). Nothing in the
  pipeline needs external tools.
- **Multi-provider libraries (LiteLLM, LangChain chat models):** give provider
  coverage quickly, but hide the retry, usage and structured-output behaviour that
  BF-20, BF-21 and BF-23 need to control, and add a second abstraction over the same
  call. Revisit if the provider count grows beyond what thin adapters can carry.
- **A gateway coupled to the Anthropic SDK, refactored later:** rejected. Gemini is
  required, and retrofitting would mean writing retry and error handling twice.

## Consequences

- **Gained:** one place for validation, cost, retries, budgets and tracing; agents
  stay pure functions; a provider's SDK is imported only when a tier uses it; the core is tested with a fake adapter and each adapter with a fake
  SDK client, so no test needs a network or a key.
- **Trade-off accepted:** we own the glue that an agent framework or library would
  provide, and adapters must be kept behaviourally aligned. Langfuse tracing will
  hook into the core (BF-25) rather than LangChain callbacks.
- **Known limitation:** only the Anthropic and Gemini adapters exist. A tier set to
  `openai` raises `GatewayConfigError` listing the available providers.
- **Known limitation:** response schemas must use lists of items, never free-form
  `dict` fields, because strict structured-output modes cannot express unknown keys
  (for example `Critique.scores`). The core enforces this for every provider. The
  critic (BF-15) uses a list of `{criterion, score}` items and converts it in code.
  Further per-provider schema limits are handled inside each adapter: Anthropic's
  transform errors become `GatewayConfigError`, and the Gemini adapter reduces the
  schema to the keywords Gemini supports (dropping, for example, `minLength` and
  `default`) because the core validates every reply anyway. OpenAI's limits have not
  been verified yet.
- **Known limitation:** only input and output tokens are priced. Gemini thinking
  tokens are counted as output, which is how Gemini bills them. Prompt-cache pricing
  differs by provider and arrives with BF-27.
- **Known limitation:** provider SDK errors pass through unchanged until BF-20
  maps them to gateway errors inside each adapter.
