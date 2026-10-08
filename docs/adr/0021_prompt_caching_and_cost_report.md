# 0021. Cache the static prompt prefix, and price it per provider

- **Status:** Accepted
- **Date:** 2026-10-08

## Context

The same instructions and brand profile are sent on every model call in a run: once per channel
for the writer, once per variant for the critic, once per failing variant for the reviser. The
architecture asks for that static prefix to be cached, and for the CLI to show the cost of a run
broken down by node. Anthropic and Gemini both cache a repeated prefix, but they do not bill it
the same way and they do not report it in the same fields (ADR 0008 left this to BF-27).

A run's usage is one number, accumulated by a reducer. That is enough for a total and not enough
for a table with one row per node.

## Decision

**The static prefix is the system prompt.** Each agent prompt from v2 has one cache break, the
line `<!-- cache -->`. Above it are the instructions and the brand profile (and the rubric,
where the agent uses one). Below it is what changes per call: the brief, the channel, the
variant, the examples. The break is applied to the template before placeholders are filled, so
text inside the brief cannot move it, and the marker is never sent to the model. v1 is the same
words with no break, kept so a config change turns caching off. The repair prompt has no break
of its own: a repair resends the original system prompt and only replaces the user text.

**Each adapter asks for the cache in that provider's way, and reports tokens in one shape.**

- Anthropic: the system prompt is one text block with `cache_control: {type: ephemeral}`, the
  5-minute cache. The breakpoint is on that block and not on the user message. A breakpoint on
  the user message would write a new entry every call and never read one, because the user
  message changes. `input_tokens` stays what Anthropic reports (tokens after the breakpoint);
  `cache_creation_input_tokens` and `cache_read_input_tokens` become cache writes and reads.
  A prefix shorter than the model's minimum is billed as ordinary input. The API does not error.
- Gemini: the same text is the `system_instruction`. Gemini has no per-request breakpoint on
  `generate_content`. Implicit caching (on for 2.5 and newer) discounts a repeated prefix when
  it is long enough, and reports the hit as `cached_content_token_count`. That count is included
  in `prompt_token_count`, so the adapter subtracts it and reports it as cache reads. There is
  no write counter: a miss is ordinary input. Explicit caches are not created. They have a
  minimum of 1,024 to 4,096 tokens, which a brand profile does not reach, and they add a storage
  charge the response does not itemise.

**The gateway prices what the adapter reports.** Cache writes and reads are multipliers on the
tier's input price. The multipliers live on the provider, because that is where the two bills
differ:

- Anthropic, 5-minute cache: write 1.25x, read 0.1x.
- Gemini implicit cache: read 0.1x (a 90% discount), write 1x. The write rate is unused while
  Gemini reports no write tokens.

Anthropic's read discount is not one number. Sonnet 5.5 and Opus 5.5 publish 0.05x, and Haiku
4.5 publishes 0.1x. The strong and judge tiers therefore set `cache_read_multiplier` to 0.05,
and that override wins over the provider rate. Pointing one of those tiers at Gemini, or at
Haiku, without clearing the override keeps the 0.05x read. An unknown provider is billed at 1x
for both, the full input price. Output tokens are never cached. Cache tokens count toward
`total_tokens`, and so toward the run's token budget.

**The CLI prints one row per node.** The graph stamps the usage a node returns with that node's
name, including what a failed node had already spent. The reducer sums a node that runs again,
so two critic passes are one row. A node that spent nothing is not listed. The total line gains
cache write and cache read counts only when they are non-zero.

## Alternatives considered

- **Automatic caching on the whole Anthropic request:** the breakpoint lands on the last block,
  which is the user message. That message changes every call, so every call pays a cache write
  and none of them read. Anthropic's own docs call this out.
- **One explicit Gemini cache per brand:** a guaranteed discount, but the prefix is below the
  minimum, and the storage charge cannot be read back from the `generate_content` response, so
  the cost report would be wrong.
- **A separate `node_usage` field on the run state:** works, but usage is already the thing the
  reducer sums. A list of per-node rows on `Usage` travels with the total into the result, the
  checkpoints and the trace without a new state key.
- **One cache rate for every Anthropic model:** simpler, and wrong for the two default tiers
  that publish 0.05x. The override is the smaller exception.

## Consequences

- **Gained:** repeated calls in a run reuse the brand profile instead of paying full input price
  for it, and `brandforge generate` shows which node the money went to. The same numbers are on
  the Langfuse generation.
- **Trade-off accepted:** a prefix that is sent only once (the planner, a single-channel writer)
  pays Anthropic's 1.25x write and gets no read. It pays for itself on the next call within five
  minutes, which is the critic and, across a batch of briefs, the next run of the same brand.
- **Trade-off accepted:** Gemini's discount is best-effort. Below the model's minimum, or when
  the prefix does not match, the call is billed as ordinary input and the report says so,
  because the cached count is zero.
- **Trade-off accepted:** the strong and judge read override does not follow the provider. Change
  it when the tier's model changes, the same way the input and output prices are changed.
- **Not covered here:** a 1-hour Anthropic cache (writes at 2x), Gemini explicit-cache storage,
  and a cache on/off switch beyond choosing the v1 prompt. The repair prompt is still v1.
