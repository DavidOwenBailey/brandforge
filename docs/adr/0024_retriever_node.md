# 0024. Retriever: nearest examples per channel, and an empty index continues

- **Status:** Accepted
- **Date:** 2026-10-09

## Context

The writer already accepts approved examples and omits that section of the prompt when the
list is empty (0011). The index holds one Chroma collection per brand (0023). The graph still
goes from the planner straight to the writer, so `examples` stays empty. The architecture
puts a retriever between them: embeddings only, the 3–5 nearest approved ads for this brand
and channel, and a run that finds nothing continues with a warning.

An empty result has to stay a warning. The assembler marks a run `partial` when any
`RunError` is recorded, including a non-fatal one. Treating "no index yet" as an error would
make every run before `brandforge index` look partial, even when every variant passed.

## Decision

`retrieve_examples(state)` is the graph node. The graph is now planner, retriever, writer,
critic, then the router. The retriever is guarded like the other model nodes (0016): an
exception becomes a fatal `RunError` and the run goes to the assembler with the plan. An
empty result is not an exception.

- **What it reads.** `brand.id` selects the collection. The plan supplies the channels, the
  angle and the audience. The brief supplies the product and the constraints, which the plan
  does not carry. The query text is the same three lines as an indexed ad (`Headline`,
  `Body`, `CTA` in `example_document`), so the local model compares an ad with an ad.
- **One search per channel.** Each channel in the plan gets its own nearest
  `retrieval_examples_per_channel` examples, filtered on the `channel` metadata. The default
  is 4, and the setting allows 1–5, which is the architecture's "3–5" with room at either
  end for an eval. Results keep the plan's channel order, and within a channel they stay in
  nearest-first order. The collection is opened without a new embedding function, so the
  query uses the function stored when the index was built (0023).
- **Empty is a warning.** No index file, no collection for the brand, or no usable ad for
  these channels returns `examples: []`, logs `retrieval_empty` with a `reason`
  (`no_index`, `no_collection`, `no_matches`), and does not record a `RunError`. The writer
  then runs with no examples section. A missing index does not create the Chroma directory.
  A record that cannot be rebuilt into an `Example` is skipped and logged; it does not fail
  the search.
- **No usage.** Embeddings are local and free, so the node returns no `usage` key and does
  not appear in the cost report.
- **`retrieval_enabled` is not read.** That flag is the with/without comparison (BF-32). This
  node always searches. The flag stays in config, default on, for that later change.

## Alternatives considered

- **One global top-k, then split by channel.** One embedding instead of one per channel (at
  most four). A channel the query happens to sit far from would get nothing while another
  filled the list. The architecture asks for the nearest examples for the channel, so each
  channel is searched on its own. The repeated local embed is a short string.
- **A `RunError` when the index is missing.** It would show up in the CLI summary. It would
  also force `partial` on a run whose variants all passed. The log warning is the signal the
  architecture asks for; the result stays `complete` when the copy passes.
- **Fail the run when the index is missing.** `brandforge generate` would not work on a fresh
  clone until someone ran `brandforge index`. The writer was built to run without examples
  for this reason.
- **An LLM-written search query.** Closer paraphrases, and a model call the retriever is not
  supposed to make. The tier for this node is embeddings only.
- **Honour `retrieval_enabled` here.** Small, but it is a separate done-when (BF-32) so evals
  can compare the two arms. Wiring it in this change would hide that comparison inside the
  node.

## Consequences

- **Gained:** a run with an index writes the nearest approved examples into state, per
  channel, and the writer sees them with no change of its own. A run without an index still
  finishes, and the log says why the list was empty.
- **Trade-off accepted:** the node reads the brief as well as `brand.id` and the plan. The
  plan has no product field, and the product is what the examples are about.
- **Trade-off accepted:** four channels embed the same query four times. The text is short
  and the model is local.
- **Trade-off accepted:** `BRANDFORGE_RETRIEVAL_ENABLED=false` does nothing until BF-32.
  The retriever cannot be switched off by config yet.
- **Trade-off accepted:** a retriever that raises (a corrupt index, not an empty one) ends
  the run `failed` with the plan kept and no variants. That is the same error edge as a
  failed writer, and it is a different case from an empty result.
