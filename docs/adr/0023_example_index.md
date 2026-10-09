# 0023. Local ONNX embeddings for the per-brand example index

- **Status:** Accepted
- **Date:** 2026-10-09

## Context

The retriever (BF-31) needs approved ads it can search. The architecture puts those ads in a
local persistent Chroma store, one example index per brand, and names the embedding source
"provider embeddings". The corpus is about 45 short fictional ads (BF-29). The default model
provider is Anthropic, which has no embeddings API. A fresh clone is documented as one API
key, `ANTHROPIC_API_KEY`, and the index has to be rebuildable from the YAML in git.

## Decision

`brandforge index` reads the corpus and writes one Chroma collection per brand.

- **Where.** `settings.chroma_dir`, default `.brandforge/chroma`, overridable with
  `BRANDFORGE_CHROMA_DIR`. The directory is git-ignored with the rest of `.brandforge/`.
- **Collection name** is the brand id (`brightleaf`, `ledgerly`, `voltride`).
- **Embeddings** are Chroma's default local model, the ONNX build of
  `all-MiniLM-L6-v2` (384 dimensions, registered under the name `default`). Constructing
  the function does not download anything. The model files download on the first real
  index, into `~/.cache/chroma/onnx_models/`, not into the repo. The collection records
  that function, so a later open resolves the same one.
- **One record per ad.** The embedded text is the headline, the body and the call to
  action. The channel is metadata, with `brand_id`, `headline`, `body` and `cta`, so a
  query can filter on channel and rebuild an `Example` without parsing the document.
  The id is `{brand_id}-{position:04d}` from the order in the corpus file.
- **Distance** is cosine. Chroma L2-normalises the vectors.
- **A second run replaces the collection.** The corpus file is the source of truth. An
  upsert would leave an ad that had been deleted from the file.
- **Telemetry is off** on the client. The index is local and has nothing to report.
- **`retrieval_enabled` does not gate this command.** That flag is for the retriever
  comparison (BF-32). The index can be built either way.
- **Tests pass their own embedding function.** The suite does not download the model and
  does not call a provider. The function is an argument, not a config switch.

The model's window is 256 tokens. Every approved ad is a few sentences, so the window
holds the whole ad.

## Alternatives considered

- **Gemini embeddings (`gemini-embedding-001`), or OpenAI embeddings.** This is the
  reading of "provider embeddings" in the architecture table, and `google-genai` is
  already a dependency. It would make `brandforge index` fail for the documented
  one-key setup, because Anthropic does not embed, and it would spend a provider call
  on 45 short ads. The injected embedding function is the seam if a later eval says the
  local model is not good enough. Re-indexing is required after a change of model,
  because the stored vectors would no longer match the query vectors.
- **One collection for every brand, filtered by `brand_id`.** Fewer files, and a
  mistake in the filter would mix two brands' voices. The done-when is a collection
  per brand, which is also how the retriever will open the index: by brand id.
- **Leave the vectors in place and upsert.** A removed ad would stay searchable until
  someone remembered to delete its id. The corpus is small enough to rewrite.

## Consequences

- **Gained:** `brandforge index` runs with no API key. Opening the collection later uses
  the same embedding function that built it. The record metadata is enough for the
  retriever to return `Example` values and to filter by channel.
- **Trade-off accepted:** the first real index downloads the MiniLM ONNX model from
  Chroma's bucket. A machine with no network can index only after that download has
  succeeded once.
- **Trade-off accepted:** MiniLM is a general English model. Whether retrieved examples
  improve the copy is the with/without comparison in BF-32, not something this index
  claims.
- **Trade-off accepted:** provider embeddings, as named in the architecture, wait until
  that comparison says the local model is the wrong tool.
- **Trade-off accepted:** if the write fails after the old collection has been deleted,
  that brand has no collection until the next successful run. The corpus is the copy of
  record, the command is re-runnable, and the retriever is specified to continue when
  it finds no examples.
