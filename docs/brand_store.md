# Brand store

The brand store holds each brand's voice, rules and scoring rubric as versioned YAML files in the repo. It is the single source of truth for "what on-brand means" in BrandForge. Agents and evals read from it and never define brand rules themselves.

## What it is

- One YAML file per brand in `src/brandforge/brands/`: `brightleaf.yaml`, `voltride.yaml`, `ledgerly.yaml`.
- Each file validates into the `BrandProfile` Pydantic model (`models.py`).
- A small loader (`brands/loader.py`) is the only way the rest of the code reads them.

### File shape

| Field | Purpose |
| :---- | :---- |
| `id` | Stable identifier. Must match the filename. |
| `version` | Brand version, e.g. `"1.0"`. |
| `voice` | Short voice descriptors ("warm", "plain-spoken"). |
| `do` / `dont` | Positive and negative style rules for the writer. |
| `banned_words` | Words that must never appear in copy. |
| `rubric` | Has its own `version` plus scoring criteria (`voice`, `clarity`, `call_to_action`), each with concrete anchors for scores 1 to 5. |

All contracts reject unknown fields, so a misspelled key fails at load time instead of being silently ignored.

## How it is used

```python
from brandforge.brands import load_brand, list_brand_ids

brand = load_brand("brightleaf")  # returns a validated BrandProfile
list_brand_ids()  # ["brightleaf", "ledgerly", "voltride"]
```

`load_brand` raises `BrandLoadError` for an unknown brand, invalid YAML, a schema violation or an id/filename mismatch. Unknown ids are checked against the directory listing, which also blocks path tricks such as `../x`.

## Where it fits in the architecture

The CLI, API and eval harness load the brand once at the start of a run and put it in `RunState.brand`. After that, agents read it from state and never touch the files.

| Component | What it takes from the brand |
| :---- | :---- |
| Planner | `brand` to shape audience, angle and channel plan |
| Retriever | `brand.id` to pick the right example collection |
| Writer / Reviser | `voice`, `do`, `dont`, `banned_words` to steer the copy |
| Brand critic | `rubric` to score each variant 1 to 5 per criterion |
| Assembler | `brand.version` and `rubric.version`, recorded in the result |
| Evals | `banned_words` for deterministic assertions; `rubric` anchors for the judge prompt |

Because every consumer reads the same profile, changing a brand changes the writer, critic and evals together, so they can't drift apart.

## Versioning

The store uses Git history for versioning (Option A):

- There is one current version per brand. Editing a file means bumping `version` (and `rubric.version` if the rubric changed) and committing.
- Every run result and eval report records `brand.version` and `rubric.version`, so any result can be traced to the exact profile that produced it.
- To reproduce an older run, check out the matching commit or release tag (for example `git switch --detach v0.6.0`) and re-run.
- Limitation: two versions of one brand can't be loaded side by side in the same process.

## Design decisions

- **YAML in the repo, not a database.** Profiles are reviewable in pull requests and need no infrastructure (ADR 0007, local-first).
- **Validation at the boundary.** A malformed profile fails at load time with a clear message and never reaches an agent.
- **Anchored rubrics.** Each score level is described concretely, which supports the judge-bias controls in the evaluation design.
- **Versioning by Git.** This is the simplest option that still gives reproducible results within the one-week scope.

## Limits and path to production

- Not multi-tenant, with no editing UI and no side-by-side version loading.
- Production would move profiles to a database (Postgres) with per-tenant isolation, and could load a specific version as `load_brand(id, version=...)`. That would be a folder per brand with one file per version.
