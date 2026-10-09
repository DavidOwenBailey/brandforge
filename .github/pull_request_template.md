# Pull Request

## Description
<!-- Provide a clear summary of what this PR does and why it is needed. -->

## Linked Issues & Milestones
<!-- Use keywords like "Closes #123" to automatically close issues upon merge. -->
* **Closes:** #
* **Related to:** #
* **Milestone:** <!-- Select from the GitHub sidebar or link here -->

## What changed
<!-- Bullet points detailing specific technical changes or additions. -->
* 
* 
* 

## Visual Content
<!-- If this PR changes the UI, include screenshots, GIFs, or wireframes. Delete section if not applicable. -->
| Before | After |
| :--- | :--- |
| <!-- Image URL or drag-and-drop here --> | <!-- Image URL or drag-and-drop here --> |

## How it was checked
<!-- Mark completed items with an [x] -->
- [ ] Surprising decisions are captured in an ADR
- [ ] New prompts (if any) are versioned files; new config values are in .env.example or config.py
- [ ] New logic (if any) has unit tests; LLM calls are mocked (no real calls in CI except the eval smoke test)
- [ ] Manually tested local environment behaviors (i.e. `brandforge generate`).
- [ ] CI is green: run `uv run poe check` to ensure Ruff, mypy strict and pytest all pass