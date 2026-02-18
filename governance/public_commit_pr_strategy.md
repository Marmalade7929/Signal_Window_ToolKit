# Public Commit and PR Strategy

This standard maximizes public signal, reviewer trust, and portfolio visibility.

## Goals
- Make each PR tell one clear story.
- Make each commit easy to understand and reuse in release notes.
- Make evidence visible: tests, demos, and risk controls.
- Keep public output aligned with the publication policy.

## Branch Strategy
- Use short-lived branches from `main` in the public repo.
- Name branches by change type and scope:
  - `feat/<area>-<brief-topic>`
  - `fix/<area>-<brief-topic>`
  - `docs/<area>-<brief-topic>`
  - `chore/<area>-<brief-topic>`

## Commit Strategy
Use Conventional Commit style:
- Format: `<type>(<scope>): <summary>`
- Keep summary in imperative mood and under 72 chars.
- Prefer 1 to 4 purposeful commits per PR.

Allowed types:
- `feat`: user-visible capability or product value
- `fix`: bug or correctness improvement
- `perf`: measurable runtime/memory improvement
- `refactor`: structure change with no behavior change
- `test`: test-only changes
- `docs`: documentation changes
- `chore`: maintenance/infrastructure

Recommended scopes:
- `characterization`, `detection`, `prediction`, `calibration`, `ui`, `docs`, `ci`, `governance`, `export`

Commit ordering:
1. Foundation/internal plumbing (if needed)
2. Behavior/API change
3. Tests and docs
4. Cleanup follow-up

## Pull Request Strategy
Each PR should represent one primary outcome.

PR sizing guidance:
- Target <= 400 net changed lines (excluding generated files).
- Split larger work into stacked PRs.

Required PR narrative:
1. Problem and why it matters
2. What changed (high level)
3. Evidence (tests, screenshots, benchmarks, artifacts)
4. Risks and rollback
5. Release note line

## Visibility and Signal Heuristics
Prefer changes that include:
- User-visible result or measurable engineering gain
- Test evidence tied to changed behavior
- Before/after artifact (chart, screenshot, metrics, or log)
- Concise release-ready summary

Avoid:
- Mixed unrelated concerns in one PR
- Long PR bodies without evidence
- “Refactor only” PRs without rationale or measured benefit

## Review Gates
All PRs must satisfy:
- CI green
- Required tests added or updated
- Docs updated when behavior changes
- Publication policy respected (no prohibited content)
- Publish checklist remains valid

## Label and Merge Policy
Recommended labels:
- `type:feature`, `type:fix`, `type:docs`, `type:chore`
- `area:model`, `area:ui`, `area:ci`, `area:governance`
- `risk:low`, `risk:medium`, `risk:high`

Merge strategy:
- Use **Squash and merge** for small PRs that form one clear story.
- Use **Rebase and merge** for curated multi-commit PRs where commit narrative matters.

## Release Signal
For each merged PR, capture one release line:
- `Added ...`
- `Improved ...`
- `Fixed ...`

Keep wording externally readable and portfolio-safe.

