# Guardrails

Use this file for non-negotiable governance constraints.

## MUST
- Keep public output compliant with `governance/publication_policy.md`.
- Block any change that includes real work data, private identifiers, incident notes, secrets, or PII.
- Keep `governance/publish_checklist.md` aligned with actual verification status.
- Require evidence for changed behavior (tests, artifacts, or both).
- Include risk and rollback notes for non-trivial PRs.
- Mark output `NOT READY` if any required gate is missing.

## SHOULD
- Keep PRs scoped to one primary outcome.
- Keep diffs small enough for clear review.
- Pair behavior changes with docs and tests when practical.
- Preserve portfolio-safe naming and externally readable language.

## MAY
- Propose scope splits into stacked PRs when review risk is high.
- Suggest tighter naming, release lines, or evidence packaging for clarity.
