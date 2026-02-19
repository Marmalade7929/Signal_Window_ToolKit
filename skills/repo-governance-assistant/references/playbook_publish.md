# Playbook: Publish Readiness

Use this when preparing work for public release or mirror promotion.

## Steps
1. Scan changed files for policy violations against `governance/publication_policy.md`.
2. Confirm `governance/publish_checklist.md` reflects current truth.
3. Verify CI or local evidence for changed behavior.
4. Confirm required sign-off artifact exists in `governance/signoffs/` when policy requires it.
5. Produce release-safe summary language.

## Required Output
1. `Policy findings`: violations or `none`.
2. `Checklist status`: each item pass/fail.
3. `Evidence`: tests/artifacts used for confidence.
4. `Decision`: `READY` or `NOT READY`.
5. `Blockers`: only when decision is `NOT READY`.

## Defaults
- Default to `NOT READY` on missing evidence.
- Prefer exact file paths when citing blockers.
