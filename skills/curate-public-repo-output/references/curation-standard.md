# Curation Standard

Use this standard to evaluate if agent-produced output is public-ready.

## Hard Gates (Must Pass)
- Policy-safe content only (no blocked/proprietary artifacts).
- CI/tests relevant to changed behavior must pass.
- PR contains evidence and rollback note.
- Commit stack follows conventional format.

If any hard gate fails, verdict is `NOT READY`.

## Signal Scorecard (0-2 each, target >= 8/10)
1. Problem clarity
- 0: unclear
- 1: somewhat clear
- 2: clear and specific
2. Outcome visibility
- 0: no visible impact
- 1: implied impact
- 2: explicit visible/measurable impact
3. Validation evidence
- 0: none
- 1: partial
- 2: complete and relevant
4. Change focus
- 0: mixed concerns
- 1: mostly focused
- 2: tightly scoped
5. Release readiness
- 0: no release line
- 1: weak release line
- 2: crisp release-ready summary

## Commit Review Rules
- One intent per commit.
- Keep migration or risk-heavy changes isolated.
- Keep docs/tests paired with behavior changes where possible.

## PR Review Rules
- Include user/developer value statement.
- Include tests and artifacts.
- Include known risks and rollback.
- Include one sentence suitable for changelog/release notes.

