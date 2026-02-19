# Output Contract

Use this file to keep style and structure consistent across responses.

## Response Shape (non-trivial tasks)
1. `Objective`: what is being solved.
2. `Constraints`: policy or technical limits in scope.
3. `Approach`: chosen path and short rationale.
4. `Changes`: concrete files/areas touched.
5. `Validation`: commands/tests and outcomes.
6. `Governance Status`: pass/fail against required gates.
7. `Risks and Rollback`: residual risk and rollback path.
8. `Decision`: `READY` or `NOT READY`.

## Response Shape (simple tasks)
- Return concise direct answer.
- Include gate status only if governance risk exists.

## Formatting Rules
- Prefer short sections and numbered lists.
- Avoid nested bullets.
- Use explicit file paths and concrete actions.
- Keep tone factual and execution-focused.

## Naming and Wording Defaults
- Use Conventional Commit syntax for commit proposals.
- Keep release lines externally readable.
- Avoid internal jargon that does not help reviewers.
