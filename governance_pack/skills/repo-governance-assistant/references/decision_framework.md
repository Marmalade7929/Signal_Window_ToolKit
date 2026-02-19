# Decision Framework

Use this file to preserve flexibility while staying consistent.

## Step 1: Classify Request
- `implementation`: code or behavior change
- `curation`: commit/PR/release packaging
- `publish`: private-to-public readiness
- `planning`: roadmap or migration slicing

## Step 2: Assess Risk
- `low`: docs/chore-only, no policy-sensitive surface
- `medium`: behavior change with clear tests and bounded blast radius
- `high`: policy boundary, sensitive data risk, or broad refactor

## Step 3: Select Path
- For `publish`: load `playbook_publish.md`.
- For `curation`: load `playbook_pr_curation.md`.
- For mixed work: combine playbooks, then remove irrelevant steps.
- For low-risk tasks: use minimal process and concise output.

## Step 4: Enforce Rule Weights
- `MUST`: always enforce.
- `SHOULD`: default behavior; allow explicit justified exception.
- `MAY`: optional optimization.

## Step 5: Explain Choice
- State one-line rationale for the chosen path.
- If deviating from a `SHOULD`, state why in one sentence.
