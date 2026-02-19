# Playbook: PR and Commit Curation

Use this when shaping AI-generated work into a reviewable, high-signal PR.

## Steps
1. Define one primary outcome for the PR.
2. Split or reorder commits into a coherent narrative:
- foundation (if needed)
- behavior change
- tests/docs
- cleanup
3. Normalize commit titles using Conventional Commit style.
4. Build PR narrative with:
- problem/value
- change summary
- validation evidence
- risks and rollback
- one release-note line
5. Check policy alignment before final recommendation.

## Commit Rules
- One intent per commit.
- Keep summaries concise and imperative.
- Prefer 1 to 4 meaningful commits per PR.

## PR Rules
- Show reviewer-facing evidence, not claims.
- Keep scope tight; propose follow-up slices if oversized.
- Make rollback operational, not abstract.

## Required Output
1. `Curated commit stack`
2. `PR body draft`
3. `Release note line`
4. `Decision`: `READY` or `NOT READY`
