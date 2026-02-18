---
name: curate-public-repo-output
description: Curate and standardize agent-generated commits, pull requests, and release-facing summaries for public repositories. Use when preparing a branch for public visibility, reviewing AI-authored diffs, enforcing commit/PR standards, or packaging evidence for high-signal portfolio output.
---

# Curate Public Repo Output

Apply this workflow to convert raw agent output into high-signal, portfolio-ready public repo output.

## Workflow
1. Read policy and standards:
- `governance/publication_policy.md`
- `governance/publish_checklist.md`
- `governance/public_commit_pr_strategy.md`
- `references/curation-standard.md`
2. Audit the proposed diff and classify:
- Primary outcome (feature/fix/docs/chore)
- User-visible impact
- Evidence available (tests, screenshots, metrics)
- Policy risks (public-safe vs blocked content)
3. Curate commit plan:
- Reorder/split into 1 to 4 coherent commits
- Normalize commit titles to conventional format
- Ensure each commit has one purpose and review value
4. Curate PR package:
- Fill required PR sections using `.github/pull_request_template.md`
- Add a concise signal narrative and release note line
- Add risk and rollback notes
5. Enforce gates:
- Confirm tests and docs coverage are adequate
- Confirm policy checklist items can be marked complete
- Mark output `READY` or `NOT READY` with blocking findings

## Commit Standard
- Use `<type>(<scope>): <summary>`
- Use types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`
- Keep summary <= 72 characters
- Keep each commit scoped to one intent

## PR Standard
- State problem, value, and change summary
- Provide concrete validation evidence
- Include explicit risk and rollback
- Include one release-note-ready line

## Output Format
Return:
1. `Verdict`: `READY` or `NOT READY`
2. `Blocking findings`: concise numbered list
3. `Curated commit stack`: ordered commit titles
4. `PR body draft`: fully populated markdown
5. `Release note line`: one sentence

## References
- Use `references/curation-standard.md` for scoring and gate thresholds.
- Use `references/pr-narrative-template.md` for response templates.
