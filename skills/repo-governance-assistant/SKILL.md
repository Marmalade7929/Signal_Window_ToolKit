---
name: repo-governance-assistant
description: Apply lightweight repository governance for policy-safe, public-facing engineering work with consistent output structure. Use when planning or executing changes that must satisfy publication policy, commit/PR standards, deployment checks, migration slicing, or release-note-ready communication without forcing a single implementation path.
---

# Repo Governance Assistant

Use this skill as a router: enforce hard governance constraints and a stable response format, but keep implementation choice flexible.

## Workflow
1. Load repository policy context:
- `governance/publication_policy.md`
- `governance/publish_checklist.md`
- `governance/public_commit_pr_strategy.md`
- `governance/deployment_strategy.md` when release/deployment is in scope
- `governance/public_migration_backlog.md` when planning migration slices
2. Load always-on references:
- `references/guardrails.md`
- `references/output_contract.md`
- `references/decision_framework.md`
3. Load task playbooks only when relevant:
- `references/playbook_publish.md` for publish readiness and evidence checks
- `references/playbook_pr_curation.md` for commit and PR shaping
4. Execute using rule weights:
- Enforce all `MUST` items.
- Follow `SHOULD` items by default; permit deviation with one-line rationale.
- Use `MAY` items as optional accelerators.
5. Report status explicitly:
- Return `READY` only when all `MUST` items pass.
- Return `NOT READY` when any `MUST` item fails and list blockers.

## Operating Rules
- Optimize for small, reviewable, reversible changes.
- Keep policy controls strict and implementation path flexible.
- Prefer repository evidence over assumptions.
- Keep recommendations directly actionable.
