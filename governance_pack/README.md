# Governance Pack

Portable governance + skill bundle for reuse across repositories.

## Included
- `governance/publication_policy.md`
- `governance/publish_checklist.md`
- `governance/public_commit_pr_strategy.md`
- `governance/public_migration_backlog.md`
- `governance/deployment_strategy.md`
- `governance/deployment_strategy_template.md`
- `governance/signoffs/.gitkeep`
- `skills/repo-governance-assistant/*`
- `skills/curate-public-repo-output/*`

## Quick Use
1. Copy this `governance_pack/` folder into the target repo.
2. From the target repo root, run:

```bash
bash governance_pack/install_here.sh
```

## Useful Options
```bash
bash governance_pack/install_here.sh --dry-run
bash governance_pack/install_here.sh --force
```

## Archive Strategy
- Keep this folder unchanged in a dedicated archive repo.
- Tag versions when policy/skill rules change.
