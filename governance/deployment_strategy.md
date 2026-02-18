# Deployment Strategy

## Selected deployment type
- GitHub Pages static documentation deployment.

## Why this strategy fits this repository
- This repository is currently library-first and portfolio-oriented.
- A static docs site provides visible public artifacts without adding app runtime risk.
- Pages deployment keeps release friction low while preserving CI quality gates.

## Required CI and deployment checks
1. Unit/regression tests pass in `CI`.
2. Deployment checks confirm:
   - `docs-site/index.html` exists and is valid HTML entrypoint.
   - `.github/workflows/pages.yml` exists.
3. `PR Signal Gate` passes for PR narrative and commit standards.

## Rollback path
1. Disable Pages deployment by reverting `.github/workflows/pages.yml`.
2. Revert `docs-site/` changes if deployment content is invalid.
3. Keep CI active to preserve core test protections while deployment is paused.
