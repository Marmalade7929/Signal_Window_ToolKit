# Public Migration Backlog

Track private-to-public migration as small, reviewable pull requests.

## Operating rules
1. One primary outcome per PR.
2. Keep each PR narrowly scoped (target under ~400 net lines when practical).
3. Include tests with each behavior migration.
4. Confirm publication policy before merge.

## Completed
1. `feat/core): add signal toolkit flow and scope modules`
2. `test(core): add synthetic validation suites for core modules`
3. `docs(repo): document public baseline and api usage`
4. `chore(deploy): add pages workflow and deployment checks`
5. `chore(pages): add streamlit launch link and URL gate`
6. `feat(core): migrate app orchestration helper policy`
7. `feat(ui): add minimal Streamlit app shell`

## In progress
1. `feat(core): migrate characterization pipeline`
   - Add `signal_toolkit.device_characterization` in public-safe form.
   - Add focused tests for cycle template, serialization, and trigger handling.

## Next queued PR slices
1. `feat(core): migrate prediction pipeline`
   - Bring prediction/calibration modules in public-safe form.
   - Add deterministic tests for offset calibration and window integration.
2. `docs(repo): add architecture and data flow references`
   - Public-safe diagrams and verification notes.

