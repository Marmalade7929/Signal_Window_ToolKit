# Signal Window Toolkit

[![CI](https://github.com/Marmalade7929/Signal_Window_ToolKit/actions/workflows/ci.yml/badge.svg)](https://github.com/Marmalade7929/Signal_Window_ToolKit/actions/workflows/ci.yml)

Signal Window Toolkit is a Python library for two core tasks:
- numerically integrating flow-rate signals into volume estimates
- detecting timing windows and cycle metrics from scope-like traces

## Why this is useful
- Gives a reproducible, test-backed baseline for cycle-aligned signal analysis.


## Quick start
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests -q
streamlit run app.py
```

## Documentation site
- GitHub Pages: `https://marmalade7929.github.io/Signal_Window_ToolKit/`
- Live Streamlit app: `https://signal-window-toolkit.streamlit.app/`
- Source content: `docs-site/index.html`

## Public migration roadmap
- PR-by-PR migration plan: `governance/public_migration_backlog.md`

## Public API
- `signal_toolkit.integrate_flow(...)`
- `signal_toolkit.integrate_flow_to_volume(...)`
- `signal_toolkit.parse_scope_csv(...)`
- `signal_toolkit.extract_fault_cycles(...)`
- `signal_toolkit.detect_ail_windows(...)`

## Safety and publication stance
This repo intentionally excludes private datasets, client/vendor identifiers, and internal-only artifacts. Tests use synthetic-only data.

## Maintainer
- `@Marmalade7929`
