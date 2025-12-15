---
trigger: always_on
---

1. When testing main.py consider that you can use --debug-output for easier processing of the returned data
2. Before any commit, you MUST run `tests/check_golden.py` to verify no regressions. If it fails, explain discrepancies to the user before proceeding.
3. You **MUST** run all python scripts, tests, and experiments using the virtual environment (`.venv/bin/python` or similar). **NEVER** use the system python for any task in this repo.
