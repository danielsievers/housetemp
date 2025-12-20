---
trigger: always_on
---

1. When testing main.py consider that you can use --debug-output for easier processing of the returned data
2. Before any commit, you MUST run `tests/check_golden.py` to verify no regressions. If it fails, explain discrepancies to the user before proceeding.
3. You **MUST** run all python scripts, tests, and experiments using the virtual environment (`.venv/bin/python` or similar). **NEVER** use the system python for any task in this repo.
4. Home Assistant specific documention belongs under custom_components/housetemp. The root level docs should not mention HASS specifics.
5. **Release Process**: To bump the version:
   a. Update the `version` key in `custom_components/housetemp/manifest.json`.
   b. Commit the change with a message like "bump version to X.Y.Z".
   c. Create a git tag matching the version number (e.g., `X.Y.Z`).
   d. Push tags and create a GitHub Release: `git push origin main --tags` and `gh release create X.Y.Z --title "vX.Y.Z" --generate-notes`.

6. **Design.md Scope**: This file MUST contain ONLY the underlying physics, math models, and core control logic. It MUST NOT contain Home Assistant specific details (services, config flows, sensors). You MUST update `Design.md` whenever the foundational models or logic change.
