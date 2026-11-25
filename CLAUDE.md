# Notes for Claude

Two Claude sessions work on this repo: a cloud session and a local one on the user's computer. **Read `HANDOFF.md`
first**: it holds the user's standing rules (runtimes, no retry loops, secrets), the current task, and the Log where
both sessions record what they did. Add to that Log when you finish something.

- Tests: `python3 -m unittest discover -s notebook_code/tests -t .` must pass before any commit.
- Notebooks are generated: edit `notebook_code/`, then `python3 -m notebook_code.make_engine_notebooks`.
- Engine versions are pinned in `notebook_code/engine_versions.json` (`python3 tools/engine_updates.py`).
