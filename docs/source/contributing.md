# Contributing

Contributions are welcome, especially focused fixes, tests, documentation
improvements, and new registry-backed methods.

## Before Opening A Pull Request

1. Start from the current development branch.
2. Keep changes scoped to the phase or feature being changed.
3. Add or update tests for behavior changes.
4. Update docs, configs, or notebooks when parameter names or user workflows change.
5. Run the relevant tests locally.

## Useful Checks

```bash
pytest streamline/tests/test_pipeline_runner.py
pytest streamline/tests/test_uci_demo_datasets.py
sphinx-build -b html docs/source docs/build/html
```

## Documentation Contributions

When updating docs:

* Prefer current P1-P11 phase names.
* Keep config names, CLI names, and notebook names aligned.
* Avoid reintroducing old 9-phase terminology except when explicitly discussing historical docs.
* Update `sample_runcommands.txt` and `run_configs/` if examples change.

## Reporting Bugs

When reporting a bug, include:

* STREAMLINE branch or commit
* operating system and Python version
* command or config used
* traceback or warning output
* a minimal dataset/config example when possible
