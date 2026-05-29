# Contributing

Thank you for improving D-ACE. Contributions are welcome for dataset characteristics, data quality checks, packaging, documentation, tests, and examples.

## Before opening an issue

- Search existing issues and pull requests.
- Use the most specific issue template available.
- Do not open public issues for security vulnerabilities; follow [SECURITY.md](SECURITY.md).
- Include dataset sources, licenses, and references when proposing data or metrics.

## Development setup

```bash
python -m venv .venv
```

On Windows, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with:

```bash
source .venv/bin/activate
```

Then install the project and development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Local checks

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
pytest
python -m build
```

Use focused tests for small changes and broader tests when changing shared behavior. If you cannot run a check locally, mention that in the pull request.

## Code guidelines

- Keep public APIs backward compatible where practical.
- Add docstrings for public functions and include expected input types.
- Prefer clear pandas, NumPy, and scikit-learn APIs over custom parsing or implicit conversions.
- Avoid adding heavy runtime dependencies unless the feature clearly needs them.
- Keep notebooks reproducible and move reusable logic into the package when possible.

## Dataset guidelines

- Include the original source, license, and citation for new datasets.
- Do not add personal, sensitive, confidential, or unlawfully obtained data.
- Prefer small example datasets that are useful for tests, demos, or validation.
- Document target columns, known limitations, and preprocessing assumptions.

## Pull requests

- Keep pull requests focused on one topic.
- Link related issues.
- Update documentation and tests with behavior changes.
- Describe validation commands and any known limitations.
