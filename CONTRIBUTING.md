# Contributing to InsightML

Thank you for your interest in contributing to InsightML. This guide covers
everything you need to get started.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Running Tests](#running-tests)
3. [Code Style](#code-style)
4. [Pull Request Process](#pull-request-process)
5. [Commit Message Convention](#commit-message-convention)
6. [Reporting Issues](#reporting-issues)
7. [Code of Conduct](#code-of-conduct)

---

## Development Setup

1. Fork the repository on GitHub and clone your fork:

   ```bash
   git clone https://github.com/rupeshbharambe24/InsightML.git
   cd InsightML
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux / macOS
   .venv\Scripts\activate      # Windows
   ```

3. Install the package in editable mode with dev dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

## Running Tests

Run the full test suite with:

```bash
pytest tests/ -x -q
```

- `-x` stops on the first failure so you can fix issues incrementally.
- `-q` keeps the output concise.

To run a specific test file:

```bash
pytest tests/eda/test_univariate.py -x -q
```

## Code Style

This project uses **ruff** for linting and formatting.

```bash
ruff check src/ tests/
```

Key style rules:

- **Line length**: 100 characters maximum.
- **Type hints**: Encouraged on all public function signatures.
- **Docstrings**: Use Google-style docstrings for public classes and functions.
- **Imports**: Sorted automatically by ruff; one import per line for clarity.

Fix auto-fixable lint issues with:

```bash
ruff check --fix src/ tests/
```

## Pull Request Process

1. **Fork** the repository and create a feature branch from `master`:

   ```bash
   git checkout -b feat/my-feature master
   ```

2. **Write code** and add or update tests as needed.

3. **Ensure all tests pass** and the linter reports no errors:

   ```bash
   pytest tests/ -x -q
   ruff check src/ tests/
   ```

4. **Push** your branch to your fork and open a Pull Request against `master`.

5. Fill out the PR template. A maintainer will review your changes and may
   request modifications before merging.

## Commit Message Convention

Use **imperative mood** and prefix each commit message with one of the
following tags:

| Prefix      | Purpose                          |
|-------------|----------------------------------|
| `feat:`     | New feature                      |
| `fix:`      | Bug fix                          |
| `docs:`     | Documentation only               |
| `test:`     | Adding or updating tests         |
| `refactor:` | Code change that is not a fix or feature |
| `chore:`    | Build scripts, CI, dependencies  |

Examples:

```
feat: add SHAP summary plot to intelligence stage
fix: handle missing values in correlation matrix
docs: update installation instructions in README
```

## Reporting Issues

Before opening an issue, please:

1. Search existing issues to avoid duplicates.
2. Use the appropriate issue template (bug report or feature request).
3. Include a minimal reproducible example when reporting bugs.
4. Specify your environment: OS, Python version, and `dissectml` version.

## Code of Conduct

All contributors are expected to follow our
[Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

---

Thank you for helping make InsightML better.
