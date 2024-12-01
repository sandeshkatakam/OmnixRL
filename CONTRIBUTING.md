# Contributing to omnixRL

## Development Setup

1. Fork and clone the repository
2. Install Poetry: `curl -sSL https://install.python-poetry.org | python3 -`
3. Install dependencies: `poetry install`
4. Install pre-commit hooks: `poetry run pre-commit install`

## Development Process

1. Create a new branch: `git checkout -b feature-name`
2. Make your changes
3. Run tests: `poetry run pytest`
4. Run pre-commit hooks: `poetry run pre-commit run --all-files`
5. Submit a pull request

## Code Style

We use:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

## Running Tests

