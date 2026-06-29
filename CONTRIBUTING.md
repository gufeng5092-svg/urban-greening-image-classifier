# Contributing

Thank you for improving this project.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Quality Checks

Run these before opening a pull request:

```bash
black .
isort .
flake8 .
pytest
```

## Contribution Rules

- Do not commit raw datasets, generated training outputs, or private checkpoints.
- Keep model label order stable unless checkpoints are retrained and documented.
- Put new runtime parameters in YAML config files under `configs/`.
- Prefer small, focused changes with tests for configuration, data loading, inference, or utilities.
