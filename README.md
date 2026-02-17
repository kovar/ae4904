# AE4904 Detailed Design Project 2026 Group B

This repository contains the code for the detailed design project of AE4904, a course at Delft University of Technology. The project is focused on ...

## Project Structure

The repository is structured as a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/), with each package living under the `packages/` directory:

- [`edac`](packages/edac/) — Error detection and correction algorithms.
- TBD

## Dependencies

Code is Python and dependencies are managed using `uv`. To install `uv`, see [uv's documentation](https://docs.astral.sh/uv/).

To install all workspace packages and their dependencies, run from the repository root:

```bash
uv sync
```

This creates a single `.venv` at the root with all packages installed in editable mode.

## Adding a New Package

1. Create a new directory under `packages/`:

   ```bash
   uv init packages/my_package --lib
   ```

2. The `packages/*` glob in the root `pyproject.toml` will pick it up automatically.

3. Run `uv sync` to install the new package into the workspace environment.

## Tests

Testing framework is `pytest`. Run all tests from the repository root:

```bash
uv run pytest
```

## Documentation

Documentation for the code is not yet available.

## Linting, Formatting and Type Checking

- Linting and formatting are performed by `ruff`.
- Type checking is done by `ty`.

## Notebooks

Notebooks are used for exploratory data analysis and prototyping. They are built in `marimo`.