"""Top-level Typer CLI exposed as the ``audio-priors`` console script.

Thin wrapper that delegates to the per-phase scripts. The actual logic
lives in ``scripts/``; this module exists so the entry point declared in
``pyproject.toml`` resolves and ``docker run --rm audio-priors --help``
exits with a useful help string.

Script-path resolution: the scripts directory is found by walking up from
the current working directory looking for a ``scripts/`` folder that
contains ``train.py``. The ``AUDIO_PRIORS_REPO`` env var overrides the
search. In the Docker image, ``WORKDIR /app`` puts the scripts at
``/app/scripts`` so the cwd-based resolution succeeds without extra
configuration.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import typer

app = typer.Typer(
    add_completion=False,
    help="audio-priors: cold-start audio-feature priors for music recommendation.",
)

SCRIPT_NAMES = {
    "download-data": "download_data.py",
    "make-demo-data": "make_demo_data.py",
    "train": "train.py",
    "interpret": "interpret.py",
    "recommend-eval": "recommend_eval.py",
    "prepare-app": "prepare_app.py",
}


def _find_scripts_dir() -> Path:
    """Resolve the scripts directory.

    Order of precedence:
    1. ``AUDIO_PRIORS_REPO`` env var, if set.
    2. The current working directory and its parents, looking for a
       ``scripts/train.py``.
    3. The repo two levels up from this file (works under editable installs).
    """

    override = os.environ.get("AUDIO_PRIORS_REPO")
    if override:
        return Path(override).resolve() / "scripts"

    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "scripts" / "train.py").is_file():
            return candidate / "scripts"

    # Editable-install fallback: src/audio_priors/cli.py -> repo root + /scripts.
    return Path(__file__).resolve().parents[2] / "scripts"


def _run_script(command_name: str, args: list[str]) -> None:
    """Execute a script's ``app`` Typer object in the current process."""

    scripts_dir = _find_scripts_dir()
    path = scripts_dir / SCRIPT_NAMES[command_name]
    if not path.is_file():
        typer.echo(
            f"Could not find {path}. Set AUDIO_PRIORS_REPO to the repo root "
            f"or run from a directory above scripts/.",
            err=True,
        )
        raise typer.Exit(code=1)

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise typer.Exit(code=1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(path), *args]
    module.app()


@app.command("download-data", help="Download Kaggle source datasets.")
def download_data(args: list[str] = typer.Argument(None)) -> None:
    _run_script("download-data", args or [])


@app.command("make-demo-data", help="Generate a 2K-row synthetic dataset.")
def make_demo_data(args: list[str] = typer.Argument(None)) -> None:
    _run_script("make-demo-data", args or [])


@app.command("train", help="Train the modeling panel.")
def train(args: list[str] = typer.Argument(None)) -> None:
    _run_script("train", args or [])


@app.command("interpret", help="Run SHAP, permutation, and calibration.")
def interpret(args: list[str] = typer.Argument(None)) -> None:
    _run_script("interpret", args or [])


@app.command("recommend-eval", help="Evaluate the cold-start recommender.")
def recommend_eval(args: list[str] = typer.Argument(None)) -> None:
    _run_script("recommend-eval", args or [])


@app.command("prepare-app", help="Pre-build artifacts for the Streamlit demo.")
def prepare_app(args: list[str] = typer.Argument(None)) -> None:
    _run_script("prepare-app", args or [])


if __name__ == "__main__":
    app()
