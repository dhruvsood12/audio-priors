"""Top-level Typer CLI exposed as the ``audio-priors`` console script.

Thin wrapper that delegates to the per-phase scripts. The actual logic
lives in ``scripts/``; this module exists so the entry point declared in
``pyproject.toml`` resolves and ``docker run --rm audio-priors --help``
exits with a useful help string.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import typer

app = typer.Typer(
    add_completion=False,
    help="audio-priors: cold-start audio-feature priors for music recommendation.",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {
    "download-data": REPO_ROOT / "scripts" / "download_data.py",
    "make-demo-data": REPO_ROOT / "scripts" / "make_demo_data.py",
    "train": REPO_ROOT / "scripts" / "train.py",
    "interpret": REPO_ROOT / "scripts" / "interpret.py",
    "recommend-eval": REPO_ROOT / "scripts" / "recommend_eval.py",
    "prepare-app": REPO_ROOT / "scripts" / "prepare_app.py",
}


def _run_script(path: Path, args: list[str]) -> None:
    """Execute a script's ``app`` Typer object in the current process."""

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise typer.Exit(code=1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(path), *args]
    module.app()


@app.command("download-data", help="Download Kaggle source datasets.")
def download_data(args: list[str] = typer.Argument(None)) -> None:
    _run_script(SCRIPTS["download-data"], args or [])


@app.command("make-demo-data", help="Generate a 2K-row synthetic dataset.")
def make_demo_data(args: list[str] = typer.Argument(None)) -> None:
    _run_script(SCRIPTS["make-demo-data"], args or [])


@app.command("train", help="Train the modeling panel.")
def train(args: list[str] = typer.Argument(None)) -> None:
    _run_script(SCRIPTS["train"], args or [])


@app.command("interpret", help="Run SHAP, permutation, and calibration.")
def interpret(args: list[str] = typer.Argument(None)) -> None:
    _run_script(SCRIPTS["interpret"], args or [])


@app.command("recommend-eval", help="Evaluate the cold-start recommender.")
def recommend_eval(args: list[str] = typer.Argument(None)) -> None:
    _run_script(SCRIPTS["recommend-eval"], args or [])


@app.command("prepare-app", help="Pre-build artifacts for the Streamlit demo.")
def prepare_app(args: list[str] = typer.Argument(None)) -> None:
    _run_script(SCRIPTS["prepare-app"], args or [])


if __name__ == "__main__":
    app()
