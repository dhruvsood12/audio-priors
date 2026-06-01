"""Tests for the ``audio-priors`` console entrypoint.

Catches the regression class the panel flagged: cli.py resolving a
bogus REPO_ROOT post-wheel-install would leave every subcommand
crashing on a missing script file. These tests invoke the typer app
in-process via ``CliRunner`` and assert ``--help`` reaches the
underlying script for every registered subcommand.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from audio_priors.cli import SCRIPT_NAMES, app

runner = CliRunner()


def test_top_level_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "audio-priors" in result.stdout


@pytest.mark.parametrize("command", sorted(SCRIPT_NAMES))
def test_subcommand_help_resolves_and_exits_zero(command: str) -> None:
    """Every registered subcommand reaches its underlying script's --help."""
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, (
        f"`audio-priors {command} --help` exited with {result.exit_code}\n"
        f"stdout: {result.stdout}\n"
    )


def test_script_names_cover_every_registered_command() -> None:
    """SCRIPT_NAMES must match the @app.command names; new commands need entries."""
    from audio_priors import cli

    registered = {c.name for c in cli.app.registered_commands}
    assert registered == set(SCRIPT_NAMES.keys())
