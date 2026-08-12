"""Smoke test: `weight-atlas --help` runs and exits 0."""

from __future__ import annotations

import pytest

from weight_atlas.cli import main


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_scan_subcommand_in_help(capsys):
    with pytest.raises(SystemExit):
        main(["scan", "--help"])
    out = capsys.readouterr().out
    assert "--out" in out


def test_serve_subcommand_in_help(capsys):
    with pytest.raises(SystemExit):
        main(["serve", "--help"])
    out = capsys.readouterr().out
    assert "--host" in out
    assert "--port" in out


def test_no_command_returns_zero(capsys):
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "scan" in out
