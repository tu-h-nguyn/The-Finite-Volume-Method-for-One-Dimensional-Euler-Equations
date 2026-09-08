"""Smoke tests for the command-line interface."""

import pytest

from euler1d.cli import main


def test_list_command(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "sod" in out and "hllc_muscl" in out


def test_run_command_reports_errors(capsys):
    assert main(["run", "--problem", "sod", "--scheme", "local_lax_friedrichs", "--nx", "50"]) == 0
    out = capsys.readouterr().out
    assert "L1 error" in out and "Local Lax-Friedrichs" in out


def test_run_command_writes_csv(tmp_path, capsys):
    target = tmp_path / "out" / "sod.csv"
    assert main(["run", "--nx", "40", "--csv", str(target)]) == 0
    lines = target.read_text().splitlines()
    assert lines[0] == "x,rho,u,p,e,mach"
    assert len(lines) == 41


def test_compare_command(capsys):
    assert main(["compare", "--nx", "50", "--schemes", "lax_friedrichs", "hllc_muscl"]) == 0
    out = capsys.readouterr().out
    assert "L1(rho)" in out


def test_converge_command(capsys):
    assert main(["converge", "--schemes", "local_lax_friedrichs", "--resolutions", "20", "40"]) == 0
    out = capsys.readouterr().out
    assert "order" in out


def test_exact_command(capsys):
    assert main(["exact", "--problem", "sod"]) == 0
    out = capsys.readouterr().out
    assert "0.3031301781" in out
    assert "rarefaction | contact | shock" in out


def test_exact_command_rejects_the_smooth_problem(capsys):
    assert main(["exact", "--problem", "smooth"]) == 1
    assert "not a Riemann problem" in capsys.readouterr().err


def test_plot_output(tmp_path):
    target = tmp_path / "figure.png"
    assert main(["run", "--nx", "40", "--plot", str(target)]) == 0
    assert target.exists() and target.stat().st_size > 0


def test_missing_command_exits():
    with pytest.raises(SystemExit):
        main([])
