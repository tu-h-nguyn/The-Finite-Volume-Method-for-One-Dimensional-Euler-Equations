#!/usr/bin/env python3
"""Regenerate every figure and table shipped in ``docs/``.

Run with ``python scripts/generate_results.py`` (or ``make results``).  The
script is deterministic: the same command reproduces byte-comparable numbers,
which is what makes the tables in the README trustworthy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from euler1d import (  # noqa: E402
    PRESETS,
    ConvergenceTable,
    get_problem,
    l1_error,
    preset,
    solve,
)
from euler1d.plotting import (  # noqa: E402
    _STYLE,
    plot_convergence,
    plot_solution_comparison,
    plot_wave_structure,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"
RESULTS = ROOT / "docs" / "RESULTS.md"

REPORT_SCHEMES = [
    "lax_friedrichs",
    "local_lax_friedrichs",
    "second_order_space",
    "second_order_time",
    "second_order_combined",
    "hllc_muscl",
]
RIEMANN_PROBLEMS = ["sod", "lax", "toro2", "toro3", "toro4"]
CONVERGENCE_SCHEMES = [
    "lax_friedrichs",
    "local_lax_friedrichs",
    "second_order_combined",
    "hllc_muscl",
]


def _run(scheme: str, problem, nx: int, cfl: float):
    config = preset(
        scheme,
        nx=nx,
        cfl=cfl,
        x_min=problem.x_min,
        x_max=problem.x_max,
        t_final=problem.t_final,
        gamma=problem.gamma,
        boundary=problem.boundary,
    )
    return solve(config, problem.initial_condition)


def sod_accuracy_table(nx: int, cfl: float) -> tuple[str, dict]:
    problem = get_problem("sod")
    solutions, rows = {}, []
    for scheme in REPORT_SCHEMES:
        solution = _run(scheme, problem, nx, cfl)
        rho_e, u_e, p_e = problem.exact(solution.x, solution.t)
        dx = solution.config.dx
        label = PRESETS[scheme]["label"]
        solutions[label] = solution
        rows.append(
            (
                label,
                l1_error(solution.density, rho_e, dx),
                l1_error(solution.velocity, u_e, dx),
                l1_error(solution.pressure, p_e, dx),
                solution.steps,
            )
        )

    best = min(row[1] for row in rows)
    header = (
        "| scheme | L1 error (rho) | L1 error (u) | L1 error (p) | steps "
        "| error vs. Lax-Friedrichs |"
    )
    lines = [
        f"Sod shock tube, N = {nx}, CFL = {cfl}, t = {problem.t_final}.",
        "",
        header,
        "|---|---:|---:|---:|---:|---:|",
    ]
    reference = rows[0][1]
    for label, e_rho, e_u, e_p, steps in rows:
        marker = " **(best)**" if e_rho == best else ""
        lines.append(
            f"| {label}{marker} | {e_rho:.3e} | {e_u:.3e} | {e_p:.3e} | "
            f"{steps} | {reference / e_rho:.1f}x smaller |"
        )
    return "\n".join(lines), solutions


def convergence_tables(resolutions: list[int], cfl: float) -> tuple[str, list]:
    problem = get_problem("smooth")
    tables, blocks = [], []
    for scheme in CONVERGENCE_SCHEMES:
        errors = []
        for nx in resolutions:
            solution = _run(scheme, problem, nx, cfl)
            exact, _, _ = problem.exact(solution.x, solution.t)
            errors.append(l1_error(solution.density, exact, solution.config.dx))
        table = ConvergenceTable(
            list(resolutions), errors, label=PRESETS[scheme]["label"]
        )
        tables.append(table)
        blocks.append(f"**{table.label}**\n\n{table.to_markdown()}")
    return "\n\n".join(blocks), tables


def plot_problem_suite(nx: int, cfl: float, path: Path) -> Path:
    """Density profiles of the whole Toro suite with a contact-resolving scheme."""
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, len(RIEMANN_PROBLEMS), figsize=(15, 2.9))
        for ax, name in zip(axes, RIEMANN_PROBLEMS):
            problem = get_problem(name)
            solution = _run("hllc_muscl", problem, nx, cfl)
            rho_e, _, _ = problem.exact(solution.x, solution.t)
            ax.plot(solution.x, rho_e, "k-", lw=1.4, label="exact")
            ax.plot(solution.x, solution.density, "o", ms=2.0, color="tab:red", label="HLLC+MUSCL")
            ax.set_title(f"{name}  (t = {problem.t_final:g})", fontsize=9)
            ax.set_xlabel("$x$")
        axes[0].set_ylabel(r"density $\rho$")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Toro's test suite, N = {nx} cells", y=1.03)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return path


def plot_limiter_study(nx: int, cfl: float, path: Path) -> Path:
    """How much resolution the choice of TVD limiter buys on Sod's contact."""
    problem = get_problem("sod")
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        rows = []
        for limiter in ["minmod", "van_leer", "mc", "superbee"]:
            config = preset("second_order_combined", nx=nx, cfl=cfl, limiter=limiter)
            solution = solve(config, problem.initial_condition)
            rho_e, _, _ = problem.exact(solution.x, solution.t)
            error = l1_error(solution.density, rho_e, config.dx)
            rows.append((limiter, error))
            ax.plot(solution.x, solution.density, lw=1.1, label=f"{limiter} ({error:.1e})")
        x_fine = get_problem("sod").config(nx=2000).cell_centers
        rho_fine, _, _ = problem.exact(x_fine, problem.t_final)
        ax.plot(x_fine, rho_fine, "k-", lw=1.2, label="exact", zorder=0)
        ax.set_xlim(0.55, 0.9)
        ax.set_xlabel("$x$")
        ax.set_ylabel(r"density $\rho$")
        ax.set_title(f"Contact and shock, limiter comparison (N = {nx})")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return path, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument(
        "--resolutions", type=int, nargs="+", default=[25, 50, 100, 200, 400, 800]
    )
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    print("• Sod accuracy table")
    accuracy_md, solutions = sod_accuracy_table(args.nx, args.cfl)

    print("• figure: sod_comparison.png")
    plot_solution_comparison(
        get_problem("sod"), solutions, FIGURES / "sod_comparison.png",
        title=f"Sod shock tube at t = 0.2 — N = {args.nx}, CFL = {args.cfl}",
    )

    print("• figure: sod_wave_structure.png")
    plot_wave_structure(get_problem("sod"), FIGURES / "sod_wave_structure.png")

    print("• convergence study")
    convergence_md, tables = convergence_tables(args.resolutions, args.cfl)
    plot_convergence(tables, FIGURES / "convergence.png")

    print("• figure: toro_suite.png")
    plot_problem_suite(args.nx, args.cfl, FIGURES / "toro_suite.png")

    print("• figure: limiters.png")
    _, limiter_rows = plot_limiter_study(args.nx, args.cfl, FIGURES / "limiters.png")
    limiter_md = "\n".join(
        ["| limiter | L1 error (ρ) |", "|---|---:|"]
        + [f"| {name} | {error:.3e} |" for name, error in limiter_rows]
    )

    RESULTS.write_text(
        "\n".join(
            [
                "# Results",
                "",
                "> Generated by `python scripts/generate_results.py`. "
                "Every number below is reproducible from a clean checkout.",
                "",
                "## 1. Accuracy on Sod's shock tube",
                "",
                accuracy_md,
                "",
                "![Sod comparison](figures/sod_comparison.png)",
                "",
                "## 2. Verified order of accuracy",
                "",
                "Discontinuous data cap every scheme at first order in $L^1$, so the design",
                "order is measured on a smooth, exactly-advected density wave instead.",
                "",
                convergence_md,
                "",
                "![Convergence](figures/convergence.png)",
                "",
                "## 3. Robustness across Toro's test suite",
                "",
                "![Toro suite](figures/toro_suite.png)",
                "",
                "## 4. Effect of the slope limiter",
                "",
                limiter_md,
                "",
                "![Limiters](figures/limiters.png)",
                "",
                "## 5. Exact wave structure",
                "",
                "![Wave structure](figures/sod_wave_structure.png)",
                "",
            ]
        )
    )
    print(f"\nwrote {RESULTS.relative_to(ROOT)} and {len(list(FIGURES.glob('*.png')))} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
