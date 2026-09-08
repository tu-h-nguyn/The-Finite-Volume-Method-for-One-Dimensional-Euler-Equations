"""Command-line interface: ``python -m euler1d`` or the ``euler1d`` script."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .errors import ConvergenceTable, l1_error
from .problems import PROBLEMS, get_problem
from .riemann import solve_star_state
from .solver import PRESETS, preset, solve

__all__ = ["main", "build_parser"]

DEFAULT_RESOLUTIONS = (25, 50, 100, 200, 400)


def _write_csv(path: Path, solution) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rho, u, p = solution.primitives
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "rho", "u", "p", "e", "mach"])
        writer.writerows(
            zip(
                solution.x, rho, u, p,
                solution.specific_internal_energy, solution.mach_number,
            )
        )


def _run(args: argparse.Namespace) -> int:
    problem = get_problem(args.problem)
    config = preset(
        args.scheme,
        nx=args.nx,
        cfl=args.cfl,
        x_min=problem.x_min,
        x_max=problem.x_max,
        t_final=args.t_final if args.t_final is not None else problem.t_final,
        gamma=problem.gamma,
        boundary=problem.boundary,
    )
    solution = solve(config, problem.initial_condition)
    rho_exact, u_exact, p_exact = problem.exact(solution.x, solution.t)

    print(f"problem : {problem.name} — {problem.description}")
    print(f"scheme  : {config.label or args.scheme} [{config.describe()}]")
    print(f"grid    : N = {config.nx}, dx = {config.dx:.5g}, CFL = {config.cfl}")
    print(f"time    : t = {solution.t:g} reached in {solution.steps} steps")
    print(f"L1 error: rho {l1_error(solution.density, rho_exact, config.dx):.4e}  "
          f"u {l1_error(solution.velocity, u_exact, config.dx):.4e}  "
          f"p {l1_error(solution.pressure, p_exact, config.dx):.4e}")

    if args.csv:
        _write_csv(Path(args.csv), solution)
        print(f"csv     : {args.csv}")
    if args.plot:
        from .plotting import plot_solution_comparison

        path = plot_solution_comparison(
            problem, {config.label or args.scheme: solution}, args.plot
        )
        print(f"figure  : {path}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    problem = get_problem(args.problem)
    schemes = args.schemes or list(PRESETS)
    solutions = {}
    rows = []
    for name in schemes:
        config = preset(
            name,
            nx=args.nx,
            cfl=args.cfl,
            x_min=problem.x_min,
            x_max=problem.x_max,
            t_final=problem.t_final,
            gamma=problem.gamma,
            boundary=problem.boundary,
        )
        solution = solve(config, problem.initial_condition)
        rho_e, u_e, p_e = problem.exact(solution.x, solution.t)
        solutions[config.label or name] = solution
        rows.append(
            (
                config.label or name,
                solution.steps,
                l1_error(solution.density, rho_e, config.dx),
                l1_error(solution.velocity, u_e, config.dx),
                l1_error(solution.pressure, p_e, config.dx),
            )
        )

    width = max(len(r[0]) for r in rows)
    print(f"{'scheme':<{width}} {'steps':>6} {'L1(rho)':>12} {'L1(u)':>12} {'L1(p)':>12}")
    print("-" * (width + 45))
    for name, steps, e_rho, e_u, e_p in rows:
        print(f"{name:<{width}} {steps:6d} {e_rho:12.4e} {e_u:12.4e} {e_p:12.4e}")

    if args.plot:
        from .plotting import plot_solution_comparison

        path = plot_solution_comparison(problem, solutions, args.plot)
        print(f"\nfigure: {path}")
    return 0


def _converge(args: argparse.Namespace) -> int:
    problem = get_problem(args.problem)
    tables = []
    for name in args.schemes or ["local_lax_friedrichs", "second_order_combined"]:
        errors = []
        for nx in args.resolutions:
            config = preset(
                name,
                nx=nx,
                cfl=args.cfl,
                x_min=problem.x_min,
                x_max=problem.x_max,
                t_final=problem.t_final,
                gamma=problem.gamma,
                boundary=problem.boundary,
            )
            solution = solve(config, problem.initial_condition)
            rho_e, _, _ = problem.exact(solution.x, solution.t)
            errors.append(l1_error(solution.density, rho_e, config.dx))
        label = PRESETS[name].get("label", name)
        table = ConvergenceTable(list(args.resolutions), errors, label=label)
        tables.append(table)
        print(f"\n{label}\n{table}")

    if args.plot:
        from .plotting import plot_convergence

        print(f"\nfigure: {plot_convergence(tables, args.plot)}")
    return 0


def _exact(args: argparse.Namespace) -> int:
    problem = get_problem(args.problem)
    if not hasattr(problem, "left"):
        print(f"{problem.name} is not a Riemann problem", file=sys.stderr)
        return 1
    star = solve_star_state(problem.left, problem.right, problem.gamma)
    print(f"problem      : {problem.name} — {problem.description}")
    print(f"left  state  : rho={problem.left.rho:g} u={problem.left.u:g} p={problem.left.p:g}")
    print(f"right state  : rho={problem.right.rho:g} u={problem.right.u:g} p={problem.right.p:g}")
    print(f"wave pattern : {star.left_wave} | contact | {star.right_wave}")
    print(f"p*           : {star.p:.10f}   (Newton converged in {star.iterations} steps)")
    print(f"u*           : {star.u:.10f}")
    print(f"rho*_L/rho*_R: {star.rho_left:.10f} / {star.rho_right:.10f}")
    print("wave speeds  :")
    for key, value in star.speeds.items():
        position = problem.x0 + value * problem.t_final
        print(f"  {key:<12} S = {value:+.6f}   x(t={problem.t_final:g}) = {position:.6f}")
    if args.plot:
        from .plotting import plot_wave_structure

        print(f"figure: {plot_wave_structure(problem, args.plot)}")
    return 0


def _list(_: argparse.Namespace) -> int:
    print("problems:")
    for name, problem in PROBLEMS.items():
        print(f"  {name:<10} {problem.description}")
    print("\nschemes:")
    for name, spec in PRESETS.items():
        print(f"  {name:<22} {spec.get('label', '')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="euler1d",
        description="Finite-volume solvers for the 1-D Euler equations.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--problem", default="sod", choices=sorted(PROBLEMS))
    common.add_argument("--nx", type=int, default=200, help="number of cells")
    common.add_argument("--cfl", type=float, default=0.5)
    common.add_argument("--plot", help="path of the figure to write")

    run = sub.add_parser("run", parents=[common], help="run one scheme")
    run.add_argument("--scheme", default="second_order_combined", choices=sorted(PRESETS))
    run.add_argument("--t-final", type=float, default=None)
    run.add_argument("--csv", help="write the final state to a CSV file")
    run.set_defaults(func=_run)

    compare = sub.add_parser("compare", parents=[common], help="compare several schemes")
    compare.add_argument("--schemes", nargs="+", choices=sorted(PRESETS))
    compare.set_defaults(func=_compare)

    converge = sub.add_parser(
        "converge", parents=[common], help="grid-convergence study"
    )
    converge.add_argument("--schemes", nargs="+", choices=sorted(PRESETS))
    converge.add_argument(
        "--resolutions", type=int, nargs="+", default=list(DEFAULT_RESOLUTIONS)
    )
    converge.set_defaults(func=_converge, problem="smooth")

    exact = sub.add_parser("exact", parents=[common], help="exact Riemann solution")
    exact.set_defaults(func=_exact)

    listing = sub.add_parser("list", help="list problems and schemes")
    listing.set_defaults(func=_list)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
