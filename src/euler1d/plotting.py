"""Matplotlib helpers used by the scripts to produce the figures in ``docs/``."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from .errors import ConvergenceTable  # noqa: E402
from .solver import Solution  # noqa: E402

__all__ = ["plot_solution_comparison", "plot_convergence", "plot_wave_structure"]

_STYLE = {
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_solution_comparison(
    problem,
    solutions: dict[str, Solution],
    path: str | Path,
    title: str | None = None,
) -> Path:
    """Density, velocity and pressure of several schemes against the exact solution."""
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        reference = next(iter(solutions.values()))
        x_fine = reference.x
        rho_e, u_e, p_e = problem.exact(x_fine, reference.t)

        panels = [
            ("Density $\\rho$", rho_e, lambda s: s.density),
            ("Velocity $u$", u_e, lambda s: s.velocity),
            ("Pressure $p$", p_e, lambda s: s.pressure),
        ]
        for ax, (name, exact_values, getter) in zip(axes, panels):
            ax.plot(x_fine, exact_values, "k-", lw=1.6, label="exact", zorder=1)
            for label, solution in solutions.items():
                ax.plot(
                    solution.x, getter(solution), lw=1.0, alpha=0.9,
                    marker="", label=label, zorder=2,
                )
            ax.set_xlabel("$x$")
            ax.set_title(name)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncol=min(len(labels), 4),
            fontsize=7, bbox_to_anchor=(0.5, -0.16),
        )
        fig.suptitle(
            title or f"{problem.name}: t = {reference.t:g}, N = {reference.config.nx}"
        )
        fig.tight_layout()
        return _save(fig, path)


def plot_convergence(tables: list[ConvergenceTable], path: str | Path) -> Path:
    """Log-log grid-convergence plot with first- and second-order guide lines."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        for table in tables:
            ax.loglog(
                table.resolutions, table.errors, "o-", lw=1.2, ms=4,
                label=f"{table.label} (q ≈ {table.asymptotic_order:.2f})",
            )
        n0 = tables[0].resolutions[0]
        e0 = tables[0].errors[0]
        grid = [tables[0].resolutions[0], tables[0].resolutions[-1]]
        ax.loglog(grid, [e0 * (n0 / n) for n in grid], "k--", lw=0.8, label="$O(N^{-1})$")
        ax.loglog(
            grid, [e0 * (n0 / n) ** 2 for n in grid], "k:", lw=0.8, label="$O(N^{-2})$"
        )
        resolutions = tables[0].resolutions
        ax.set_xticks(resolutions)
        ax.set_xticks([], minor=True)
        ax.set_xticklabels([str(n) for n in resolutions])
        ax.set_xlabel("number of cells $N$")
        ax.set_ylabel("$L^1$ error in density")
        ax.set_title("Grid convergence, smooth density wave")
        ax.legend(fontsize=7)
        fig.tight_layout()
        return _save(fig, path)


def plot_wave_structure(problem, path: str | Path, t_max: float | None = None) -> Path:
    """The exact wave fan of a Riemann problem drawn in the $x$-$t$ plane."""
    from .riemann import solve_star_state

    star = solve_star_state(problem.left, problem.right, problem.gamma)
    t_max = t_max or problem.t_final
    styles = {
        "left_head": ("tab:blue", "--", "rarefaction head"),
        "left_tail": ("tab:blue", "-.", "rarefaction tail"),
        "left_shock": ("tab:red", "-", "left shock"),
        "contact": ("tab:green", "-", "contact"),
        "right_shock": ("tab:red", "-", "right shock"),
        "right_head": ("tab:blue", "--", "rarefaction head"),
        "right_tail": ("tab:blue", "-.", "rarefaction tail"),
    }
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        for key, speed in star.speeds.items():
            color, dash, label = styles.get(key, ("gray", "-", key))
            ax.plot(
                [problem.x0, problem.x0 + speed * t_max], [0, t_max],
                color=color, ls=dash, lw=1.4, label=label,
            )
        # Shade the rarefaction fans and label the constant regions.
        if star.left_wave == "rarefaction":
            ax.fill_betweenx(
                [0, t_max],
                [problem.x0, problem.x0 + star.speeds["left_head"] * t_max],
                [problem.x0, problem.x0 + star.speeds["left_tail"] * t_max],
                color="tab:blue", alpha=0.12, lw=0,
            )
        if star.right_wave == "rarefaction":
            ax.fill_betweenx(
                [0, t_max],
                [problem.x0, problem.x0 + star.speeds["right_tail"] * t_max],
                [problem.x0, problem.x0 + star.speeds["right_head"] * t_max],
                color="tab:blue", alpha=0.12, lw=0,
            )

        left_edge = star.speeds.get("left_shock", star.speeds.get("left_head"))
        left_inner = star.speeds.get("left_shock", star.speeds.get("left_tail"))
        right_inner = star.speeds.get("right_shock", star.speeds.get("right_tail"))
        right_edge = star.speeds.get("right_shock", star.speeds.get("right_head"))
        t_label = 0.72 * t_max

        def position(speed: float) -> float:
            return problem.x0 + speed * t_label

        contact = position(star.speeds["contact"])
        regions = [
            ("L", 0.5 * (problem.x_min + position(left_edge))),
            ("L*", 0.5 * (position(left_inner) + contact)),
            ("R*", 0.5 * (contact + position(right_inner))),
            ("R", 0.5 * (position(right_edge) + problem.x_max)),
        ]
        for name, x_label in regions:
            ax.text(
                x_label, t_label, name,
                ha="center", va="center", fontsize=9, color="0.35",
            )

        ax.set_xlim(problem.x_min, problem.x_max)
        ax.set_ylim(0, t_max)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$t$")
        ax.set_title(
            f"{problem.name}: {star.left_wave} | contact | {star.right_wave}\n"
            f"$p_* = {star.p:.5f}$, $u_* = {star.u:.5f}$"
        )
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=7, loc="upper left")
        fig.tight_layout()
        return _save(fig, path)
