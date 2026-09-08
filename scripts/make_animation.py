#!/usr/bin/env python3
"""Render the time evolution of Sod's shock tube as an animated GIF.

The solver is asked for snapshots at evenly spaced times (``output_times``), so
every frame is a state the scheme actually produced — no interpolation between
stored steps. Run with ``python scripts/make_animation.py`` or ``make animation``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from euler1d import get_problem, preset, solve  # noqa: E402
from euler1d.plotting import _STYLE  # noqa: E402
from euler1d.thermo import conservative_to_primitive  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "figures" / "sod_evolution.gif"


def build_animation(nx: int, frames: int, scheme: str, fps: int, path: Path) -> Path:
    problem = get_problem("sod")
    config = preset(scheme, nx=nx, cfl=0.4, t_final=problem.t_final)
    times = np.linspace(0.0, problem.t_final, frames)
    solution = solve(config, problem.initial_condition, output_times=list(times))

    states = [conservative_to_primitive(U, config.gamma) for _, U in solution.snapshots]
    x = solution.x

    style = {**_STYLE, "figure.dpi": 100, "savefig.dpi": 100}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
        panels = [
            (r"density $\rho$", 0, "tab:blue", (0.0, 1.1)),
            ("velocity $u$", 1, "tab:red", (-0.1, 1.05)),
            ("pressure $p$", 2, "tab:green", (0.0, 1.1)),
        ]
        numeric_lines, exact_lines = [], []
        for ax, (title, _, color, ylim) in zip(axes, panels):
            (exact_line,) = ax.plot([], [], "k-", lw=1.3, label="exact")
            (numeric_line,) = ax.plot(
                [], [], color=color, lw=1.6, label=config.label
            )
            exact_lines.append(exact_line)
            numeric_lines.append(numeric_line)
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(*ylim)
            ax.set_xlabel("$x$")
            ax.set_title(title)
        axes[0].legend(loc="lower left", fontsize=7)
        title = fig.suptitle("")
        fig.tight_layout()

        def update(frame: int):
            t = float(solution.snapshots[frame][0])
            values = states[frame]
            exact = problem.exact(x, t)
            for k in range(3):
                numeric_lines[k].set_data(x, values[k])
                exact_lines[k].set_data(x, exact[k])
            title.set_text(f"Sod shock tube,  N = {nx} cells,  t = {t:.3f}")
            return [*numeric_lines, *exact_lines, title]

        animation = FuncAnimation(
            fig, update, frames=len(states), blit=False, interval=1000 // fps
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(path, writer=PillowWriter(fps=fps))
        plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--scheme", default="hllc_muscl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = build_animation(args.nx, args.frames, args.scheme, args.fps, args.output)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
