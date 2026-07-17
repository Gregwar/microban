# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Plot a session log recorded with [l] (see src/robot_logger.py).

Tick a joint to give it its own pair of plots — position (goal vs read) and velocity —
and untick to take it away. Nothing is plotted until you tick something. Run it on logs
pulled over with `make get-logs`:

    uv run --group debug src/debug/plot_log.py         # most recent log in logs/
    uv run --group debug src/debug/plot_log.py logs/2026-07-17_14-32-05_walk-test.json
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import CheckButtons

LOG_DIR = "logs"

GOAL_STYLE = {"color": "black", "linestyle": "--", "linewidth": 1.0}
READ_STYLE = {"color": "tab:blue", "linewidth": 1.0}

# Plot area, to the right of the checkbox panel.
GRID_BOX = {"left": 0.27, "right": 0.97, "top": 0.9, "bottom": 0.08}


def latest_log(log_dir: str = LOG_DIR) -> Path:
    """Most recently started log — file names sort chronologically by construction."""
    logs = sorted(Path(log_dir).glob("*.json"))
    if not logs:
        raise SystemExit(f"No logs found in {log_dir}/. Record one with [l], then `make get-logs`.")
    return logs[-1]


def load(path: Path) -> dict:
    log = json.loads(path.read_text(encoding="utf-8"))
    missing = {"time", "target_position", "position", "velocity"} - set(log)
    if missing:
        raise SystemExit(f"{path} is missing {sorted(missing)} — not a session log?")
    return log


def series(values: list) -> list[float]:
    """Nulls (dropped reads) become NaN so matplotlib leaves a gap instead of joining across."""
    return [float("nan") if v is None else v for v in values]


def build_figure(path: Path, log: dict):
    """Build the figure. Returns (fig, check) — the caller must hold on to `check`,
    since a garbage-collected matplotlib widget silently stops responding."""
    time = log["time"]
    joints = list(log["position"])

    fig = plt.figure(figsize=(13, 8))
    meta = log.get("metadata", {})
    fig.suptitle(f"{path.name}  —  {meta.get('ticks', len(time))} ticks, {meta.get('duration_s', '?')} s")

    check_ax = fig.add_axes([0.01, 0.08, 0.18, 0.84])
    check_ax.set_title("Joints", fontsize=10)
    check = CheckButtons(check_ax, joints, [False] * len(joints))

    hint = fig.text(0.6, 0.5, "Tick a joint to plot it", ha="center", va="center", color="0.5")
    axes: list = []

    def render() -> None:
        """Rebuild the plot area for the ticked joints — one row each.

        The axes are recreated rather than hidden: matplotlib does not reflow a layout
        around invisible axes, so hiding would leave the gaps behind.
        """
        for ax in axes:
            fig.delaxes(ax)
        axes.clear()

        selected = [j for j, on in zip(joints, check.get_status()) if on]
        hint.set_visible(not selected)

        if selected:
            grid = GridSpec(len(selected), 2, figure=fig, hspace=0.3, wspace=0.16, **GRID_BOX)
            shared = None
            for row, joint in enumerate(selected):
                ax_pos = fig.add_subplot(grid[row, 0], sharex=shared)
                shared = shared or ax_pos
                ax_vel = fig.add_subplot(grid[row, 1], sharex=shared)
                axes.extend((ax_pos, ax_vel))

                ax_pos.plot(time, series(log["target_position"][joint]), **GOAL_STYLE)
                ax_pos.plot(time, series(log["position"][joint]), **READ_STYLE)
                ax_vel.plot(time, series(log["velocity"][joint]), **READ_STYLE)

                ax_pos.set_ylabel(joint, fontsize=9)
                for ax in (ax_pos, ax_vel):
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=8)

                if row == 0:
                    ax_pos.set_title("Position (rad)", fontsize=10)
                    ax_vel.set_title("Velocity (rad/s)", fontsize=10)
                    ax_pos.legend(
                        handles=[plt.Line2D([], [], **GOAL_STYLE), plt.Line2D([], [], **READ_STYLE)],
                        labels=["goal", "read"],
                        loc="upper right",
                        fontsize=8,
                    )
                # Only the bottom row carries the shared x axis labels.
                last = row == len(selected) - 1
                for ax in (ax_pos, ax_vel):
                    ax.tick_params(labelbottom=last)
                    if last:
                        ax.set_xlabel("Time (s)", fontsize=9)

        fig.canvas.draw_idle()

    check.on_clicked(lambda _label: render())
    render()
    return fig, check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", type=Path, help=f"log file (default: newest in {LOG_DIR}/)")
    args = parser.parse_args()

    path = args.log or latest_log()
    log = load(path)

    fig, check = build_figure(path, log)  # `check` must stay referenced while the window is open
    plt.show()


if __name__ == "__main__":
    main()
