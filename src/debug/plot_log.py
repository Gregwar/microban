# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Plot session logs recorded with [l] (see src/robot_logger.py).

Tick a joint to give it its own pair of plots — position (goal vs read) and velocity —
or tick an IMU trace (roll, pitch, gyro) to get a full-width row. Nothing is plotted
until you tick something.

Pass several logs to overlay them and compare the same signal across runs. If every log
carries a `policy_t0` (stamped when a policy went active), time is shifted so t=0 is the
policy start in each run and the traces line up; otherwise raw log time is used.

    uv run --group debug src/debug/plot_log.py                    # newest log in logs/
    uv run --group debug src/debug/plot_log.py logs/a.json
    uv run --group debug src/debug/plot_log.py logs/a.json logs/b.json logs/c.json
"""

import json
import math
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import CheckButtons

LOG_DIR = "logs"

# Single log: goal is dashed black over a blue read. Comparing several, each log takes a
# colour of its own instead, keeping dashed=goal / solid=read.
SOLO_GOAL_COLOR = "black"
SOLO_READ_COLOR = "tab:blue"

# IMU traces, plotted full width since they have no goal/read pair. Roll and pitch come
# from body_quat (the trunk frame); the gyro is logged raw, in the IMU sensor frame.
IMU_UNITS = {
    "roll": "deg",
    "pitch": "deg",
    "gyro x": "rad/s",
    "gyro y": "rad/s",
    "gyro z": "rad/s",
}

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


def _has_data(channel: dict | None, axis: str) -> bool:
    return bool(channel) and any(v is not None for v in channel.get(axis, []))


def _roll_pitch(w: float, x: float, y: float, z: float) -> tuple[float, float]:
    """Roll and pitch in degrees from a (w, x, y, z) quaternion.

    NaN in, NaN out: clamping the asin argument would otherwise turn a dropped IMU read
    into a confident +90 deg, because min()/max() silently pass NaN through.
    """
    if any(v != v for v in (w, x, y, z)):
        return float("nan"), float("nan")
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return math.degrees(roll), math.degrees(pitch)


def imu_channels(log: dict) -> dict[str, list[float]]:
    """IMU traces available in this log, keyed by the label shown on its checkbox."""
    out: dict[str, list[float]] = {}

    body_quat = log.get("body_quat")
    if all(_has_data(body_quat, axis) for axis in ("w", "x", "y", "z")):
        quats = zip(*(series(body_quat[axis]) for axis in ("w", "x", "y", "z")))
        angles = [_roll_pitch(*q) for q in quats]
        out["roll"] = [a[0] for a in angles]
        out["pitch"] = [a[1] for a in angles]

    gyro = log.get("gyro")
    for axis in ("x", "y", "z"):
        if _has_data(gyro, axis):
            out[f"gyro {axis}"] = series(gyro[axis])

    return out


def policy_t0(log: dict) -> float | None:
    """When a policy went active in this log, in log seconds (None if none did)."""
    return log.get("metadata", {}).get("policy_t0")


def common_joints(logs: list[dict]) -> list[str]:
    """Joints present in every log, in the first log's order."""
    shared = [j for j in logs[0]["position"] if all(j in log["position"] for log in logs)]
    if not shared:
        raise SystemExit("The logs have no joints in common.")
    return shared


def build_figure(entries: list[tuple[Path, dict]]):
    """Build the figure from (path, log) pairs. Returns (fig, check) — the caller must
    hold on to `check`, since a garbage-collected matplotlib widget stops responding."""
    logs = [log for _, log in entries]
    joints = common_joints(logs)
    solo = len(entries) == 1

    # Only offer an IMU trace when every log has it, so a comparison is always like-for-like.
    imu = [imu_channels(log) for log in logs]
    imu_labels = [label for label in IMU_UNITS if all(label in channels for channels in imu)]
    labels = joints + imu_labels

    # Align on the policy start only when every log can be — a partial alignment would
    # silently compare runs against different origins.
    aligned = all(policy_t0(log) is not None for log in logs)
    times = [
        [t - (policy_t0(log) if aligned else 0.0) for t in log["time"]]
        for log in logs
    ]

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [SOLO_READ_COLOR] if solo else [palette[i % len(palette)] for i in range(len(entries))]

    def goal_style(i: int) -> dict:
        return {"color": SOLO_GOAL_COLOR if solo else colors[i], "linestyle": "--", "linewidth": 1.0}

    def read_style(i: int) -> dict:
        return {"color": colors[i], "linewidth": 1.0}

    fig = plt.figure(figsize=(13, 8))
    if solo:
        path, log = entries[0]
        meta = log.get("metadata", {})
        title = f"{path.name}  —  {meta.get('ticks', len(log['time']))} ticks, {meta.get('duration_s', '?')} s"
    else:
        origin = "aligned on policy start" if aligned else "raw log time — policy_t0 missing from some logs"
        title = f"Comparing {len(entries)} logs  —  {origin}"
    fig.suptitle(title)

    check_ax = fig.add_axes([0.01, 0.05, 0.18, 0.88])
    check_ax.set_title("Signals", fontsize=10)
    check = CheckButtons(check_ax, labels, [False] * len(labels))
    for text in check.labels:
        text.set_fontsize(8)

    hint = fig.text(0.6, 0.5, "Tick a signal to plot it", ha="center", va="center", color="0.5")
    axes: list = []

    def render() -> None:
        """Rebuild the plot area for the ticked signals — one row each.

        The axes are recreated rather than hidden: matplotlib does not reflow a layout
        around invisible axes, so hiding would leave the gaps behind.
        """
        for ax in axes:
            fig.delaxes(ax)
        axes.clear()

        selected = [label for label, on in zip(labels, check.get_status()) if on]
        hint.set_visible(not selected)

        if selected:
            grid = GridSpec(len(selected), 2, figure=fig, hspace=0.3, wspace=0.16, **GRID_BOX)
            shared = None
            titled = False

            for row, label in enumerate(selected):
                is_joint = label in joints
                row_axes = []

                if is_joint:
                    ax_pos = fig.add_subplot(grid[row, 0], sharex=shared)
                    shared = shared or ax_pos
                    ax_vel = fig.add_subplot(grid[row, 1], sharex=shared)
                    row_axes = [ax_pos, ax_vel]

                    for i, (_, log) in enumerate(entries):
                        ax_pos.plot(times[i], series(log["target_position"][label]), **goal_style(i))
                        ax_pos.plot(times[i], series(log["position"][label]), **read_style(i))
                        ax_vel.plot(times[i], series(log["velocity"][label]), **read_style(i))

                    ax_pos.set_ylabel(label, fontsize=9)
                    if not titled:
                        # Column headings live on the first joint row, wherever it lands.
                        suffix = "" if solo else " — dashed: goal, solid: read"
                        ax_pos.set_title(f"Position (rad){suffix}", fontsize=10)
                        ax_vel.set_title("Velocity (rad/s)", fontsize=10)
                        titled = True
                else:
                    # IMU traces have no goal/read pair, so they take the full width.
                    ax = fig.add_subplot(grid[row, :], sharex=shared)
                    shared = shared or ax
                    row_axes = [ax]

                    for i, _ in enumerate(entries):
                        ax.plot(times[i], imu[i][label], **read_style(i))
                    ax.set_ylabel(f"{label} ({IMU_UNITS[label]})", fontsize=9)

                axes.extend(row_axes)

                for ax in row_axes:
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=8)
                    if aligned and not solo:
                        ax.axvline(0.0, color="0.6", linewidth=0.8, zorder=0)

                if row == 0:
                    first = row_axes[0]
                    if solo and is_joint:
                        handles = [plt.Line2D([], [], **goal_style(0)), plt.Line2D([], [], **read_style(0))]
                        legend_labels = ["goal", "read"]
                    else:
                        handles = [plt.Line2D([], [], **read_style(i)) for i in range(len(entries))]
                        legend_labels = [p.stem for p, _ in entries]
                    if not solo or is_joint:
                        first.legend(handles=handles, labels=legend_labels, loc="upper right", fontsize=8)

                # Only the bottom row carries the shared x axis labels.
                last = row == len(selected) - 1
                for ax in row_axes:
                    ax.tick_params(labelbottom=last)
                    if last:
                        ax.set_xlabel(
                            "Time since policy start (s)" if aligned else "Time (s)", fontsize=9
                        )

        fig.canvas.draw_idle()

    check.on_clicked(lambda _label: render())
    render()
    return fig, check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "logs", nargs="*", type=Path, help=f"log files to overlay (default: newest in {LOG_DIR}/)"
    )
    args = parser.parse_args()

    paths = args.logs or [latest_log()]
    entries = [(path, load(path)) for path in paths]

    if len(entries) > 1 and not all(policy_t0(log) is not None for _, log in entries):
        missing = [p.name for p, log in entries if policy_t0(log) is None]
        print(f"No policy_t0 in {', '.join(missing)} — comparing on raw log time.")

    if not any("roll" in imu_channels(log) for _, log in entries):
        print("No body_quat in these logs — roll/pitch unavailable (logs predate that channel).")

    fig, check = build_figure(entries)  # `check` must stay referenced while the window is open
    plt.show()


if __name__ == "__main__":
    main()
