# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Plot session logs recorded with [l] (see src/robot_logger.py).

Tick a joint to give it its own row of plots — position (goal vs read), velocity, and, when
they were recorded, that servo's own voltage and current. Tick an IMU trace (roll, pitch,
gyro) or `voltage`/`current` (the pack as a whole) to get a full-width row. Nothing is
plotted until you tick something.

Pass several logs to overlay them and compare the same signal across runs. If every log
carries a `policy_t0` (stamped when a policy went active), time is shifted so t=0 is the
policy start in each run and the traces line up; otherwise raw log time is used.

Comparing exactly two policy-aligned logs also offers an `MAE` trace: the position error
between the runs, averaged over the joints and then averaged over time from t=2.5 s
onwards (before that the robot is still settling into the gait, so the curve is held at 0).

`--odometry` replays the log through the placo estimator in src/odometry.py and adds where
the robot went — trunk position, heading, and body-frame linear velocity — as traces of
their own. `--view` replays it in the MuJoCo viewer instead of plotting, on a loop, with
the floating base placed by that same odometry.

    uv run --group debug src/debug/plot_log.py                    # newest log in logs/
    uv run --group debug src/debug/plot_log.py logs/a.json
    uv run --group debug src/debug/plot_log.py logs/a.json logs/b.json logs/c.json
    uv run --group debug src/debug/plot_log.py logs/a.json --odometry
    uv run --group debug --group sim src/debug/plot_log.py logs/a.json --view
"""

import sys
import json
import math
import time
import bisect
import argparse
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import CheckButtons

# src/odometry.py and the constants it needs live one directory up; this script is run by
# path (`uv run src/debug/plot_log.py`), so nothing else puts src/ on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOG_DIR = "logs"

# Scene rather than the bare robot: --view wants the floor and lights to judge the replay
# against. The odometry itself loads the robot alone (odometry.DEFAULT_MODEL_PATH).
SCENE_PATH = "src/model/mjcf/scene.xml"

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

# Odometry traces, offered only with --odometry since computing them means replaying the
# whole log through placo. They ride on the odometry's own finer time grid rather than the
# log's ticks. Position and heading are dead reckoned and drift; the velocities are
# instantaneous estimates and are as noisy as the joint velocity reads feeding them.
ODOMETRY_UNITS = {
    "odom x": "m",
    "odom y": "m",
    "odom z": "m",
    "odom yaw": "deg",
    "body vx": "m/s",
    "body vy": "m/s",
    # The trunk yaw rate, with the IMU mount rotation taken out. The raw `gyro y` channel
    # above is the *negative* of this one (see odometry.gyro_to_trunk), so this is the axis
    # to read a turn off without having to remember the sign.
    "body vyaw": "rad/s",
}

# Velocity traces are per-tick estimates riding on noisy servo velocity reads, and they
# swing hard within every stride. The dashed overlay is what you read a sustained walking
# speed or a steady turn rate off; the raw trace is what you read the stride structure off.
EMA_TRACES = frozenset({"body vx", "body vy", "body vyaw"})

# The velocity target handed to the policy, logged by the scheduler in physical units
# (src/robot_logger.py "command", scaled by scale_velocity before both the policy and the
# log see it). Overlaid on the odometry trace of the same axis, so what the robot was asked
# for and what it actually did read off one pair of curves. Note `vtheta` pairs with the
# yaw *rate*, which is why `body vyaw` replaced `body vz` here.
COMMAND_AXIS = {"body vx": "vx", "body vy": "vy", "body vyaw": "vtheta"}

# Black in both solo and comparison figures: the command is the target being tracked, not
# one more per-log measurement, so it should not take a log's colour. Solid, to stay
# distinct from the dashed EMA sharing the same axes.
COMMAND_STYLE = {"color": "black", "linewidth": 1.2}

# Time constant for that overlay. At 0.75 s it spans a stride or two — long enough to
# average the within-stride swing away, short enough to still follow a change of command.
VELOCITY_EMA_TAU_S = 0.75

# Everything plotted full width, with no goal/read pair: one label -> unit lookup for all
# of them, so they share a single rendering path.
TRACE_UNITS = {**IMU_UNITS, **ODOMETRY_UNITS}

# Servo telemetry recorded only while [u] was on during the run. Each channel gets a column
# on every ticked joint plus a full-width aggregate row: the motors share one supply, so the
# pack as a whole is worth seeing next to the per-joint traces.
class Telemetry(NamedTuple):
    label: str          # checkbox label
    key: str            # channel name in the log
    unit: str           # y-axis unit
    ema: bool = False   # add a dashed trend line, for channels too noisy to read a level off
    magnitude: bool = False  # plot |value|: the sign is a direction, the magnitude is the load
    total: bool = False  # aggregate by summing the motors rather than averaging them


TELEMETRY = (
    # Every motor sees the same supply, so averaging is the meaningful summary and the
    # min/max band shows the drop along the daisy chain.
    Telemetry("voltage", "voltage", "V"),
    # Current is signed by drive direction, which averages towards zero and hides the load,
    # so every current trace is |I|. The motors draw from one pack, so they are summed:
    # the total is what the battery and the overcurrent safety actually see, and a mean
    # would read 19x lower than OVERCURRENT_CUTOFF_A.
    Telemetry("current", "current", "A", ema=True, magnitude=True, total=True),
)

# Divergence between exactly two policy-aligned runs: the mean |position| error over the
# joints, averaged forward in time. Offered only for a pair of logs, since "the error
# between the runs" has no meaning for one log or for three.
MAE_LABEL = "MAE"

# The first seconds after a policy goes active are the robot settling into the gait from
# whatever pose it started in, and that transient dwarfs the steady-state difference we are
# actually comparing. The integral therefore holds at 0 until the runs have settled.
MAE_START_S = 2.5

# EMA time constant for that trend line. At ~2 s it averages over several walking cycles,
# so the dashed line shows the sustained draw rather than the within-stride peaks.
EMA_TAU_S = 8.0

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


def series(values: list, magnitude: bool = False) -> list[float]:
    """Nulls (dropped reads) become NaN so matplotlib leaves a gap instead of joining across.

    With `magnitude`, values are taken as |v| — see Telemetry.magnitude.
    """
    out = [float("nan") if v is None else v for v in values]
    return [abs(v) for v in out] if magnitude else out


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


def odometry_samples(log: dict) -> list:
    """Replay `log` through the placo odometry (see src/odometry.py).

    The samples come back on the odometry's own finer grid, not on the log's ticks — the
    log is interpolated up to ODOMETRY_DT first — so they carry their own timestamps.

    Imported lazily: placo pulls in pinocchio and takes a moment to load the model, and a
    plain plotting run has no use for either.
    """
    from odometry import replay_log

    return replay_log(log)


def _unwrap_degrees(angles: list[float]) -> list[float]:
    """Remove the +-180 deg wraps, so a turn reads as a continuous curve.

    Heading is dead reckoned and unbounded — a robot that turns twice really is at 720 deg
    — but atan2 folds it back into a half turn, putting a vertical jump in the plot exactly
    where nothing happened.
    """
    out: list[float] = []
    offset = 0.0
    previous: float | None = None
    for angle in angles:
        if previous is not None:
            delta = angle - previous
            if delta > 180.0:
                offset -= 360.0
            elif delta < -180.0:
                offset += 360.0
        previous = angle
        out.append(angle + offset)
    return out


def odometry_channels(samples: list) -> dict[str, list[float]]:
    """Odometry traces, keyed by the label shown on its checkbox."""
    return {
        "odom x": [float(s.position[0]) for s in samples],
        "odom y": [float(s.position[1]) for s in samples],
        "odom z": [float(s.position[2]) for s in samples],
        "odom yaw": _unwrap_degrees([math.degrees(s.yaw) for s in samples]),
        "body vx": [float(s.body_velocity[0]) for s in samples],
        "body vy": [float(s.body_velocity[1]) for s in samples],
        "body vyaw": [float(s.omega_trunk[2]) for s in samples],
    }


def view_replay(samples: list, scene_path: str = SCENE_PATH) -> None:
    """Loop the log in the MuJoCo viewer, with the base placed by the odometry.

    Purely kinematic: the joints are written straight from the recorded readback and the
    free base from the estimated trunk pose, then only forward kinematics is run. Nothing
    is simulated, so this shows what the estimator believes happened — including the feet
    sinking into or floating above the floor wherever it believes wrong.

    Runs on the odometry's own interpolated grid, so the playback is smoother than the
    50 Hz the log was recorded at. Time still comes from the log's clock, and the replay
    restarts from the top until the viewer window is closed.
    """
    import mujoco
    import mujoco.viewer

    from constants import MOTOR_TO_ID
    from odometry import matrix_to_quat

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    qpos_adr = {}
    for name in MOTOR_TO_ID:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            qpos_adr[name] = model.jnt_qposadr[joint_id]

    print(f"Replaying {len(samples)} ticks in the viewer — close the window to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            wall_start = time.perf_counter()
            for i, sample in enumerate(samples):
                if not viewer.is_running():
                    break

                T = sample.T_world_trunk
                data.qpos[0:3] = T[:3, 3]
                data.qpos[3:7] = matrix_to_quat(T[:3, :3])
                for name, adr in qpos_adr.items():
                    data.qpos[adr] = sample.joints.get(name, 0.0)

                mujoco.mj_forward(model, data)
                viewer.sync()

                # Hold this frame until the next one is due. The deadline is absolute —
                # measured from wall_start, not from the previous frame — and `now` is
                # read after the forward kinematics and the sync, so the tick's own cost
                # comes out of the sleep and a slow frame does not push the ones after it
                # late. A frame that overruns simply does not sleep.
                if i + 1 < len(samples):
                    sleep_s = (
                        wall_start + (samples[i + 1].t - samples[0].t) - time.perf_counter()
                    )
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)


def channel_stats(
    log: dict, key: str, magnitude: bool = False
) -> tuple[list[float], list[float], list[float], list[float]] | None:
    """Per-tick (mean, min, max, total) across the motors, or None if not recorded.

    The caller picks the summary that suits the quantity: voltage is shared, so its mean
    is the pack level and the min/max spread is the drop along the daisy chain; current is
    drawn per motor, so its total is what the battery sees. Ticks where nothing was read
    stay NaN, so a run with [u] toggled mid-session leaves a gap rather than a line drawn
    to zero.
    """
    channel = log.get(key)
    if not channel:
        return None

    columns = [series(values, magnitude) for values in channel.values()]
    if not columns or not any(any(v == v for v in col) for col in columns):
        return None  # channel present but empty — [u] was never on

    mean, lo, hi, total = [], [], [], []
    for tick in zip(*columns):
        valid = [v for v in tick if v == v]
        if valid:
            mean.append(sum(valid) / len(valid))
            lo.append(min(valid))
            hi.append(max(valid))
            total.append(sum(valid))
        else:
            mean.append(float("nan"))
            lo.append(float("nan"))
            hi.append(float("nan"))
            total.append(float("nan"))
    return mean, lo, hi, total


def ema(times: list[float], values: list[float], tau_s: float = EMA_TAU_S) -> list[float]:
    """Exponential moving average of `values`, sampled at `times`.

    Uses alpha = 1 - exp(-dt/tau) rather than a fixed weight, so the smoothing is set by
    the time constant and not by the log rate — a log recorded at a different frequency,
    or one with dropped ticks, still gives the same curve.

    Gaps (NaN) pass through as NaN without polluting the running value, and the longer dt
    across the gap is accounted for when the signal resumes.
    """
    out: list[float] = []
    state: float | None = None
    prev_t: float | None = None

    for t, v in zip(times, values):
        if v != v:  # NaN
            out.append(float("nan"))
            continue
        if state is None:
            state = v
        else:
            dt = t - prev_t
            if dt > 0.0:
                state += (1.0 - math.exp(-dt / tau_s)) * (v - state)
        prev_t = t
        out.append(state)
    return out


def policy_t0(log: dict) -> float | None:
    """When a policy went active in this log, in log seconds (None if none did)."""
    return log.get("metadata", {}).get("policy_t0")


def _interpolate(times: list[float], values: list[float], t: float) -> float:
    """`values` sampled at `t`, linearly between the two surrounding ticks.

    NaN outside the recorded span, and NaN across a gap rather than a straight line drawn
    over it: a dropped read is missing data, not a measurement.
    """
    i = bisect.bisect_left(times, t)
    if i == 0:
        return values[0] if times and times[0] == t else float("nan")
    if i >= len(times):
        return float("nan")
    t0, t1 = times[i - 1], times[i]
    v0, v1 = values[i - 1], values[i]
    if v0 != v0 or v1 != v1:  # NaN either side
        return float("nan")
    if t1 == t0:
        return v0
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


def cumulative_mae(
    times: list[list[float]], logs: list[dict], joints: list[str], start_s: float = MAE_START_S
) -> list[float]:
    """Running mean of the position MAE between two runs, on the first run's time grid.

    At each tick of the first log, the second is interpolated to that same policy-relative
    time — the two runs are logged independently, so their ticks do not line up — and the
    error is averaged over the joints. Point `k` is then that per-tick MAE averaged over
    everything from `start_s` up to `k`, so the curve reads in rad and settles towards the
    run's overall divergence rather than growing without bound.

    The average is over time (trapezoid integral / elapsed), not over samples, so it does
    not shift with the log rate. Ticks where either run has no reading contribute to
    neither the integral nor the elapsed time, so a gap leaves the mean untouched instead
    of diluting it.
    """
    (times_a, times_b), (log_a, log_b) = times, logs
    a_pos = {j: series(log_a["position"][j]) for j in joints}
    b_pos = {j: series(log_b["position"][j]) for j in joints}

    out: list[float] = []
    total = elapsed = 0.0
    prev_t = prev_mae = None

    for k, t in enumerate(times_a):
        if t < start_s:
            out.append(0.0)
            continue

        errors = []
        for j in joints:
            a = a_pos[j][k]
            b = _interpolate(times_b, b_pos[j], t)
            if a == a and b == b:
                errors.append(abs(a - b))
        mae = sum(errors) / len(errors) if errors else float("nan")

        if prev_mae is not None and mae == mae and prev_mae == prev_mae:
            dt = t - prev_t
            total += 0.5 * (prev_mae + mae) * dt
            elapsed += dt
        out.append(total / elapsed if elapsed > 0.0 else float("nan"))
        prev_t, prev_mae = t, mae

    return out


def common_joints(logs: list[dict]) -> list[str]:
    """Joints present in every log, in the first log's order."""
    shared = [j for j in logs[0]["position"] if all(j in log["position"] for log in logs)]
    if not shared:
        raise SystemExit("The logs have no joints in common.")
    return shared


def build_figure(entries: list[tuple[Path, dict]], odometry: list[list] | None = None):
    """Build the figure from (path, log) pairs. Returns (fig, check) — the caller must
    hold on to `check`, since a garbage-collected matplotlib widget stops responding.

    `odometry` holds the replayed samples per log (see odometry_samples), or None when
    --odometry was not asked for."""
    logs = [log for _, log in entries]
    joints = common_joints(logs)
    solo = len(entries) == 1

    # Same rule as the IMU traces: only offered when every log has it. Each available
    # channel becomes both a per-joint column and a full-width aggregate row.
    telemetry = [
        (t, [channel_stats(log, t.key, t.magnitude) for log in logs]) for t in TELEMETRY
    ]
    telemetry = [(t, stats) for t, stats in telemetry if all(s is not None for s in stats)]
    telemetry_labels = [t.label for t, _ in telemetry]

    # Align on the policy start only when every log can be — a partial alignment would
    # silently compare runs against different origins.
    aligned = all(policy_t0(log) is not None for log in logs)
    offsets = [(policy_t0(log) if aligned else 0.0) for log in logs]
    times = [[t - offset for t in log["time"]] for log, offset in zip(logs, offsets)]

    # Full-width traces: the IMU channels, plus the odometry ones when they were computed.
    # Only offered when every log has them, so a comparison is always like-for-like.
    #
    # The odometry is replayed on its own finer grid (odometry.ODOMETRY_DT), so it does not
    # share the log's tick times and carries an x axis of its own. `trace_times` holds that
    # override; anything absent from it is sampled at the log's own ticks.
    traces = [imu_channels(log) for log in logs]
    trace_times: list[dict[str, list[float]]] = [{} for _ in logs]
    if odometry is not None:
        for i, samples in enumerate(odometry):
            channels = odometry_channels(samples)
            traces[i] |= channels
            trace_times[i] |= {
                label: [s.t - offsets[i] for s in samples] for label in channels
            }
    trace_labels = [label for label in TRACE_UNITS if all(label in ch for ch in traces)]

    # Comparing the two runs against each other only makes sense once they share an origin,
    # so the MAE row rides on the same policy alignment as the rest of the figure.
    mae = cumulative_mae(times, logs, joints) if len(entries) == 2 and aligned else None

    labels = joints + trace_labels + telemetry_labels + ([MAE_LABEL] if mae else [])

    # A ticked joint gets position, velocity, then one column per recorded channel.
    ncols = 2 + len(telemetry)

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
            grid = GridSpec(len(selected), ncols, figure=fig, hspace=0.3, wspace=0.16, **GRID_BOX)
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

                    for col, (tel, _) in enumerate(telemetry, start=2):
                        ax_extra = fig.add_subplot(grid[row, col], sharex=shared)
                        row_axes.append(ax_extra)
                        for i, (_, log) in enumerate(entries):
                            # A joint missing from the channel (older log) leaves the panel
                            # empty rather than dropping the whole row.
                            values = log[tel.key].get(label)
                            if values:
                                ax_extra.plot(
                                    times[i], series(values, tel.magnitude), **read_style(i)
                                )

                    if not titled:
                        # Column headings live on the first joint row, wherever it lands.
                        suffix = "" if solo else " — dashed: goal, solid: read"
                        ax_pos.set_title(f"Position (rad){suffix}", fontsize=10)
                        ax_vel.set_title("Velocity (rad/s)", fontsize=10)
                        for col, (tel, _) in enumerate(telemetry, start=2):
                            prefix = "|" if tel.magnitude else ""
                            suffix = "|" if tel.magnitude else ""
                            row_axes[col].set_title(
                                f"{prefix}{tel.label.capitalize()}{suffix} ({tel.unit})", fontsize=10
                            )
                        titled = True
                elif label in telemetry_labels:
                    tel, stats = telemetry[telemetry_labels.index(label)]
                    ax = fig.add_subplot(grid[row, :], sharex=shared)
                    shared = shared or ax
                    row_axes = [ax]

                    for i, _ in enumerate(entries):
                        mean, lo, hi, total = stats[i]
                        aggregate = total if tel.total else mean
                        # The band is the spread of individual motors, which only shares an
                        # axis with the mean — against a total it would sit near zero and
                        # read as a second, unrelated signal.
                        if not tel.total:
                            ax.fill_between(
                                times[i], lo, hi, color=colors[i], alpha=0.2, linewidth=0
                            )
                        ax.plot(times[i], aggregate, **read_style(i))
                        if tel.ema:
                            ax.plot(
                                times[i], ema(times[i], aggregate),
                                color=colors[i], linestyle="--", linewidth=1.6,
                            )

                    name = f"|{label}|" if tel.magnitude else label
                    name = f"Σ {name}" if tel.total else name
                    ax.set_ylabel(f"{name} ({tel.unit})", fontsize=9)
                    if tel.ema:
                        summary = "sum over motors" if tel.total else "mean across motors"
                        ax.legend(
                            handles=[
                                plt.Line2D([], [], color=colors[0], linewidth=1.0),
                                plt.Line2D([], [], color=colors[0], linestyle="--", linewidth=1.6),
                            ],
                            labels=[summary, f"EMA ({EMA_TAU_S:.0f} s)"],
                            loc="upper right", fontsize=8,
                        )
                elif label == MAE_LABEL:
                    # One curve for the pair, not one per log, so it takes the full width.
                    ax = fig.add_subplot(grid[row, :], sharex=shared)
                    shared = shared or ax
                    row_axes = [ax]

                    ax.plot(times[0], mae, color=SOLO_GOAL_COLOR, linewidth=1.2)
                    ax.axvline(MAE_START_S, color="0.6", linestyle=":", linewidth=0.8)
                    ax.set_ylabel("mean MAE (rad)", fontsize=9)
                    ax.legend(
                        handles=[plt.Line2D([], [], color=SOLO_GOAL_COLOR, linewidth=1.2)],
                        labels=[
                            f"{entries[0][0].stem} vs {entries[1][0].stem}"
                            f"  (from t={MAE_START_S:g} s)"
                        ],
                        loc="upper left", fontsize=8,
                    )
                else:
                    # IMU and odometry traces have no goal/read pair, so they take the
                    # full width.
                    ax = fig.add_subplot(grid[row, :], sharex=shared)
                    shared = shared or ax
                    row_axes = [ax]

                    smoothed = label in EMA_TRACES
                    command_axis = COMMAND_AXIS.get(label)
                    commanded = False

                    for i, (_, log) in enumerate(entries):
                        x = trace_times[i].get(label, times[i])
                        ax.plot(x, traces[i][label], **read_style(i))
                        if smoothed:
                            ax.plot(
                                x, ema(x, traces[i][label], VELOCITY_EMA_TAU_S),
                                color=colors[i], linestyle="--", linewidth=1.6,
                            )
                        # The command was recorded per control tick, so it rides on the
                        # log's own time axis rather than the odometry's finer grid. Logs
                        # predating the channel simply get no overlay.
                        values = (log.get("command") or {}).get(command_axis) if command_axis else None
                        if values:
                            ax.plot(times[i], series(values), **COMMAND_STYLE)
                            commanded = True

                    ax.set_ylabel(f"{label} ({TRACE_UNITS[label]})", fontsize=9)
                    if smoothed:
                        handles = [
                            plt.Line2D([], [], color=colors[0], linewidth=1.0),
                            plt.Line2D([], [], color=colors[0], linestyle="--", linewidth=1.6),
                        ]
                        legend_names = ["per tick", f"EMA ({VELOCITY_EMA_TAU_S:g} s)"]
                        if commanded:
                            handles.append(plt.Line2D([], [], **COMMAND_STYLE))
                            legend_names.append("command")
                        ax.legend(
                            handles=handles, labels=legend_names, loc="upper right", fontsize=8
                        )

                axes.extend(row_axes)

                for ax in row_axes:
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=8)
                    if aligned and not solo:
                        ax.axvline(0.0, color="0.6", linewidth=0.8, zorder=0)

                # The MAE row and the EMA-overlaid velocity rows carry legends of their own,
                # which a per-log one would silently replace (a matplotlib axis holds one).
                # On those rows the log colours are still readable from the joint rows.
                if row == 0 and label != MAE_LABEL and label not in EMA_TRACES:
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
    parser.add_argument(
        "--odometry",
        action="store_true",
        help="replay the logs through the placo odometry and offer its traces "
        "(trunk position and heading, body linear velocity)",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="replay in the MuJoCo viewer on a loop instead of plotting, with the floating "
        "base placed by the odometry (needs --group sim)",
    )
    args = parser.parse_args()

    paths = args.logs or [latest_log()]
    entries = [(path, load(path)) for path in paths]

    if len(entries) > 1 and not all(policy_t0(log) is not None for _, log in entries):
        missing = [p.name for p, log in entries if policy_t0(log) is None]
        print(f"No policy_t0 in {', '.join(missing)} — comparing on raw log time.")

    if not any("roll" in imu_channels(log) for _, log in entries):
        print("No body_quat in these logs — roll/pitch unavailable (logs predate that channel).")

    for tel in TELEMETRY:
        if not all(channel_stats(log, tel.key) is not None for _, log in entries):
            print(f"No {tel.label} in these logs — press [u] during a run to record it.")

    # The viewer replays one run, so it takes the first log and ignores the rest — there is
    # only one robot in the scene to move.
    if args.view and len(entries) > 1:
        print(f"--view replays one log; showing {entries[0][0].name} and ignoring the rest.")

    odometry = None
    if args.odometry or args.view:
        try:
            print("Computing odometry...")
            odometry = [odometry_samples(log) for _, log in entries]
        except Exception as error:
            # Without the estimate --view has nothing to show, but the plots stand alone,
            # so only the viewer treats this as fatal.
            if args.view:
                raise SystemExit(f"Odometry failed: {error}")
            print(f"Odometry unavailable ({error}) — plotting without it.")

    if args.view:
        view_replay(odometry[0])
        return

    # `check` must stay referenced while the window is open
    fig, check = build_figure(entries, odometry)
    plt.show()


if __name__ == "__main__":
    main()
