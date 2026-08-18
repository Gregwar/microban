# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Record encoders and IMU together, torque OFF, to measure how late the IMU is.

    make imu-delay                       # 30 s recording, legs only, ~200 Hz
    make imu-delay ARGS="--duration 60"

WHAT THIS IS FOR. The trunk's pitch can be known two ways: forward kinematics from the
leg encoders with a foot flat on the floor, and the IMU's own attitude. Forward
kinematics is effectively instantaneous — one bus read, no filter — so the shift between
the two traces is what the IMU costs. Rocking the robot by hand fore/aft drives both
signals at once, which is all the estimate needs.

HOW TO RUN IT. Torque stays off, so the robot is limp and must be held:

    1. stand the robot on a flat, level floor, both feet flat;
    2. start the recording and hold the trunk;
    3. rock it slowly backward and forward — the feet must stay FLAT on the floor,
       the motion is absorbed by the ankles and hips, not by rolling on heel and toe;
    4. give it both slow (~0.5 Hz) and brisk (~2 Hz) rocking: a filter's lag depends on
       frequency and a dead time does not, and the analysis can only tell them apart if
       both are in the recording;
    5. aim for +/- 10 deg or so of trunk pitch, and 20-30 s of it.

Feet rolling on heel or toe is the one thing that breaks the reference — forward
kinematics would then be measuring a lie. The analysis flags it: the FK trace comes out
smaller than the IMU one, and the two feet stop agreeing.

WHAT IT WRITES. One JSON per run in logs/, same shape as a session log plus the IMU
sample timestamps (logs/2026-08-07_15-04-12_imudelay.json). Fetch it with `make
get-logs` and run src/debug/imu_delay_analyze.py on it.

NO TORQUE. The script writes torque_enable=False on every motor before it reads anything
and never writes a goal position, so nothing here can make the robot move on its own.
"""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

from constants import MOTOR_TO_ID
from imu_reader import imu_quat_to_body
from robot_controller import RobotController
from robot_logger import LOG_DIR, sanitize_name

# The chain forward kinematics actually walks: foot -> ... -> trunk. The arms and head hang
# off the trunk and cannot move it, so reading them would only make the sync read longer
# (19 motors instead of 12) for nothing. --all-motors puts them back.
LEG_MOTORS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
]

DEFAULT_HZ = 200.0
DEFAULT_DURATION_S = 30.0

# Let the Madgwick filter settle before recording. It is warm-started from a static
# accelerometer average at startup, but the first seconds still drift toward the true
# attitude, and a transient at the head of the trace is a trend the correlation would fit.
DEFAULT_WARMUP_S = 3.0

STATUS_INTERVAL_S = 0.5
STATUS_WINDOW_S = 1.0


def _pitch_deg(quat) -> float:
    """Pitch of a (w, x, y, z) body quaternion, in degrees.

    ZYX convention, so this is the tilt away from vertical about the trunk's left axis and
    it does not depend on heading — the same quantity the analysis extracts from forward
    kinematics, whose world frame has an arbitrary yaw too.
    """
    w, x, y, z = quat
    return math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))


class Recording:
    """Channels accumulated in memory, written once at the end.

    Nothing touches the disk during the run: this loop is timing a few milliseconds and a
    write stalling it would land straight in the measurement.
    """

    def __init__(self, motor_names: list[str]) -> None:
        self.motor_names = motor_names
        self.time: list[float] = []          # motor read, midpoint of the round trip [s]
        self.read_span_ms: list[float] = []  # width of that round trip [ms]
        self.imu_time: list[float] = []      # when the IMU sample itself was taken [s]
        self.position = {name: [] for name in motor_names}
        self.velocity = {name: [] for name in motor_names}
        self.quat = {axis: [] for axis in "wxyz"}
        self.body_quat = {axis: [] for axis in "wxyz"}
        self.gyro = {axis: [] for axis in "xyz"}
        self.acc = {axis: [] for axis in "xyz"}

    def append(self, t_mid, span_ms, imu_t, positions, velocities, snapshot) -> None:
        self.time.append(t_mid)
        self.read_span_ms.append(span_ms)
        self.imu_time.append(imu_t)
        for name, p, v in zip(self.motor_names, positions, velocities):
            self.position[name].append(p)
            self.velocity[name].append(v)
        body = imu_quat_to_body(snapshot.quat)
        for axis, value in zip("wxyz", snapshot.quat):
            self.quat[axis].append(float(value))
        for axis, value in zip("wxyz", body):
            self.body_quat[axis].append(float(value))
        for axis, value in zip("xyz", snapshot.gyro):
            self.gyro[axis].append(float(value))
        for axis, value in zip("xyz", snapshot.acc):
            self.acc[axis].append(float(value))

    def __len__(self) -> int:
        return len(self.time)


def record(controller: RobotController, ids: list[int], names: list[str],
           duration_s: float, period_s: float) -> tuple[Recording, dict]:
    """Poll motors and IMU until `duration_s` has elapsed (or Ctrl-C).

    Each tick brackets the motor read between two timestamps and keeps the midpoint: the
    encoders were latched somewhere inside that round trip, and at these rates its width
    is a real part of the error budget rather than a rounding detail. The IMU snapshot
    carries its own timestamp from the reader thread, so it is used as-is — reading it
    after the motors costs nothing.
    """
    data = Recording(names)
    t_start = time.perf_counter()
    next_tick = t_start
    errors = 0

    last_status = t_start
    pitch_window: list[tuple[float, float]] = []

    print("Recording — rock the robot backward and forward, feet flat. Ctrl-C to stop early.")
    # Ctrl-C is caught inside the loop rather than around the call: stopping early is a
    # normal way to end a run, and it must return everything captured up to that point
    # instead of throwing the recording away.
    try:
        while True:
            now = time.perf_counter()
            elapsed = now - t_start
            if duration_s > 0.0 and elapsed >= duration_s:
                break

            try:
                t0 = time.perf_counter()
                positions, velocities, _currents, _voltages = controller.sync_read_state(ids)
                t1 = time.perf_counter()
            except Exception:
                errors += 1
                continue

            snapshot = controller.get_imu_sample()
            data.append(
                round(0.5 * (t0 + t1) - t_start, 6),
                round((t1 - t0) * 1e3, 4),
                round(snapshot.timestamp_s - t_start, 6),
                positions,
                velocities,
                snapshot,
            )

            pitch = _pitch_deg([data.body_quat[axis][-1] for axis in "wxyz"])
            pitch_window.append((elapsed, pitch))
            if (now - last_status) >= STATUS_INTERVAL_S:
                pitch_window = [(t, p) for t, p in pitch_window if t >= elapsed - STATUS_WINDOW_S]
                swing = max(p for _, p in pitch_window) - min(p for _, p in pitch_window)
                age_ms = max(0.0, (now - snapshot.timestamp_s) * 1e3)
                remaining = "" if duration_s <= 0.0 else f"{duration_s - elapsed:5.1f} s left  "
                print(
                    f"{remaining}pitch {pitch:+6.1f} deg   swing {swing:5.1f} deg over the last "
                    f"{STATUS_WINDOW_S:.0f} s   rate {len(data) / max(elapsed, 1e-3):5.0f} Hz"
                    f"   imu age {age_ms:4.1f} ms   errors {errors}",
                    flush=True,
                )
                last_status = now

            next_tick += period_s
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()

    except KeyboardInterrupt:
        print("\nStopped early.")

    return data, {"read_errors": errors, "duration_s": time.perf_counter() - t_start}


def summarise(data: Recording, stats: dict, controller: RobotController) -> dict:
    """Whatever tells you, before you leave the robot, whether the run is usable."""
    n = len(data)
    if n < 2:
        return {"samples": n}

    intervals = [b - a for a, b in zip(data.time, data.time[1:])]
    intervals.sort()
    # The IMU thread runs at its own rate; ticks that read it twice between two of its
    # samples are duplicates, and only the distinct ones carry information.
    unique_imu = len(set(data.imu_time))
    ages = [t - i for t, i in zip(data.time, data.imu_time)]
    pitches = [
        _pitch_deg((data.body_quat["w"][k], data.body_quat["x"][k],
                    data.body_quat["y"][k], data.body_quat["z"][k]))
        for k in range(n)
    ]

    imu_times = sorted(set(data.imu_time))
    imu_intervals = sorted(b - a for a, b in zip(imu_times, imu_times[1:])) or [0.0]

    return {
        "samples": n,
        "rate_hz": n / max(stats["duration_s"], 1e-6),
        "interval_median_ms": intervals[len(intervals) // 2] * 1e3,
        "interval_max_ms": intervals[-1] * 1e3,
        "read_span_median_ms": sorted(data.read_span_ms)[n // 2],
        "unique_imu_samples": unique_imu,
        "imu_rate_hz": unique_imu / max(stats["duration_s"], 1e-6),
        "imu_interval_median_ms": imu_intervals[len(imu_intervals) // 2] * 1e3,
        "imu_age_median_ms": sorted(ages)[n // 2] * 1e3,
        "imu_age_max_ms": max(ages) * 1e3,
        "pitch_range_deg": max(pitches) - min(pitches),
        "read_errors": stats["read_errors"],
        "imu_errors": int(controller.get_imu_status()["error_count"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S,
                        help="seconds to record; 0 records until Ctrl-C")
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ,
                        help="polling rate. The default matches the IMU reader's own rate; "
                             "going much faster only starves its thread")
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP_S,
                        help="seconds to let the attitude filter settle before recording")
    parser.add_argument("--all-motors", action="store_true",
                        help="read all 19 motors instead of the 12 leg joints (slower, and "
                             "the arms cannot move the trunk anyway)")
    parser.add_argument("--name", default="imudelay", help="suffix for the log filename")
    args = parser.parse_args()

    if args.hz <= 0:
        raise SystemExit("--hz must be > 0")

    names = list(MOTOR_TO_ID) if args.all_motors else LEG_MOTORS
    ids = [MOTOR_TO_ID[name] for name in names]
    all_ids = list(MOTOR_TO_ID.values())

    controller = RobotController()

    # Torque off first, before anything else touches the bus: this whole experiment is only
    # valid on a limp robot, and it is also the only state in which it is safe to hold.
    controller.sync_write_torque_enable(all_ids, [False] * len(all_ids))
    # Same as main.py: motors answer reads only. Left at the factory's level 2 they would
    # also acknowledge the write above, and those bytes would sit in the buffer waiting to
    # corrupt the first read.
    controller.sync_write_status_return_level(all_ids, [1] * len(all_ids))
    for _ in range(10):  # flush whatever the writes left on the wire
        try:
            controller.sync_read_present_position(all_ids)
        except RuntimeError:
            pass

    print(f"Torque disabled on all {len(all_ids)} motors — the robot is limp, hold it.")
    print(f"Reading {len(ids)} motors at {args.hz:.0f} Hz; IMU thread runs at 200 Hz.")
    print(f"Settling for {args.warmup:.1f} s — keep the robot still, feet flat.")
    time.sleep(max(0.0, args.warmup))

    try:
        data, stats = record(controller, ids, names, args.duration, 1.0 / args.hz)
    finally:
        # Torque was never enabled, but a stray write from anywhere else would be latched
        # on the motors and this is the last chance to clear it before the robot is let go.
        controller.sync_write_torque_enable(all_ids, [False] * len(all_ids))
        controller.shutdown()

    if len(data) < 10:
        print("Nothing usable recorded.")
        return

    report = summarise(data, stats, controller)
    started = datetime.now()
    suffix = sanitize_name(args.name)
    path = Path(LOG_DIR) / (started.strftime("%Y-%m-%d_%H-%M-%S") + (f"_{suffix}" if suffix else "") + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "started_at": started.isoformat(timespec="seconds"),
                    "kind": "imu_delay",
                    "motors": names,
                    "target_hz": args.hz,
                    "baudrate": controller.baudrate,
                    "torque": False,
                    **report,
                },
                "time": data.time,
                "read_span_ms": data.read_span_ms,
                "imu_time": data.imu_time,
                "position": data.position,
                "velocity": data.velocity,
                "quat": data.quat,
                "body_quat": data.body_quat,
                "gyro": data.gyro,
                "acc": data.acc,
            }
        ),
        encoding="utf-8",
    )

    print()
    print(f"  samples          {report['samples']} in {stats['duration_s']:.1f} s "
          f"({report['rate_hz']:.0f} Hz, median interval {report['interval_median_ms']:.2f} ms, "
          f"worst {report['interval_max_ms']:.1f} ms)")
    print(f"  motor read       {report['read_span_median_ms']:.2f} ms per round trip (median)")
    print(f"  IMU samples      {report['unique_imu_samples']} distinct "
          f"({report['imu_rate_hz']:.0f} Hz, median interval {report['imu_interval_median_ms']:.2f} ms)")
    print(f"  IMU age at read  {report['imu_age_median_ms']:.1f} ms median, "
          f"{report['imu_age_max_ms']:.1f} ms worst")
    print(f"  trunk pitch      {report['pitch_range_deg']:.1f} deg peak-to-peak"
          + ("" if report["pitch_range_deg"] >= 8.0
             else "   <-- thin; rock it further next time, the estimate scales with this"))
    print(f"  errors           {report['read_errors']} motor reads, {report['imu_errors']} IMU reads")
    print()
    print(f"Written to {path}")
    print("Next:  make get-logs")
    print(f"       PYTHONPATH=\"src:$PYTHONPATH\" python src/debug/imu_delay_analyze.py {path} --plot")


if __name__ == "__main__":
    main()
