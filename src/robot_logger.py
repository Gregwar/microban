# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Session logger, toggled with [l] from the keyboard and driven by the scheduler.

One JSON file per logging session, named after the date it started, with an optional
user-supplied suffix: ``logs/2026-07-17_14-32-05_walk-test.json``.

Samples are accumulated in memory and written once on stop, so the 50 Hz control loop
never blocks on disk. Channels are keyed by motor name, matching the format the
``src/debug/plot_*_log.py`` scripts already read.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

from constants import MOTOR_TO_ID

LOG_DIR = "logs"

# Everything outside this set is folded into "-" so a typed name is always a safe filename.
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def sanitize_name(name: str) -> str:
    """Fold a user-typed log name into a filesystem-safe suffix ("" if nothing is left)."""
    return _UNSAFE_NAME_CHARS.sub("-", name.strip()).strip("-")


def _as_float(value) -> float | None:
    """Coerce a sample to a plain float, keeping None for missing readings.

    The walk policy commands come out of ONNX as numpy scalars, which json cannot encode.
    Converting per sample rather than at write time means a bad value is caught on the tick
    that produced it, instead of destroying a whole session at stop time.
    """
    return None if value is None else float(value)


class RobotLogger:
    """Records one sample per scheduler tick while active.

    Not thread-safe: start/record/stop are all called from the scheduler loop.
    """

    def __init__(self, log_dir: str = LOG_DIR) -> None:
        self._log_dir = Path(log_dir)
        self._path: Path | None = None
        self._t0: float = 0.0
        self._started_at: str = ""
        self._samples: dict = {}
        self._policy_t0: float | None = None
        self._policy_name: str | None = None

    @property
    def active(self) -> bool:
        return self._path is not None

    def start(self, name: str = "") -> Path:
        """Open a session. `name` is an optional suffix appended to the date stamp."""
        now = datetime.now()
        suffix = sanitize_name(name)
        stem = now.strftime("%Y-%m-%d_%H-%M-%S") + (f"_{suffix}" if suffix else "")

        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._log_dir / f"{stem}.json"
        self._started_at = now.isoformat(timespec="seconds")
        self._t0 = time.perf_counter()
        self._policy_t0 = None
        self._policy_name = None
        self._samples = {
            "time": [],
            "target_position": {name: [] for name in MOTOR_TO_ID},
            "position": {name: [] for name in MOTOR_TO_ID},
            "velocity": {name: [] for name in MOTOR_TO_ID},
            # Servo input voltage and present current, only read while [u] is on. Null on
            # ticks where they were not read, so the channels stay aligned with "time".
            "voltage": {name: [] for name in MOTOR_TO_ID},
            "current": {name: [] for name in MOTOR_TO_ID},
            "gyro": {axis: [] for axis in ("x", "y", "z")},
            "quat": {axis: [] for axis in ("w", "x", "y", "z")},
            # Orientation in the trunk frame. The raw quat above is in the IMU sensor
            # frame (IMU_MOUNT_QUAT is not identity), so it is this one — not `quat` —
            # that yields the robot's roll/pitch.
            "body_quat": {axis: [] for axis in ("w", "x", "y", "z")},
            "command": {axis: [] for axis in ("vx", "vy", "vtheta")},
        }
        return self._path

    def mark_policy_start(self, move_name: str) -> None:
        """Stamp when a policy went active, as seconds into this session.

        Recorded once per session (the first policy to start wins), so that logs of the
        same manoeuvre can be lined up on it even when the operator hit [l] at a
        different moment in each run. A policy already running when logging starts is
        not stamped — its real start time is outside the log.
        """
        if not self.active or self._policy_t0 is not None:
            return
        self._policy_t0 = round(time.perf_counter() - self._t0, 4)
        self._policy_name = move_name

    def record(
        self,
        robot_state,
        target_angles: dict[str, float],
        command_velocity: dict[str, float],
    ) -> None:
        """Append one tick. `command_velocity` is the scaled (vx, vy, vtheta) fed to the policy."""
        if not self.active:
            return

        s = self._samples
        s["time"].append(round(time.perf_counter() - self._t0, 4))

        for name in MOTOR_TO_ID:
            s["target_position"][name].append(_as_float(target_angles.get(name)))
            s["position"][name].append(_as_float(robot_state.motor_positions.get(name)))
            s["velocity"][name].append(_as_float(robot_state.motor_velocities.get(name)))
            s["voltage"][name].append(_as_float(robot_state.motor_voltages.get(name)))
            s["current"][name].append(_as_float(robot_state.motor_currents.get(name)))

        # The IMU read can fail — the observer leaves the field empty rather than raising,
        # so pad with null to keep every channel the same length as "time".
        gyro = robot_state.gyro or [None] * 3
        for axis, value in zip(("x", "y", "z"), gyro):
            s["gyro"][axis].append(_as_float(value))

        quat = robot_state.quat or [None] * 4
        for axis, value in zip(("w", "x", "y", "z"), quat):
            s["quat"][axis].append(_as_float(value))

        body_quat = robot_state.body_quat or [None] * 4
        for axis, value in zip(("w", "x", "y", "z"), body_quat):
            s["body_quat"][axis].append(_as_float(value))

        for axis in ("vx", "vy", "vtheta"):
            s["command"][axis].append(_as_float(command_velocity.get(axis, 0.0)))

    def stop(self) -> Path | None:
        """Write the session to disk and return its path (None if nothing was recorded).

        The session is closed before the write, so a failure to serialize leaves the logger
        inactive rather than re-raising on every following tick.
        """
        if not self.active:
            return None

        path = self._path
        payload = {
            "metadata": {
                "started_at": self._started_at,
                "duration_s": round(time.perf_counter() - self._t0, 3),
                "ticks": len(self._samples["time"]),
                # Seconds into the log at which a policy went active, or null if none did.
                "policy_t0": self._policy_t0,
                "policy": self._policy_name,
            },
            **self._samples,
        }
        self._path = None
        self._samples = {}

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
