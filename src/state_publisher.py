# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Live robot state over ZeroMQ, one JSON message per scheduler tick.

The scheduler owns a `StatePublisher` and calls `publish` every tick, on the robot and in
simulation alike, so anything on the network can watch the robot run without being in the
control loop. The pattern is PUB/SUB: the publisher binds `tcp://*:DEFAULT_PORT` and
sends without waiting, subscribers come and go as they please, and a subscriber that
cannot keep up loses messages rather than slowing the loop down (the send queue is capped
by SNDHWM, and the send never blocks).

The consumer is `src/debug/live_viewer.py` (`make viewer [hostname]`), which replays the
stream through the odometry and shows it in a MuJoCo window.

Each message is a single JSON object. The channels are named like the session log's
(src/robot_logger.py), so the two can be read by the same code::

    {
      "t": 12.34,                          # seconds since the control loop started
      "source": "robot",                   # what produced it: robot / simulation / ...
      "position": {"head": 0.01, ...},     # read joint positions, rad
      "velocity": {"head": 0.0, ...},      # read joint velocities, rad/s
      "target_position": {"head": ...},    # goal positions sent this tick (null = none)
      "gyro": [x, y, z] | null,            # raw IMU frame, rad/s
      "acc": [x, y, z] | null,             # raw IMU frame, g
      "quat": [w, x, y, z] | null,         # IMU sensor frame
      "body_quat": [w, x, y, z] | null,    # trunk frame — this is the robot's attitude
      "projected_gravity": [x, y, z] | null,
      "command": {"vx": 0.0, "vy": 0.0, "vtheta": 0.0},
      "active_moves": ["walk"]
    }

A null IMU field means that tick's IMU read failed (the observer leaves them empty).
"""

import json

import zmq

from constants import MOTOR_TO_ID

DEFAULT_PORT = 5555

# Messages queued for a subscriber before the newest ones are dropped. A viewer that stalls
# should see a gap, not a growing backlog it then has to chew through late.
SEND_HWM = 50


def _as_float(value) -> float | None:
    """Plain float or None — the policies hand out numpy scalars, which json refuses."""
    return None if value is None else float(value)


def _as_floats(values) -> list[float] | None:
    """A sequence as a list of floats, or None when empty (a failed IMU read)."""
    return [float(v) for v in values] if values else None


class StatePublisher:
    """PUB socket bound on `port`, broadcasting the scheduler's state once per tick."""

    def __init__(self, port: int = DEFAULT_PORT) -> None:
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, SEND_HWM)
        # Do not hold the process at exit on a subscriber that never drained its queue.
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(f"tcp://*:{port}")
        self.port = port

    def publish(
        self,
        robot_state,
        target_angles: dict[str, float],
        user_input,
        source: dict | None = None,
    ) -> None:
        """Send this tick. Never blocks; a full queue simply drops the message."""
        message = {
            "t": float(robot_state.time_s),
            "source": (source or {}).get("source", "unknown"),
            "position": {name: _as_float(robot_state.motor_positions.get(name)) for name in MOTOR_TO_ID},
            "velocity": {name: _as_float(robot_state.motor_velocities.get(name)) for name in MOTOR_TO_ID},
            "target_position": {name: _as_float(target_angles.get(name)) for name in MOTOR_TO_ID},
            "gyro": _as_floats(robot_state.gyro),
            "acc": _as_floats(robot_state.acc),
            "quat": _as_floats(robot_state.quat),
            "body_quat": _as_floats(robot_state.body_quat),
            "projected_gravity": _as_floats(robot_state.projected_gravity),
            "command": {axis: _as_float(user_input.velocity.get(axis, 0.0)) for axis in ("vx", "vy", "vtheta")},
            "active_moves": sorted(user_input.active_moves),
        }
        try:
            self._socket.send_string(json.dumps(message), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self) -> None:
        self._socket.close()
