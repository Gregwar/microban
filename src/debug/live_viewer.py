# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Live MuJoCo view of a running robot or simulation, fed over ZeroMQ.

Subscribes to the state the scheduler publishes every tick (src/state_publisher.py), runs
it through the odometry as it arrives (odometry.OdometryTracker) and shows the result in a
MuJoCo window (src/debug/odometry_view.py): the joints from the readback, the floating base
from the odometry, and the anchor — the sole corner the estimator holds still — as a dot on
the floor, blue for the left foot and red for the right. Works the same against `make run`
and `make sim`; nothing here is deployed to the robot.

    make viewer                  # a simulation running on this machine (make sim)
    make viewer microban         # the robot
    uv run --group sim src/debug/live_viewer.py microban --port 5555

Keys, in the viewer window: R re-plants the robot at the world origin.

The odometry is dead reckoning: the position drifts and nothing corrects it, and heading
is the IMU's gyro-integrated yaw. See the Odometry section of docs/usage.md.
"""

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import zmq

# src/odometry.py, src/state_publisher.py and the constants they need live one directory
# up; this script is run by path, so nothing else puts src/ on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odometry import OdometryTracker  # noqa: E402
from odometry_view import SCENE_PATH, OdometryView  # noqa: E402
from state_publisher import DEFAULT_PORT  # noqa: E402

# How often the terminal readout (pose, anchor, active moves) is refreshed.
STATUS_INTERVAL_S = 1.0
# Without a message for this long the readout says so: the loop stopped, or the link did.
STALE_AFTER_S = 1.0
# Pause between polls when nothing arrived, so an idle viewer does not spin a core.
IDLE_SLEEP_S = 0.002


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a running robot (or simulation) in the MuJoCo viewer.")
    parser.add_argument("host", nargs="?", default="localhost", help="where the control loop runs (default: localhost)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"state publisher port (default: {DEFAULT_PORT})")
    parser.add_argument("--scene", default=SCENE_PATH, help=f"MJCF scene to display (default: {SCENE_PATH})")
    args = parser.parse_args()

    endpoint = f"tcp://{args.host}:{args.port}"
    socket = zmq.Context.instance().socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.connect(endpoint)

    tracker = OdometryTracker()

    reset_requested = threading.Event()

    def key_callback(keycode: int) -> None:
        # GLFW reports letters as uppercase ASCII.
        if 32 <= keycode < 127 and chr(keycode).lower() == "r":
            reset_requested.set()

    print(f"Waiting for the robot state on {endpoint} (start `make run` or `make sim`)...")
    print("Press R in the viewer to re-plant the robot at the origin.")

    received = 0
    last_message_s: float | None = None
    last_status_s = 0.0
    last_tick: dict | None = None
    last_sample = None
    # Ticks seen since the last readout, and whether that readout already said the stream
    # had gone stale: the stale line is printed once, not every second until data resumes.
    unreported = 0
    stale_reported = False

    with OdometryView(args.scene, key_callback=key_callback) as view:
        while view.is_running():
            if reset_requested.is_set():
                reset_requested.clear()
                tracker.reset()
                print("Odometry reset: the robot is re-planted at the origin.")

            # Drain everything that arrived since the last frame, stepping the odometry on
            # each tick (the anchor transfers must not be skipped), and draw only the last.
            new_sample = None
            while True:
                try:
                    raw = socket.recv_string(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                tick = json.loads(raw)
                if received == 0:
                    print(f"Receiving from {endpoint} (source: {tick.get('source', 'unknown')}).")
                elif last_tick is not None and tick["t"] < last_tick["t"]:
                    print("Source restarted: odometry reset, robot re-planted at the origin.")
                received += 1
                unreported += 1
                last_message_s = time.perf_counter()
                last_tick = tick

                samples = tracker.push(
                    tick["t"], tick["position"], tick["velocity"], tick["body_quat"], tick["gyro"]
                )
                if samples:
                    new_sample = samples[-1]

            if new_sample is not None:
                last_sample = new_sample
                view.show(new_sample)
            else:
                time.sleep(IDLE_SLEEP_S)

            now = time.perf_counter()
            if last_tick is not None and (now - last_status_s) >= STATUS_INTERVAL_S:
                last_status_s = now
                stale = last_message_s is not None and (now - last_message_s) > STALE_AFTER_S
                if unreported == 0 and stale_reported:
                    continue
                stale_reported = stale
                unreported = 0
                moves = ",".join(last_tick.get("active_moves") or []) or "-"
                if last_sample is None:
                    pose = "no attitude yet (IMU read failing?)"
                else:
                    x, y, _ = last_sample.position
                    pose = (
                        f"x={x:+.3f} y={y:+.3f} m  yaw={math.degrees(last_sample.yaw):+.1f} deg"
                        f"  anchor={last_sample.support_frame}"
                    )
                print(
                    f"t={last_tick['t']:8.2f}s  {pose}  moves={moves}  ticks={received}"
                    f"{'  [no data for >1 s]' if stale else ''}"
                )

    print(f"Viewer closed after {received} ticks.")


if __name__ == "__main__":
    main()
