# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Replay a recorded session log through the MuJoCo simulation.

Feeds the ``target_position`` channel of a log (recorded with [l], see src/robot_logger.py)
straight into the simulated motors, tick by tick. The robot starts standing in the neutral
pose on the floor, exactly where a fresh sim run or a [r] reset lands it, and then tracks
whatever goal positions the log commanded. Useful for seeing what a real-robot run would
have looked like in simulation, or for producing a matching sim log to overlay on the real
one with src/debug/plot_log.py.

Position gain follows the log: the motors run at KP_DEFAULT until the log's ``policy_t0``
(the moment a policy went active), then switch to KP_RL — every policy in this codebase
(squat_rl, walk, getup) commands that same gain in its ``on_start``. A log with no
``policy_t0`` stays at KP_DEFAULT throughout.

    PYTHONPATH=src uv run --group sim src/debug/sim_log.py logs/run.json
    PYTHONPATH=src uv run --group sim src/debug/sim_log.py logs/run.json --view
    PYTHONPATH=src uv run --group sim src/debug/sim_log.py logs/run.json --log run_sim
    PYTHONPATH=src uv run --group sim src/debug/sim_log.py logs/run.json --settle-time 0.5

With --view the MuJoCo viewer opens and the replay runs in real time (paced by the log's
own timestamps). Without it the replay runs headless, as fast as the physics allows. With
--log a fresh log is written to logs/ using the very same RobotLogger the live control loop
uses, so the two logs are directly comparable.

With --settle-time N, for the first N seconds after policy_t0 the simulated joints are
reset every tick to the log's *read* positions (velocity zeroed), leaving the free base to
settle onto the floor. This lets the robot drop into the recorded pose the policy actually
started from — e.g. flat on its back for getup — before the physics runs the joints freely.
"""

import time
import json
import argparse
from pathlib import Path

from constants import MOTOR_TO_ID, NEUTRAL_POSE, KP_DEFAULT, KP_RL
from observer import Observer
from robot_logger import RobotLogger
from sim.mujoco_controller import MuJoCoController

MJCF_PATH = "src/model/mjcf/scene.xml"


def _tick_dt(times: list[float]) -> float:
    """Average tick spacing of the log, falling back to the 50 Hz control period."""
    if len(times) < 2:
        return 0.02
    return (times[-1] - times[0]) / (len(times) - 1)


def _align_time_axis(path: Path, times: list[float], policy_t0: float | None) -> None:
    """Rewrite a produced log's time axis onto the source log's clock.

    RobotLogger timestamps with wall time, which is fine live but compresses a headless
    replay into whatever the physics took to run. The replay is tick-for-tick against the
    source, so the honest timestamps are the source's own — with them the two logs line up
    directly in plot_log.py. Everything else RobotLogger wrote is left untouched.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    aligned = times[: len(data["time"])]
    data["time"] = aligned
    if aligned:
        data["metadata"]["duration_s"] = round(aligned[-1] - aligned[0], 3)
    data["metadata"]["policy_t0"] = policy_t0
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def replay(log: dict, view: bool, log_name: str | None, settle_time: float) -> None:
    times = log["time"]
    if not times:
        print("Log has no samples, nothing to replay.")
        return

    target_position = log["target_position"]
    position = log["position"]
    command = log.get("command", {})
    metadata = log.get("metadata", {})
    policy_t0 = metadata.get("policy_t0")
    policy_name = metadata.get("policy") or "policy"

    names = list(MOTOR_TO_ID.keys())
    ids = list(MOTOR_TO_ID.values())
    dt = _tick_dt(times)

    controller = MuJoCoController(mjcf_path=MJCF_PATH, enable_viewer=view)
    controller.sync_write_kp(ids, [KP_DEFAULT] * len(ids))

    observer = Observer(controller)
    logger = RobotLogger()
    if log_name is not None:
        # Record voltage and current too, so the produced log carries the same channels a
        # live run with [u] on would — otherwise they would be null throughout.
        observer.observe_voltage = True
        observer.observe_current = True
        path = logger.start(log_name)
        print(f"Logging replay to {path}")

    print(
        f"Replaying {len(times)} ticks ({times[-1] - times[0]:.1f} s) "
        f"{'in the viewer (real time)' if view else 'headless'}"
    )
    if policy_t0 is not None:
        print(f"Policy '{policy_name}' starts at t={policy_t0:.3f}s -> switching to KP_RL")
        if settle_time > 0:
            print(
                f"Settling: joints held at read positions for t in "
                f"[{policy_t0:.3f}, {policy_t0 + settle_time:.3f}]s"
            )
    elif settle_time > 0:
        print("--settle-time ignored: log has no policy_t0 to anchor the settle window to.")

    # Carry-forward for any missing goal: the log leaves a channel null on ticks where a
    # reading was absent, and the sim always needs a concrete target for every motor.
    last_targets = dict(NEUTRAL_POSE)
    switched = False

    wall_start = time.perf_counter()
    for i, t in enumerate(times):
        if not controller.is_running():
            print("Viewer closed, stopping replay.")
            break

        # Follow the log's gain schedule: default gain until the policy went active.
        if policy_t0 is not None and not switched and t >= policy_t0:
            controller.sync_write_kp(ids, [KP_RL] * len(ids))
            switched = True

        # Read the sim state first (mirrors the scheduler: observe, then command), so the
        # produced log pairs this tick's feedback with this tick's goal.
        robot_state = observer.read_state(dt)
        robot_state.time_s = t

        in_settle = (
            policy_t0 is not None
            and settle_time > 0
            and policy_t0 <= t < policy_t0 + settle_time
        )

        if in_settle:
            # Settle window: pin the joints onto the recorded readback and command that same
            # pose, so they hold steady through the step while the free base drops into the
            # configuration the policy actually started from (e.g. flat on the floor for
            # getup). The reset lands before the step so the base settles against the right
            # pose rather than one drifting toward a far policy goal.
            targets = {
                name: (last_targets[name] if position[name][i] is None else float(position[name][i]))
                for name in names
            }
            controller.reset_joints_to(targets)
        else:
            targets = {
                name: (last_targets[name] if target_position[name][i] is None else float(target_position[name][i]))
                for name in names
            }
        last_targets = targets

        controller.sync_write_goal_position(ids, [targets[name] for name in names])

        if logger.active:
            command_velocity = {
                axis: (command.get(axis, [])[i] if i < len(command.get(axis, [])) else 0.0)
                for axis in ("vx", "vy", "vtheta")
            }
            # Carried over from the source log rather than recomputed: the replay commands
            # joint targets, not a height, so the only honest height command is the one the
            # original run recorded.
            heights = command.get("height", [])
            height_target = heights[i] if i < len(heights) else None
            logger.record(robot_state, targets, command_velocity, height_target)
            if switched:
                # First call wins, so this stamps the produced log's policy_t0 once.
                logger.mark_policy_start(policy_name)

        # In viewer mode, pace the replay to the log's own timestamps so it plays back in
        # real time. Headless just runs flat out.
        if view and i + 1 < len(times):
            target_wall = wall_start + (times[i + 1] - times[0])
            sleep_s = target_wall - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    if logger.active:
        path = logger.stop()
        if path is not None:
            _align_time_axis(path, times, policy_t0)
            print(f"Replay log written to {path}")

    controller.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a recorded log through MuJoCo.")
    parser.add_argument("logfile", type=Path, help="Path to the session log JSON to replay")
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open the MuJoCo viewer and replay in real time (headless otherwise)",
    )
    parser.add_argument(
        "--log",
        nargs="?",
        const="",
        default=None,
        metavar="NAME",
        help="Write a fresh log of the replay to logs/, with an optional name suffix",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Hold joints at the logged read positions for this long after policy_t0, "
        "letting the base settle onto the floor (default: 0, disabled)",
    )
    args = parser.parse_args()

    with open(args.logfile, encoding="utf-8") as f:
        log = json.load(f)

    replay(log, view=args.view, log_name=args.log, settle_time=args.settle_time)


if __name__ == "__main__":
    main()
