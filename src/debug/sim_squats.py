# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Re-run a recorded squat in simulation under every bam actuator model.

Takes a real-robot log, seeds the simulation with the exact state the robot was in when the
policy went active, and runs ``squat_rl`` from there for N squats under each of the m1..m6
friction models. One log per model lands in ``sim_logs/`` in the same format
``src/robot_logger.py`` writes, so the results drop straight into the log tooling:

    uv run --group debug --group sim src/debug/squat_ref_error.py sim_logs/*.json
    uv run --group debug src/debug/plot_log.py logs/real.json sim_logs/one.json

This is not a replay. ``src/debug/sim_log.py`` feeds the recorded *goal* positions into the
motors, which reproduces the commands whatever the robot does; here the policy runs live and
closes its own loop, so the sim diverges from the real run exactly where the actuator model
is wrong. That is the point: the only thing changing between the six runs is the friction
model, so what differs between the logs is what that model is responsible for.

Three things are taken from the real log so the comparison starts honest:

  seed pose      the read joint angles at policy_t0, the trunk attitude from the log's
                 ``body_quat``, and a trunk height solved so the lowest sole corner rests on
                 the floor (MuJoCoController.reset_to_pose). Starting from the neutral
                 standing pose instead would give the policy half a squat of catching up
                 that the real run never had.
  supply voltage the mean motor voltage half a second into the run, fed to bam as its vin.
                 A squat on a 7.9 V pack and one on a 6.8 V pack are different plants, and
                 the current model is voltage-driven.
  clock          the policy's phase clock is driven by *simulated* time, not the wall clock,
                 so a headless run at 20x real time still squats at SQUAT_FREQUENCY.

The produced log's time axis is rewritten onto simulated time and its ``policy_t0`` set to
0.0: the policy is active from the first tick, so the log starts where the squat starts.

Headless and flat out by default. ``--view`` opens the MuJoCo viewer and paces the run in
real time, which costs ~40 s per model.

    uv run --group sim src/debug/sim_squats.py logs/2026-08-29_14-24-33_squat_m1_45.json
    uv run --group sim src/debug/sim_squats.py logs/run.json --models m1 m6 --squats 4
    uv run --group sim src/debug/sim_squats.py logs/run.json --view --models m3

Run it from the repository root: the policy is loaded from ``src/agents/`` by relative path.
"""

import sys
import time
import json
import argparse
from pathlib import Path

# src/ is not on the import path when this file is run by path, and everything below lives
# there (the same insert plot_log.py does).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from constants import MOTOR_TO_ID, BAM_VIN, SIM_SCENE_MJCF  # noqa: E402
from input.input_source import UserInput  # noqa: E402
from moves.move import MotorCommand  # noqa: E402
import moves.squat_rl as squat_rl  # noqa: E402
from moves.squat_rl import SQUAT_FREQUENCY, SquatRlMove  # noqa: E402
from observer import Observation, Observer  # noqa: E402
from robot_logger import RobotLogger  # noqa: E402
from sim.mujoco_controller import MuJoCoController  # noqa: E402

MJCF_PATH = SIM_SCENE_MJCF
OUTPUT_DIR = "sim_logs"

# The bundled bam parameter sets, in the order they add terms: m1 Coulomb, m2 + Stribeck,
# m3 + load-dependent, m4 + both, m5/m6 + directional. Sweeping all six is the point of the
# script, so it is the default.
MODELS = ("m1", "m2", "m3", "m4", "m5", "m6")

# Control period [s]. The scheduler runs at 50 Hz and the policy was trained against that
# rate, so the tick count comes from it rather than from the log's own (jittery) spacing.
CONTROL_DT = 0.02

# When to sample the supply voltage, in seconds after policy start. Early enough to be the
# battery this run actually had, late enough that the first-tick transient has passed.
VOLTAGE_SAMPLE_S = 0.5


def load_log(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tick_at(times: list[float], when: float) -> int:
    """Index of the log tick closest to `when` (seconds on the log's own clock)."""
    return int(np.argmin(np.abs(np.asarray(times, dtype=float) - when)))


def seed_state(log: dict, index: int) -> tuple[dict[str, float], tuple[float, float, float, float] | None]:
    """The joint angles and trunk attitude the robot was in at log tick `index`.

    Read positions, not goals: the goal is where the policy asked the robot to be, the read
    is where it was, and the second is the state a simulation has to start from. A joint
    with no reading on that tick is left out, and the seeded pose keeps the model's zero
    there rather than inventing a value.
    """
    angles = {
        name: float(log["position"][name][index])
        for name in MOTOR_TO_ID
        if log["position"].get(name) and log["position"][name][index] is not None
    }
    quat_channel = log.get("body_quat") or {}
    quat = tuple(
        quat_channel.get(axis, [None] * (index + 1))[index] for axis in ("w", "x", "y", "z")
    )
    if any(v is None for v in quat):
        return angles, None
    return angles, tuple(float(v) for v in quat)


def supply_voltage(log: dict, policy_t0: float) -> float | None:
    """Mean motor voltage `VOLTAGE_SAMPLE_S` into the run, or None if it was not recorded.

    Averaged across the motors rather than taken from one: they share a pack, and a single
    servo's reading is quantised coarsely enough that the mean is the better estimate of
    what the bus was actually at.
    """
    channel = log.get("voltage") or {}
    index = tick_at(log["time"], policy_t0 + VOLTAGE_SAMPLE_S)
    values = [
        float(channel[name][index])
        for name in MOTOR_TO_ID
        if channel.get(name) and channel[name][index] is not None
    ]
    return float(np.mean(values)) if values else None


def finalize_log(path: Path, times: list[float]) -> None:
    """Rewrite the produced log's time axis onto simulated time.

    RobotLogger stamps wall time, which is right live but meaningless for a headless run
    that finished 20x faster than the motion it simulated. The honest axis is the simulated
    one, and ``policy_t0`` is 0.0 because the policy is active from the first tick.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    aligned = [round(t, 4) for t in times[: len(data["time"])]]
    data["time"] = aligned
    data["metadata"]["duration_s"] = round(aligned[-1] - aligned[0], 3) if aligned else 0.0
    data["metadata"]["policy_t0"] = 0.0
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def simulate(
    log: dict,
    source_name: str,
    model: str,
    n_ticks: int,
    vin: float,
    view: bool,
    out_dir: str,
) -> Path | None:
    """Run the squat policy under one bam model and write its log. Returns the path."""
    angles, quat = seed_state(log, tick_at(log["time"], log["metadata"]["policy_t0"]))

    controller = MuJoCoController(
        mjcf_path=MJCF_PATH,
        enable_viewer=view,
        bam_model=model,
        bam_vin=vin,
    )
    height = controller.reset_to_pose(angles, quat)

    observer = Observer(controller)
    # Match the channels a real run records with [u] on, so the produced log can be compared
    # against one without holes.
    observer.observe_voltage = True
    observer.observe_current = True

    move = SquatRlMove(controller)

    logger = RobotLogger(log_dir=out_dir)
    # Which policy ran is stamped alongside which plant it ran on. A log records neither
    # otherwise, and a sweep over models is meaningless if the agent silently differed.
    source = controller.get_log_metadata() | {"agent": squat_rl.AGENT_NAME, "agent_run": move.describe()}
    path = logger.start(f"{source_name}_sim_{model}", source)
    user_input = UserInput()
    times: list[float] = []

    print(
        f"  {model}: seeded at trunk z {height * 1000:.1f} mm, vin {vin:.2f} V, "
        f"{n_ticks} ticks -> {path.name}"
    )

    wall_start = time.perf_counter()
    for tick in range(n_ticks):
        if not controller.is_running():
            print("  viewer closed, stopping.")
            break

        sim_time = tick * CONTROL_DT
        robot_state = observer.read_state(CONTROL_DT)
        # The policy times its phase off this, and a headless run's wall clock has nothing
        # to do with the motion being simulated. Overriding it is what keeps the squat at
        # SQUAT_FREQUENCY however fast the physics runs.
        robot_state.time_s = sim_time
        obs = Observation(robot_state=robot_state, user_input=user_input)

        command = MotorCommand()
        if tick == 0:
            move.on_start(obs, command)
        move.step(obs, command)

        controller.sync_write_goal_position(
            [MOTOR_TO_ID[name] for name in command.target_angles],
            list(command.target_angles.values()),
        )

        logger.record(robot_state, command.target_angles, user_input.velocity, move.height_target_command)
        logger.mark_policy_start("squat_rl")
        times.append(sim_time)

        if view:
            sleep_s = wall_start + sim_time - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    written = logger.stop()
    controller.close()
    if written is not None:
        finalize_log(written, times)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("logfile", type=Path, help="real-robot log to take the starting state from")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS),
        metavar="M",
        help=f"bam friction models to run (default: {' '.join(MODELS)})",
    )
    parser.add_argument(
        "--squats", type=float, default=10.0, help="squat cycles to simulate per model (default: 10)"
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="open the MuJoCo viewer and run in real time (headless and flat out otherwise)",
    )
    parser.add_argument("--out", default=OUTPUT_DIR, help=f"output directory (default: {OUTPUT_DIR})")
    parser.add_argument(
        "--agent",
        default=None,
        metavar="FILE.onnx",
        help="policy to run from src/agents/, overriding the one squat_rl.py names. The "
        "source log does not record which policy produced it, so this has to be given by "
        "hand when re-simulating anything but the currently configured one",
    )
    parser.add_argument(
        "--vin",
        type=float,
        default=None,
        help="override the supply voltage [V] instead of reading it from the log",
    )
    args = parser.parse_args()

    if args.agent is not None:
        # SquatRlMove reads the module global in its constructor, so setting it here is what
        # a --agent flag can do without touching the move.
        squat_rl.AGENT_NAME = args.agent

    log = load_log(args.logfile)
    policy_t0 = log.get("metadata", {}).get("policy_t0")
    if policy_t0 is None:
        raise SystemExit(
            f"{args.logfile.name} has no policy_t0 — there is no 'moment the squat started' "
            "to seed from. Record the log before starting the policy."
        )

    vin = args.vin if args.vin is not None else supply_voltage(log, policy_t0)
    if vin is None:
        vin = BAM_VIN
        print(f"No voltage recorded in {args.logfile.name} (press [u] during a run) — using {vin} V.")

    _, quat = seed_state(log, tick_at(log["time"], policy_t0))
    if quat is None:
        print("No body_quat in this log — seeding upright, so any real lean is lost.")

    n_ticks = round(args.squats / SQUAT_FREQUENCY / CONTROL_DT)
    # The logger prefixes its own timestamp, so the source's is dropped to keep names short.
    source_name = args.logfile.stem
    stripped = source_name.split("_", 2)
    if len(stripped) == 3 and stripped[0][:2] == "20":
        source_name = stripped[2]

    print(
        f"{args.logfile.name}: policy_t0 {policy_t0:.3f} s, supply {vin:.2f} V, "
        f"agent {squat_rl.AGENT_NAME}\n"
        f"simulating {args.squats:g} squats ({n_ticks} ticks, {n_ticks * CONTROL_DT:.0f} s) "
        f"per model {'in the viewer' if args.view else 'headless'}"
    )

    written = []
    for model in args.models:
        try:
            path = simulate(log, source_name, model, n_ticks, vin, args.view, args.out)
        except Exception as error:  # a bad model name should not lose the other five
            print(f"  {model}: failed — {error}")
            continue
        if path is not None:
            written.append(path)

    if not written:
        raise SystemExit("Nothing was written.")
    print(f"\n{len(written)} logs in {args.out}/. Score them with:")
    print(f"  uv run --group debug --group sim src/debug/squat_ref_error.py {args.out}/*.json")


if __name__ == "__main__":
    main()
