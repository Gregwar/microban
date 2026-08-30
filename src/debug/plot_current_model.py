# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Check the bam current model against the current the servos actually reported.

Loads a session log (src/robot_logger.py) and replays the bam XL330 firmware + motor model
over the log's own data — the commanded target, the measured position and the measured
velocity of every tick — to predict the current each servo drew. That prediction is plotted
against the recorded `current` channel, so what the figure shows is how far the model is
from the robot on a run the robot really did.

Nothing is simulated: the model is evaluated open loop on recorded state, so a mismatch is
the current model being wrong and not a trajectory that drifted apart. What it cannot see is
what happened *inside* a tick: the firmware closes its loop at a few kHz while the log only
holds the error at the top of each 20 ms tick, which is the largest error of the tick. The
duty cycle — and with it the prediction — is therefore an upper bound within the tick.
Running this on a *sim* log measures exactly that: the same bam model is on both sides, so
what is left is the cost of evaluating it once per tick (~2.5x on the bus current of a
squat) rather than every physics step.

The model variant is picked at runtime, `--model m1` ... `m6` (several may be given, and
each gets a curve of its own). The variants differ in their friction terms, which do not
enter the current equation, but each carries its own fitted `kt` and `R`, which do — m1 and
m6 are ~7% apart in kt and ~40% in R.

The servo reports current on the input side of its H-bridge, so `bus` is the comparable
quantity and the default; `--current phase` plots the winding current instead, which is
what the overcurrent proxy in src/scheduler.py uses and is larger by 1/duty.

    uv run --group debug --group sim src/debug/plot_current_model.py
    uv run --group debug --group sim src/debug/plot_current_model.py logs/a.json --model m4
    uv run --group debug --group sim src/debug/plot_current_model.py logs/a.json --model m1 m6
    uv run --group debug --group sim src/debug/plot_current_model.py logs/a.json --current phase

The log must have been recorded with [u] on, otherwise there is no measured current to
compare against.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Same trick as plot_log.py: this script is run by path, so src/ is not on the import path
# and `constants` would not resolve. plot_log itself lives right here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bam.model import load_model
from bam.simulate import fractional_delay_shift

from constants import BAM_VIN, BAM_MAX_CURRENT, KP_DEFAULT, KP_RL

# Sits next to this file, and reused rather than copied: the log loading, the Σ|I| aggregate
# and the sliding-window MAE are all the ones plot_log.py already draws, so a figure from
# either script shows the same quantity computed the same way.
from plot_log import LOG_DIR, latest_log, load, policy_t0, series, sliding_mean

# The motor every joint of the robot uses; the bam params are fitted per motor.
MOTOR = "xl330"

MODELS = ("m1", "m2", "m3", "m4", "m5", "m6")

# Policies that drop the firmware gain to KP_RL for as long as they run (see the moves in
# src/moves/). The gain scales the duty cycle and therefore the whole current estimate, so
# using the wrong one is not a detail. It is not in the log, but the policy name is, and the
# policy start is stamped as policy_t0 — which is enough to reconstruct when it changed.
RL_POLICIES = frozenset({"walk", "squat_rl", "getup", "leftstand"})

# Width of the sliding window the model's error is averaged over, matching the MAE row
# plot_log.py draws when comparing two logs. At 5 s it spans a few squat or walk cycles.
ERROR_WINDOW_S = 5.0

# The measurement is the reference here, so it takes black and the models take colours.
MEASURED_STYLE = {"color": "black", "linewidth": 0.9}

# The first ticks of a policy are a step onto a target far from the current pose, which the
# model turns into a spike an order of magnitude above everything that follows. Left to
# autoscale it flattens the whole run into a band a few pixels tall, so the y axes are sized
# to hold this percentile of the samples instead and the transient is allowed off the top.
YLIM_PERCENTILE = 99.0
YLIM_HEADROOM = 1.15

# Per-joint grid, 19 joints over 4 rows.
GRID_COLS = 5

# One servo's current is quantised to 1 mA and swings between zero and a couple of LSB from
# tick to tick, which at 3000 ticks in a panel this size draws as a solid block. The
# per-joint panels therefore show a short sliding mean of |I| — long enough to make the
# panel readable, short enough to keep the shape of a squat or a stride.
JOINT_SMOOTH_S = 1.0


def resolve_kp(log: dict, times: list[float], override: int | None) -> np.ndarray:
    """Firmware P gain per tick, as a column vector broadcastable over the joints.

    An RL policy writes KP_RL to every servo when it starts and KP_DEFAULT back when it
    stops, so a log of an RL run holds both regimes: KP_DEFAULT until policy_t0, KP_RL
    after. The tail after the policy stopped is not recoverable (nothing stamps that), so
    it stays at KP_RL and reads too soft — the run is normally stopped with the log.
    """
    if override is not None:
        return np.full((len(times), 1), float(override))

    policy = log.get("metadata", {}).get("policy")
    t0 = policy_t0(log)
    if policy not in RL_POLICIES or t0 is None:
        return np.full((len(times), 1), float(KP_DEFAULT))

    kp = np.where(np.asarray(times) < t0, float(KP_DEFAULT), float(KP_RL))
    return kp.reshape(-1, 1)


def resolve_vin(log: dict, joints: list[str], override: float | None) -> np.ndarray:
    """Supply voltage per tick, as a column vector broadcastable over the joints.

    The pack sags under load — a couple of tenths of a volt over a squat — and the duty
    cycle is turned into a voltage by exactly this number, so the logged reading is used
    when there is one. The motors share the supply, so the ticks are averaged over the
    servos that answered; ticks with no reading (or a log recorded without [u]) fall back
    to the nominal BAM_VIN.
    """
    if override is not None:
        return np.full((len(log["time"]), 1), float(override))

    voltage = log.get("voltage") or {}
    columns = [series(voltage[j]) for j in joints if j in voltage]
    if not columns:
        return np.full((len(log["time"]), 1), BAM_VIN)

    stacked = np.asarray(columns, dtype=float).T
    read = ~np.isnan(stacked)
    count = np.count_nonzero(read, axis=1)
    # Summed over the servos that answered and divided by how many did, rather than
    # np.nanmean, which warns its way through every tick of a log recorded without [u].
    summed = np.sum(np.where(read, stacked, 0.0), axis=1)
    mean = np.where(count > 0, summed / np.maximum(count, 1), BAM_VIN)
    return mean.reshape(-1, 1)


def model_currents(
    log: dict,
    joints: list[str],
    times: list[float],
    name: str,
    kp: np.ndarray,
    vin: np.ndarray,
    command_delay: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint (bus, phase) current predicted by bam model `name`, shape (ticks, joints).

    Evaluated in one vectorised call rather than tick by tick: bam's control law is
    elementwise and broadcasts, so the whole log goes through it as a (ticks, joints)
    array with `kp` and `vin` as columns that vary down the ticks. A dropped read is NaN
    in the inputs and stays NaN in the output.

    The phase current is what flows through the winding, `(duty * vin - kt * dq) / R`; the
    bus current is `duty` times that. The H-bridge is a buck stage — the winding draws
    continuously while the battery only sources during the PWM on-time — and it is the bus
    side the XL330 senses and reports, so the two are worth keeping apart (see also
    MujocoController._bus_current, which logs the same quantity in sim).
    """
    model = load_model(motor_name=MOTOR, model=name)
    actuator = model.actuator
    actuator.kp = kp
    actuator.vin = vin
    # The firmware current limiter, modelled by bam as a bound on the duty cycle. The XL330
    # actuator carries no limit of its own, so it comes from the robot's own constant.
    actuator.max_current = BAM_MAX_CURRENT

    q = np.asarray([series(log["position"][j]) for j in joints], dtype=float).T
    dq = np.asarray([series(log["velocity"][j]) for j in joints], dtype=float).T
    q_target = np.asarray([series(log["target_position"][j]) for j in joints], dtype=float).T

    # The fitted transport lag between writing a goal position and the servo acting on it
    # (bus round-trip + firmware period), a fraction of a tick at 50 Hz. Shifting the goal
    # by it is what bam does when it rolls a model over a log, and it lines the predicted
    # peaks up with the measured ones instead of leaving them half a tick early.
    if command_delay and model.command_delay.value > 0.0:
        dt = (times[-1] - times[0]) / max(len(times) - 1, 1)
        q_target = fractional_delay_shift(q_target, model.command_delay.value, dt)

    volts = actuator.compute_control(q_target, q, dq, dt=0.0)
    duty = np.asarray(actuator.duty_cycle, dtype=float)
    # torque = kt * phase current, by definition of the DC motor equation bam applies, so
    # this is bam's own current rather than the formula written out a second time here.
    phase = actuator.compute_torque(volts, True, q, dq) / model.kt.value
    return duty * phase, phase


def robust_top(traces: list[np.ndarray], mask: np.ndarray) -> float | None:
    """Upper y limit holding YLIM_PERCENTILE of the plotted samples, None to leave autoscale.

    None whenever the limit would not actually cut anything, so an axis with no transient
    keeps matplotlib's own margins rather than a slightly tighter lookalike.
    """
    stacked = np.concatenate([np.abs(np.asarray(t, dtype=float))[mask] for t in traces])
    stacked = stacked[~np.isnan(stacked)]
    if stacked.size == 0:
        return None
    top = float(np.percentile(stacked, YLIM_PERCENTILE)) * YLIM_HEADROOM
    return top if 0.0 < top < float(np.max(stacked)) else None


def total(per_joint: np.ndarray) -> np.ndarray:
    """Σ|I| over the joints, NaN on ticks where no joint had a reading.

    Same aggregate as the `current` row of plot_log.py: the motors share one pack, so the
    sum is what the battery sees, and the sign is a drive direction rather than a load.
    """
    valid = np.count_nonzero(~np.isnan(per_joint), axis=1)
    summed = np.nansum(np.abs(per_joint), axis=1)
    return np.where(valid > 0, summed, np.nan)


def report(
    name: str, joints: list[str], measured_joints: np.ndarray, predicted_joints: np.ndarray
) -> None:
    """Print how far this model landed, overall and on its worst joints."""
    measured, predicted = total(measured_joints), total(predicted_joints)
    both = ~np.isnan(measured) & ~np.isnan(predicted)
    if not both.any():
        print(f"{name}: nothing to compare — no tick has both a reading and a prediction.")
        return

    error = predicted[both] - measured[both]
    mean_measured = float(np.mean(measured[both]))
    mean_predicted = float(np.mean(predicted[both]))
    ratio = mean_predicted / mean_measured if mean_measured else float("nan")
    print(
        f"{name}: measured {mean_measured:.3f} A, model {mean_predicted:.3f} A "
        f"(x{ratio:.2f}), MAE {float(np.mean(np.abs(error))):.3f} A, "
        f"bias {float(np.mean(error)):+.3f} A"
    )

    per_error = np.abs(np.abs(predicted_joints) - np.abs(measured_joints))
    seen = ~np.isnan(per_error)
    counted = np.count_nonzero(seen, axis=0)
    per_joint = np.where(
        counted > 0,
        np.sum(np.where(seen, per_error, 0.0), axis=0) / np.maximum(counted, 1),
        np.nan,
    )
    worst = np.argsort(np.where(np.isnan(per_joint), -1.0, per_joint))[::-1][:5]
    listed = ", ".join(
        f"{joints[i]} {per_joint[i]:.3f}" for i in worst if per_joint[i] == per_joint[i]
    )
    print(f"      worst joints (MAE, A): {listed}")


def build_figures(
    path: Path, joints: list[str], times: list[float], aligned: bool,
    measured_joints: np.ndarray, predictions: dict[str, np.ndarray], kind: str,
) -> None:
    """Total-current figure plus the per-joint grid, for every model in `predictions`."""
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(predictions)}
    x_label = "Time since policy start (s)" if aligned else "Time (s)"

    measured_total = total(measured_joints)
    # The transient is measured against the policy start, so when the log is aligned the
    # limits are read off the policy run itself rather than off whatever the robot was
    # doing beforehand.
    steady = np.asarray(times) >= 0.0 if aligned else np.ones(len(times), dtype=bool)

    fig, (ax_total, ax_error) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle(f"{path.name}  —  bam {MOTOR} current model vs measurement ({kind} current)")

    ax_total.plot(times, measured_total, label="measured", **MEASURED_STYLE)
    for name, per_joint in predictions.items():
        ax_total.plot(times, total(per_joint), color=colors[name], linewidth=1.0, label=f"bam {name}")
    ax_total.set_ylabel(f"Σ |{kind} current| (A)", fontsize=9)
    ax_total.legend(loc="upper right", fontsize=8)

    top = robust_top([measured_total] + [total(p) for p in predictions.values()], steady)
    if top is not None:
        ax_total.set_ylim(0.0, top)
        ax_total.text(
            0.004, 0.93, f"y clipped at the {YLIM_PERCENTILE:g}th percentile",
            transform=ax_total.transAxes, va="top", fontsize=7, color="0.5",
        )

    for name, per_joint in predictions.items():
        error = np.abs(total(per_joint) - measured_total)
        ax_error.plot(
            times, sliding_mean(times, list(error), ERROR_WINDOW_S),
            color=colors[name], linewidth=1.6, label=f"bam {name}",
        )
    ax_error.set_ylabel(f"MAE ({ERROR_WINDOW_S:g} s window) (A)", fontsize=9)
    ax_error.set_xlabel(x_label, fontsize=9)
    ax_error.legend(loc="upper right", fontsize=8)

    for ax in (ax_total, ax_error):
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if aligned:
            ax.axvline(0.0, color="0.6", linewidth=0.8, zorder=0)

    # Per joint, so a model that is only wrong on the loaded joints can be told apart from
    # one that is wrong everywhere. |I| again, to match the aggregate above.
    def smooth(values: np.ndarray) -> list[float]:
        return sliding_mean(times, list(np.abs(values)), JOINT_SMOOTH_S)

    rows = -(-len(joints) // GRID_COLS)
    grid, axes = plt.subplots(rows, GRID_COLS, figsize=(15, 2.2 * rows), sharex=True)
    clipped = False
    for i, ax in enumerate(axes.flat):
        if i >= len(joints):
            ax.set_visible(False)
            continue
        traces = [smooth(measured_joints[:, i])]
        ax.plot(times, traces[0], **MEASURED_STYLE)
        for name, per_joint in predictions.items():
            traces.append(smooth(per_joint[:, i]))
            ax.plot(times, traces[-1], color=colors[name], linewidth=0.9)
        panel_top = robust_top(traces, steady)
        if panel_top is not None:
            ax.set_ylim(0.0, panel_top)
            clipped = True
        ax.set_title(joints[i], fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    note = f", y clipped at the {YLIM_PERCENTILE:g}th percentile" if clipped else ""
    grid.suptitle(
        f"{path.name}  —  per joint |{kind} current| (A), "
        f"{JOINT_SMOOTH_S:g} s sliding mean{note}"
    )
    axes.flat[0].legend(
        handles=[plt.Line2D([], [], **MEASURED_STYLE)]
        + [plt.Line2D([], [], color=colors[n], linewidth=0.9) for n in predictions],
        labels=["measured"] + [f"bam {n}" for n in predictions],
        loc="upper right", fontsize=7,
    )
    grid.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", nargs="?", type=Path, help=f"log file (default: newest in {LOG_DIR}/)")
    parser.add_argument(
        "--model", nargs="+", default=["m6"], choices=MODELS, metavar="mN",
        help=f"bam model variant(s) to evaluate, one curve each: {', '.join(MODELS)} (default: m6)",
    )
    parser.add_argument(
        "--current", choices=("bus", "phase"), default="bus",
        help="which current to predict: bus, what the servo reports (default), or phase, "
        "what flows in the winding",
    )
    parser.add_argument(
        "--kp", type=int, default=None,
        help=f"force the firmware P gain (default: {KP_DEFAULT}, or {KP_RL} while an RL policy ran)",
    )
    parser.add_argument(
        "--vin", type=float, default=None,
        help=f"force the supply voltage [V] (default: the logged reading, else {BAM_VIN})",
    )
    parser.add_argument(
        "--no-command-delay", action="store_true",
        help="do not shift the goal by the model's fitted command delay",
    )
    args = parser.parse_args()

    path = args.log or latest_log()
    log = load(path)
    joints = list(log["position"])

    measured_joints = np.asarray(
        [series(log["current"][j]) for j in joints], dtype=float
    ).T
    if np.isnan(measured_joints).all():
        raise SystemExit(
            f"No current recorded in {path.name} — press [u] during the run to read it."
        )

    # Policy-relative time, like plot_log.py, so a figure from either script reads on the
    # same axis. Nothing is compared across logs here, so a missing stamp is not fatal.
    t0 = policy_t0(log)
    aligned = t0 is not None
    times = [t - (t0 if aligned else 0.0) for t in log["time"]]

    kp = resolve_kp(log, times, args.kp)
    vin = resolve_vin(log, joints, args.vin)
    print(
        f"{path.name}: {len(times)} ticks, policy {log.get('metadata', {}).get('policy')}, "
        f"kp {sorted({int(v) for v in kp.ravel()})}, vin {float(vin.mean()):.2f} V"
    )

    predictions = {}
    for name in args.model:
        bus, phase = model_currents(
            log, joints, times, name, kp, vin, not args.no_command_delay
        )
        predictions[name] = bus if args.current == "bus" else phase
        report(name, joints, measured_joints, predictions[name])

    build_figures(path, joints, times, aligned, measured_joints, predictions, args.current)
    plt.show()


if __name__ == "__main__":
    main()
