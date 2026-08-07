# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Measure the head servo's command -> motion delay, on the robot, one motor at a time.

Standalone: it needs `rustypot` and nothing else from this repo, so it can be copied to the
robot and run on its own. It talks only to the head (ID 51) -- every transaction is a
single-motor read or write, never a sync operation over the bus -- so the other 18 motors
are neither addressed nor disturbed, and the poll rate is set by one round trip rather than
by nineteen.

WHAT IT MEASURES. For each trial: settle the head, timestamp a goal-position write, then
poll the servo as fast as the bus allows and find when the reading first changes. The
answer splits into two numbers that are worth keeping apart:

    order taken into account   present_current leaves its resting value: the firmware has
                               applied the new goal and is driving the motor, before the
                               output has gone anywhere. Needs --with-current.
    started moving             present_position leaves its resting count: the joint has
                               physically moved. This is the number most people mean.

WHY NOT JUST READ IT OFF A SESSION LOG. A [l] log samples at the control rate, ~20 ms, so
it cannot place an onset to better than a tick and the estimate ends up leaning on a fitted
model of the response shape (see src/debug/bench_delay.py, which does exactly that and
reports about +/- 7 ms for its trouble). Polling one motor at 2 Mbaud is a round trip of a
few hundred microseconds, which resolves the onset directly and makes the model assumption
unnecessary.

THREE BIASES, ALL HANDLED EXPLICITLY:

  the read is not instantaneous   the value describes the joint at some instant between the
                                  instruction leaving and the status packet arriving. Both
                                  are timestamped; the midpoint is used and the half-width
                                  reported as a systematic rather than ignored.

  one count is 0.088 deg          "first changed reading" is not "started moving" -- the
                                  joint must first cross a whole count, which at a realistic
                                  acceleration takes a few ms. Reported as-is (model free),
                                  and separately corrected by extrapolating the early rise
                                  back to zero, which the fast sampling makes well posed.

  return delay time               the servo waits RDT/2 microseconds before answering. It
                                  is on the EEPROM, defaults to 500 us, and lands directly
                                  in the round trip. Read out and reported; a large value is
                                  called out because it inflates the measurement floor.

The floor is measured, not assumed: a calibration pass times bare round trips with the joint
at rest, and a null trial re-writes the goal the servo already holds and confirms nothing is
detected. If the null trial detects motion, the threshold is in the noise and the numbers
that follow are not trustworthy.

    uv run src/debug/bench_head_delay.py                       # 20 trials, +/- 20 deg
    uv run src/debug/bench_head_delay.py --with-current        # also time the current rise
    uv run src/debug/bench_head_delay.py --trials 50 --json out.json
    uv run src/debug/bench_head_delay.py --port /dev/ttyUSB0 --amplitude-deg 30

Torque is disabled on the head on the way out, including on Ctrl-C.
"""

import os
import sys
import json
import math
import time
import struct
import argparse
import statistics

from rustypot import Xl330PyController

# The head, and only the head. Every call below passes this one id.
HEAD_ID = 51

DEFAULT_PORT = "/dev/ttyAMA0"
DEFAULT_BAUDRATE = 2_000_000

# XL330 control table. The block is Present Current (126, i16), Present Velocity (128, i32)
# and Present Position (132, i32) back to back, so --with-current costs six extra bytes on
# the wire rather than a second round trip.
STATE_ADDR = 126
STATE_LEN = 10
_STATE_STRUCT = struct.Struct("<hii")

POSITION_TICKS_PER_TURN = 4096
COUNT_DEG = 360.0 / POSITION_TICKS_PER_TURN
CURRENT_UNIT_A = 0.001

# Motion is called at the first whole count of movement, but only once it *stays* there for
# CONFIRM_SAMPLES more reads. A joint resting exactly on a count boundary will occasionally
# dither across it -- observed as a lone 1820 -> 1821 -> 1820 blip -- and a bare threshold
# reads that as motion several ms early. Confirming costs nothing, because the reported time
# is still that of the first sample over the line; and real motion here is accelerating, so
# it never falls back. Raising the threshold instead would work too, but it would make the
# joint travel further before being noticed, which is the bias the fit below has to undo.
MOTION_THRESHOLD_COUNTS = 1
CONFIRM_SAMPLES = 5
# Current is noisier and idles a few mA off zero, so it gets a threshold with real headroom.
CURRENT_THRESHOLD_A = 0.05

# Counts spanned by the constant-acceleration fit that corrects the one-count bias. Starts
# above the threshold and stops well before the joint is up to speed, where the acceleration
# is no longer constant.
FIT_COUNTS = (1, 12)
FIT_MIN_SAMPLES = 4

# A trial captures this long after the write, which at a few hundred microseconds a round
# trip is thousands of samples -- far more than needed, but the capture is cheap and having
# the tail makes the traces worth dumping with --json.
DEFAULT_CAPTURE_MS = 60.0
DEFAULT_SETTLE_S = 0.6
DEFAULT_AMPLITUDE_DEG = 20.0
DEFAULT_TRIALS = 20

CALIBRATION_READS = 400


def ticks_to_deg(ticks: int) -> float:
    return (ticks - POSITION_TICKS_PER_TURN // 2) * COUNT_DEG


def deg_to_ticks(deg: float) -> int:
    return round(deg / COUNT_DEG) + POSITION_TICKS_PER_TURN // 2


def _unstuff(data: bytes) -> bytes:
    """Undo Protocol 2.0 byte stuffing.

    A payload that happens to contain 0xFF 0xFF 0xFD gets an extra 0xFD inserted so it
    cannot be mistaken for a packet header. Negative currents and velocities produce that
    pattern routinely once the joint is moving, which is exactly when this script is
    reading, so the raw block path has to undo it.
    """
    if b"\xff\xff\xfd\xfd" not in data:
        return data
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        out.append(data[i])
        if (out[-1] == 0xFD and len(out) >= 3 and out[-2] == 0xFF and out[-3] == 0xFF
                and i + 1 < n and data[i + 1] == 0xFD):
            i += 1
        i += 1
    return bytes(out)


def _one(value):
    """Unwrap a single-motor read.

    The driver's per-motor reads return a one-element list rather than a scalar, mirroring
    the sync calls. Unwrapped in one place so the rest of the script deals in numbers.
    """
    return value[0] if isinstance(value, list) else value


class HeadProbe:
    """Every bus transaction this script makes, all of them single-motor."""

    def __init__(self, port: str, baudrate: int, motor_id: int, with_current: bool) -> None:
        self.io = Xl330PyController(serial_port=port, baudrate=baudrate, timeout=0.05)
        self.id = motor_id
        self.with_current = with_current

    def read_state(self) -> tuple[float, float, int, float]:
        """One sample: (t_before, t_after, position_ticks, current_a).

        Both timestamps are returned because the servo latched the value somewhere between
        them, and at these speeds that interval is a real part of the error budget.

        Position-only is the default because it is the fastest and most robust path the
        driver offers; the raw block is only worth its extra bytes when current is wanted.
        """
        if self.with_current:
            t0 = time.perf_counter()
            block = self.io.read_raw_data(self.id, STATE_ADDR, STATE_LEN)
            t1 = time.perf_counter()
            data = _unstuff(bytes(bytearray(block)))
            if len(data) != STATE_LEN:
                raise RuntimeError(f"short state block: {len(data)} bytes")
            current_raw, _velocity, position = _STATE_STRUCT.unpack(data)
            return t0, t1, position, current_raw * CURRENT_UNIT_A

        t0 = time.perf_counter()
        position = self.io.read_raw_present_position(self.id)
        t1 = time.perf_counter()
        return t0, t1, int(_one(position)), math.nan

    def write_goal_ticks(self, ticks: int) -> tuple[float, float]:
        """Send the order. Returns (t_before, t_after) around the write itself.

        t_after is when the instruction packet is out, so it is the moment the order can
        first have been acted on -- delays are measured from it. The gap to t_before is the
        write's own transmission, reported so it is visible rather than folded in silently.
        """
        t0 = time.perf_counter()
        self.io.write_raw_goal_position(self.id, ticks)
        t1 = time.perf_counter()
        return t0, t1

    def settings(self) -> dict:
        """The servo settings that move the answer, read once and reported."""
        return {
            "firmware": _one(self.io.read_firmware_version(self.id)),
            "return_delay_time_us": _one(self.io.read_return_delay_time(self.id)) * 2,
            "status_return_level": _one(self.io.read_status_return_level(self.id)),
            "position_p_gain": _one(self.io.read_position_p_gain(self.id)),
            "position_d_gain": _one(self.io.read_position_d_gain(self.id)),
            "profile_velocity": _one(self.io.read_profile_velocity(self.id)),
            "profile_acceleration": _one(self.io.read_profile_acceleration(self.id)),
            "current_limit": _one(self.io.read_current_limit(self.id)),
        }

    def torque(self, on: bool) -> None:
        self.io.write_torque_enable(self.id, bool(on))


def calibrate_round_trip(probe: HeadProbe, n: int) -> dict:
    """Time bare reads with the joint at rest: the floor of everything below.

    Nothing measured by this script can be sharper than a round trip, so it is measured
    rather than assumed, and the resting spread doubles as a check that the encoder really
    is quiet enough for a one-count threshold to mean something.
    """
    durations, positions = [], []
    for _ in range(n):
        t0, t1, ticks, _ = probe.read_state()
        durations.append((t1 - t0) * 1e3)
        positions.append(ticks)
    durations.sort()
    return {
        "n": n,
        "median_ms": statistics.median(durations),
        "p05_ms": durations[int(0.05 * n)],
        "p95_ms": durations[int(0.95 * n)],
        "max_ms": durations[-1],
        "rate_hz": 1e3 / statistics.median(durations),
        "rest_spread_counts": max(positions) - min(positions),
    }


def run_trial(probe: HeadProbe, goal_ticks: int, capture_ms: float) -> dict:
    """Write one goal and poll flat out until `capture_ms` has elapsed."""
    rest_t0, rest_t1, rest_ticks, rest_current = probe.read_state()

    t_write0, t_write1 = probe.write_goal_ticks(goal_ticks)

    samples = []
    deadline = t_write1 + capture_ms * 1e-3
    while time.perf_counter() < deadline:
        try:
            t0, t1, ticks, current = probe.read_state()
        except Exception:
            continue  # a dropped frame costs one sample, not the trial
        samples.append((t0 - t_write1, t1 - t_write1, ticks, current))

    return {
        "rest_ticks": rest_ticks,
        "rest_current_a": rest_current,
        "rest_age_ms": (t_write1 - rest_t1) * 1e3,
        "write_ms": (t_write1 - t_write0) * 1e3,
        "goal_ticks": goal_ticks,
        "samples": samples,
    }


def _crossing(samples, rest_ticks: int, sign: float, threshold: int):
    """First sample that reaches `threshold` counts toward the step and stays there.

    Returns (t_before, t_after, index) of that first sample, or None. `sign` of 0 means
    "either direction", which is what the null trial wants: it is looking for any movement
    at all, and a drift the wrong way is just as much a false detection.
    """
    def displaced(sample) -> int:
        delta = sample[2] - rest_ticks
        return abs(delta) if sign == 0.0 else delta * sign

    for i, sample in enumerate(samples):
        if displaced(sample) < threshold:
            continue
        confirm = samples[i + 1:i + 1 + CONFIRM_SAMPLES]
        if len(confirm) < CONFIRM_SAMPLES:
            return None  # ran out of capture before it could be confirmed
        if all(displaced(s) >= threshold for s in confirm):
            return sample[0], sample[1], i
    return None


def _current_crossing(samples, rest_current: float, sign: float):
    for t0, t1, _ticks, current in samples:
        if not math.isnan(current) and (current - rest_current) * sign >= CURRENT_THRESHOLD_A:
            return t0, t1
    return None


def _fit_onset_ms(samples, rest_ticks: int, sign: float, start: int) -> float | None:
    """Extrapolate the early rise back to zero displacement.

    Crossing the first count is not the same event as starting to move: the joint has to
    travel 0.088 deg to get there, which at a realistic acceleration is a few ms of genuine
    motion that the threshold cannot see. From rest under near-constant torque displacement
    goes as (t - onset)^2, so sqrt(displacement) is linear in time and a straight line
    through the early samples hits zero at the onset. Fitting sqrt rather than the
    displacement keeps the fit weighted toward the samples nearest the onset, which are the
    ones that carry the information.

    Starts at the confirmed crossing, so a pre-motion encoder blip cannot anchor the line.
    """
    lo, hi = FIT_COUNTS
    xs, ys = [], []
    for t0, t1, ticks, _current in samples[start:]:
        counts = (ticks - rest_ticks) * sign
        if lo <= counts <= hi:
            xs.append(0.5 * (t0 + t1) * 1e3)
            ys.append(math.sqrt(counts))
    if len(xs) < FIT_MIN_SAMPLES:
        return None

    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0 or sxy <= 0.0:
        return None
    slope = sxy / sxx
    return mx - my / slope


def analyse(trial: dict, null: bool = False) -> dict:
    """Turn one trial's samples into the delays."""
    sign = 0.0 if null else math.copysign(1.0, trial["goal_ticks"] - trial["rest_ticks"])
    samples = trial["samples"]

    motion = _crossing(samples, trial["rest_ticks"], sign, MOTION_THRESHOLD_COUNTS)
    current = None if null else _current_crossing(samples, trial["rest_current_a"], sign)
    intervals = [b[0] - a[1] for a, b in zip(samples, samples[1:])] if len(samples) > 1 else []
    onset = (
        None if (null or motion is None)
        else _fit_onset_ms(samples, trial["rest_ticks"], sign, motion[2])
    )

    return {
        "motion_ms": None if motion is None else 0.5 * (motion[0] + motion[1]) * 1e3,
        "motion_bracket_ms": None if motion is None else 0.5 * (motion[1] - motion[0]) * 1e3,
        "current_ms": None if current is None else 0.5 * (current[0] + current[1]) * 1e3,
        "onset_ms": onset,
        "samples": len(samples),
        "gap_ms": (max(intervals) * 1e3) if intervals else math.nan,
        "write_ms": trial["write_ms"],
        "direction": "up" if sign > 0 else "down",
    }


def run_campaign(probe: HeadProbe, args, amplitude_deg: float, verbose: bool) -> tuple[list, list]:
    """Settle, then alternate steps of +/- `amplitude_deg`, returning (trials, analyses)."""
    up_ticks = deg_to_ticks(amplitude_deg)
    down_ticks = deg_to_ticks(-amplitude_deg)

    probe.write_goal_ticks(down_ticks)
    time.sleep(max(args.settle_s, 1.0))

    trials, analyses = [], []
    for i in range(args.trials):
        goal = up_ticks if i % 2 == 0 else down_ticks
        trial = run_trial(probe, goal, args.capture_ms)
        result = analyse(trial)
        trials.append(trial)
        analyses.append(result)
        if verbose:
            detected = "  no motion detected" if result["motion_ms"] is None else (
                f"  motion {result['motion_ms']:6.2f} ms"
                + (f"   onset {result['onset_ms']:6.2f} ms" if result["onset_ms"] is not None else "   onset n/a")
                + (f"   current {result['current_ms']:6.2f} ms" if result["current_ms"] is not None else "")
            )
            print(f"  trial {i + 1:3d} {result['direction']:<5}{detected}   "
                  f"({result['samples']} samples, worst gap {result['gap_ms']:.3f} ms)")
        time.sleep(args.settle_s)
    return trials, analyses


def _median_of(analyses: list[dict], key: str) -> float | None:
    values = [a[key] for a in analyses if a[key] is not None]
    return statistics.median(values) if values else None


def report_sweep(rows: list[tuple[float, list[dict]]]) -> None:
    """Compare amplitudes. This is the check that says whether to believe the correction.

    A dead time is a property of the servo, not of how far it was asked to go, so any
    quantity that is really a dead time must come out the same at every amplitude. Anything
    that drifts across the sweep is measuring the response instead -- and since the whole
    point of the fitted onset is to undo an amplitude-dependent bias, this is precisely
    where an imperfect correction shows itself.
    """
    print()
    print("Amplitude sweep — a dead time must not depend on how far the joint was asked to go")
    print("  amplitude    order acted on    first count      motion onset")
    for amplitude, analyses in rows:
        cells = []
        for key in ("current_ms", "motion_ms", "onset_ms"):
            value = _median_of(analyses, key)
            cells.append("     n/a    " if value is None else f"  {value:6.2f} ms  ")
        print(f"  {amplitude:6.1f} deg  {cells[0]}    {cells[1]}   {cells[2]}")

    print()
    for key, label in (("current_ms", "order acted on"), ("onset_ms", "motion onset")):
        values = [_median_of(a, key) for _, a in rows]
        values = [v for v in values if v is not None]
        if len(values) < 2:
            continue
        drift = max(values) - min(values)
        verdict = "flat — behaves like a true dead time" if drift < 0.5 else (
            f"drifts {drift:.1f} ms — still carries some of the response, treat as an upper bound")
        print(f"  {label:<18} {verdict}")


def _stats(values: list[float]) -> str:
    if not values:
        return "no detections"
    values = sorted(values)
    n = len(values)
    median = statistics.median(values)
    spread = f"sd {statistics.stdev(values):.2f}" if n > 1 else "single trial"
    return (f"{median:6.2f} ms   [{values[0]:.2f}, {values[-1]:.2f}]   {spread}   n={n}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default=os.environ.get("MICROBAN_PORT", DEFAULT_PORT))
    parser.add_argument("--baud", type=int, default=int(os.environ.get("MICROBAN_BAUD", DEFAULT_BAUDRATE)))
    parser.add_argument("--id", type=int, default=HEAD_ID, help="motor id to benchmark")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--amplitude-deg", type=float, default=DEFAULT_AMPLITUDE_DEG,
                        help="the head steps between -A and +A")
    parser.add_argument("--sweep", metavar="A1,A2,...",
                        help="repeat the run at several amplitudes and check which delays "
                             "stay put — a real dead time does not depend on step size")
    parser.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S,
                        help="hold time before each step, so every trial starts from rest")
    parser.add_argument("--capture-ms", type=float, default=DEFAULT_CAPTURE_MS)
    parser.add_argument("--with-current", action="store_true",
                        help="read the current+position block instead of position alone, to "
                             "time the firmware acting on the order as well as the motion")
    parser.add_argument("--json", metavar="PATH", help="dump every raw trace for offline analysis")
    parser.add_argument("--keep-torque", action="store_true",
                        help="leave torque enabled on exit (default is to disable it)")
    args = parser.parse_args()

    probe = HeadProbe(args.port, args.baud, args.id, args.with_current)

    try:
        probe.io.ping(args.id)
    except Exception as exc:
        raise SystemExit(f"No answer from motor {args.id} on {args.port} at {args.baud} baud: {exc}")

    settings = probe.settings()
    print(f"Motor          id {args.id} on {args.port} at {args.baud} baud, firmware {settings['firmware']}")
    print(f"Servo          P={settings['position_p_gain']} D={settings['position_d_gain']}  "
          f"profile vel={settings['profile_velocity']} acc={settings['profile_acceleration']}  "
          f"current limit={settings['current_limit']}")
    print(f"Return delay   {settings['return_delay_time_us']:.0f} us"
          + ("   <- lands straight in the round trip; consider writing it to 0"
             if settings["return_delay_time_us"] > 100 else ""))
    # This decides what t=0 actually means, so it is spelled out rather than left implicit.
    if settings["status_return_level"] == 2:
        print("Status return  2 — the servo acknowledges writes, so the write call returns only")
        print("               once it has the order. t=0 is therefore 'order received', which is")
        print("               exactly the reference wanted, at the cost of an ACK in the write.")
    else:
        print(f"Status return  {settings['status_return_level']} — writes are not acknowledged, so t=0 is when the")
        print("               packet finished leaving the master, not when the servo took it.")
    if settings["profile_velocity"] or settings["profile_acceleration"]:
        print("               WARNING: a motion profile is set, so the servo ramps the goal")
        print("                        itself and what follows is the profile, not the delay")
    print(f"Channel        {'position + current' if args.with_current else 'position only'}")
    print()

    trials: list[dict] = []
    analyses: list[dict] = []
    sweep: list[tuple[float, list[dict]]] = []
    amplitudes = (
        [float(v) for v in args.sweep.split(",")] if args.sweep else [args.amplitude_deg]
    )

    try:
        probe.torque(True)

        down_ticks = deg_to_ticks(-amplitudes[0])
        probe.write_goal_ticks(down_ticks)
        time.sleep(max(args.settle_s, 1.0))

        calibration = calibrate_round_trip(probe, CALIBRATION_READS)
        print(f"Round trip     {calibration['median_ms']:.3f} ms median "
              f"[{calibration['p05_ms']:.3f}, {calibration['p95_ms']:.3f}], max {calibration['max_ms']:.3f}"
              f"  =>  {calibration['rate_hz']:.0f} Hz polling")
        print(f"Encoder rest   {calibration['rest_spread_counts']} count spread over "
              f"{calibration['n']} reads (threshold is {MOTION_THRESHOLD_COUNTS})")

        # Null trial: re-command the goal the servo is already holding. Nothing should trip,
        # in either direction. If something does, the threshold is inside the noise and
        # every delay below is measuring the noise instead of the servo.
        null = analyse(run_trial(probe, down_ticks, args.capture_ms), null=True)
        null_ok = null["motion_ms"] is None
        print(f"Null trial     {'clean — no false detection' if null_ok else 'DETECTED MOTION: threshold is in the noise, results below are unsafe'}")
        print()

        for amplitude in amplitudes:
            print(f"Stepping +/-{amplitude:.0f} deg, {args.trials} trials")
            campaign_trials, campaign_analyses = run_campaign(probe, args, amplitude, verbose=True)
            sweep.append((amplitude, campaign_analyses))
            trials.extend(campaign_trials)
            # Only the last amplitude feeds the headline; the sweep table carries the rest.
            analyses = campaign_analyses

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            if not args.keep_torque:
                probe.torque(False)
                print("\nTorque disabled on the head")
        except Exception as exc:
            print(f"\nWarning: could not disable torque: {exc}")

    if not analyses:
        return

    motion = [a["motion_ms"] for a in analyses if a["motion_ms"] is not None]
    onset = [a["onset_ms"] for a in analyses if a["onset_ms"] is not None]
    current = [a["current_ms"] for a in analyses if a["current_ms"] is not None]
    bracket = [a["motion_bracket_ms"] for a in analyses if a["motion_bracket_ms"] is not None]
    writes = [a["write_ms"] for a in analyses]

    print()
    print("Delay from the goal-position write leaving the bus")
    print("                        median      range              spread")
    if current:
        print(f"  order acted on     {_stats(current)}")
    print(f"  first count moved  {_stats(motion)}")
    if onset:
        print(f"  motion onset       {_stats(onset)}")
        print("                     (first count, corrected back for the 0.088 deg it takes to reach it)")
        # Split by direction: the two are mechanically different (gravity, backlash, and
        # whichever way the joint was last loaded), so a gap here is real rather than noise.
        for way in ("up", "down"):
            per = [a["onset_ms"] for a in analyses if a["onset_ms"] is not None and a["direction"] == way]
            if per:
                print(f"    {way:<16} {_stats(per)}")
    print()
    if bracket:
        print(f"  read bracket       +/- {statistics.median(bracket) * 1e3:.0f} us "
              f"(the value is latched somewhere inside the round trip)")
    print(f"  write cost         {statistics.median(writes) * 1e3:.0f} us "
          f"(t=0 is the end of it, so this is not in the numbers above)")
    print()
    if len(sweep) > 1:
        report_sweep(sweep)

    print()
    headline = onset if onset else motion
    if headline:
        print(f"DELAY          {statistics.median(headline):.1f} ms from the order reaching the "
              f"servo to the head moving")
        if onset and motion:
            print(f"               ({statistics.median(motion):.1f} ms to a reading that changes, "
                  f"{statistics.median(motion) - statistics.median(onset):.1f} ms of which is "
                  f"crossing the first count)")
        if current:
            print(f"               of which {statistics.median(current):.1f} ms passes before the "
                  f"servo even starts driving")
    print()
    print("Note           this is the servo's own dead time: bus turnaround, firmware, and")
    print("               mechanical response. It does not include the control loop's")
    print("               sampling, which in a 50 Hz session log adds up to another tick.")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "settings": settings,
                    "calibration": calibration,
                    "args": vars(args),
                    "trials": trials,
                    "analyses": analyses,
                },
                handle,
            )
        print(f"\nRaw traces written to {args.json}")


if __name__ == "__main__":
    main()
