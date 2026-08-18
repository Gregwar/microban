# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Measure the IMU's delay from a recording made by src/imu_delay_record.py.

    PYTHONPATH="src:$PYTHONPATH" python src/debug/imu_delay_analyze.py logs/..._imudelay.json --plot

(placo does not import from the uv venv — see the note at the bottom of this docstring.)

THE IDEA. With a foot flat on the floor, the leg encoders already say what the trunk's
pitch is: forward kinematics through the ankle, knee and hip, one bus read and no filter.
The IMU says the same thing, later. Rocking the robot by hand drives both traces, and the
shift between them is the number wanted.

TWO DELAYS, MEASURED SEPARATELY, because they are not the same number and code uses both:

    gyro         raw angular rate, compared against the derivative of the FK pitch. This
                 is the sensor and the I2C read and nothing else, and it is what the
                 policies' `gyro` observation carries.
    attitude     the fused quaternion (`body_quat`, and the projected gravity derived
                 from it), compared against the FK pitch directly. It adds the Madgwick
                 filter on top of the gyro path, so it is always the slower of the two.

Each is quoted twice: as the sensor's own delay (sample timestamp to sample timestamp),
and as what a control tick actually sees, which adds the age of the freshest sample
sitting in the reader thread.

HOW IT IS MEASURED. Cross-correlation: slide the FK trace against the IMU trace and take
the peak, refined between grid points by a parabola. That is one number for the whole
recording, and on its own it is not enough — it assumes the lag is a pure delay, and
silently returns a compromise if it is not. So the same estimate is repeated:

    per octave band   both traces are band-limited by a zero-phase spectral mask and
                      correlated again, band by band. A dead time delays every frequency
                      alike; a low-pass delays the slow motion more and shrinks the fast.
                      This is what tells the two apart, and it is why the recording is
                      worth making with both slow and brisk rocking in it.
    per window        the recording is cut into slices and each is estimated on its own.
                      A delay that moves from slice to slice is a hand rocking
                      differently, not a property of the IMU.

WHAT MAKES A RUN INVALID. The reference is only as good as the flat-foot assumption:

    gain far from 1     the FK pitch swings less than the IMU pitch — the feet were
                        rolling on heel and toe, so part of the real motion never reached
                        the encoders. The delay is then biased and the run should be
                        redone.
    feet disagree       left-anchored and right-anchored FK should give the same pitch. A
                        gap means a foot lifted or twisted.
    band not excited    a band holding little of the motion, or correlating badly, is
                        reported and then ignored — its delay is fitted noise.

PLACO. `import placo` fails inside the uv venv (eigenpy/numpy ABI clash), so this script
is run with the system python that has a working placo build, hence the PYTHONPATH above
(`PYTHONPATH=src` on its own would *replace* the entry that provides placo).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import placo

from constants import MOTOR_TO_ID, NEUTRAL_POSE
from odometry import gyro_to_trunk, quat_to_matrix

DEFAULT_MODEL_PATH = "src/model/mjcf/robot.xml"

# Everything is compared on one uniform grid, finer than either sensor: the delay is read
# off the shape of the whole trace, not off individual samples, so resolution here costs
# nothing and keeps the sub-sample peak fit honest.
DEFAULT_GRID_HZ = 1000.0

# Lags searched. Negative is allowed on purpose: the IMU cannot lead the encoders, so a
# peak at a negative lag is a sign the run is broken, and clamping the search at zero
# would hide it.
DEFAULT_LAG_RANGE_MS = (-30.0, 250.0)

# Slow drift is removed with a centred moving average — the IMU's attitude wanders over
# tens of seconds and forward kinematics does not, and a shared trend is something the
# correlation would happily fit. Centred, so it is zero-phase and cannot itself shift the
# answer; and applied identically to both traces either way.
DEFAULT_HIGHPASS_S = 3.0

# The FK pitch has to be differentiated to compare against the gyro, which amplifies the
# encoder's 1.5 mrad quantisation. Both traces get the same centred smoother afterwards:
# symmetric, so no phase, so no effect on the delay — only on the noise.
DEFAULT_RATE_SMOOTH_MS = 15.0

DEFAULT_WINDOWS = 6
DEFAULT_FMAX_HZ = 8.0

# Octave bands the delay is measured in separately, to tell a dead time from a filter.
BANDS = ((0.3, 0.6), (0.6, 1.2), (1.2, 2.4), (2.4, 4.8), (4.8, 9.6))

# A band has to hold this share of the motion, and correlate this well, before its delay
# means anything. Below either floor the band was not really excited: what is left there is
# the taper's residual skirt from a strong neighbouring band, which correlates beautifully
# and reports that neighbour's delay scaled by the ratio of the two frequencies. Five
# percent is what it took to reject that on a 12 s recording; a longer one makes the skirt
# narrower still, which is the other reason to record 20-30 s rather than 10.
BAND_POWER_FLOOR = 0.05
BAND_CORR_FLOOR = 0.9


# ----------------------------------------------------------------------------------
# Geometry


def pitch_of(R: np.ndarray) -> float:
    """Pitch of a rotation matrix (ZYX), in radians.

    Yaw-invariant, which is what makes the comparison possible at all: the IMU's heading
    is a gyro integration from an arbitrary start, and the FK frame's heading is whichever
    way the foot happens to point. Neither is known, and neither is needed.
    """
    return float(np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0])))


def roll_of(R: np.ndarray) -> float:
    return float(np.arctan2(R[2, 1], R[2, 2]))


def _column(values: list) -> np.ndarray:
    """A log channel as floats, with dropped reads (nulls) interpolated across.

    src/imu_delay_record.py never writes a null — a failed read costs its whole tick — but
    a session log recorded by the control loop does, and those are worth being able to
    analyse too.
    """
    x = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    bad = np.isnan(x)
    if bad.all():
        return np.zeros_like(x)
    if bad.any():
        x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
    return x


def fk_series(log: dict, model_path: str) -> dict:
    """Trunk pitch and roll from the encoders, once per motor sample, per foot.

    The trunk pose is taken *relative to the foot frame* rather than by pinning the foot
    to the world: a foot flat on the floor is at identity up to a heading, and pitch does
    not depend on heading, so the relative transform already carries the answer without
    having to assume where the robot is standing.
    """
    robot = placo.RobotWrapper(model_path, placo.Flags.mjcf)
    position = log["position"]
    names = [name for name in MOTOR_TO_ID if name in position]
    # Joints that were not recorded cannot move the trunk relative to the feet (arms,
    # head), so they only need to be somewhere legal.
    for name in MOTOR_TO_ID:
        if name not in position:
            robot.set_joint(name, NEUTRAL_POSE.get(name, 0.0))

    n = len(log["time"])
    out = {key: np.empty(n) for key in ("pitch_left", "pitch_right", "roll_left", "roll_right")}
    columns = {name: _column(position[name]) for name in names}

    for i in range(n):
        for name in names:
            robot.set_joint(name, float(columns[name][i]))
        robot.update_kinematics()
        R_trunk = robot.get_T_world_frame("trunk")[:3, :3]
        for side in ("left", "right"):
            R_foot = robot.get_T_world_frame(f"{side}_foot")[:3, :3]
            R = R_foot.T @ R_trunk
            out[f"pitch_{side}"][i] = pitch_of(R)
            out[f"roll_{side}"][i] = roll_of(R)
    return out


def imu_series(log: dict) -> dict:
    """Trunk pitch, roll and pitch rate from the IMU, at the sample times it reports.

    Ticks that read the reader thread twice between two of its samples are duplicates
    carrying one measurement, and keeping them would weight that measurement twice; only
    the first occurrence of each sample timestamp survives.
    """
    t = np.asarray(log.get("imu_time") or log["time"], dtype=float)
    keep = np.ones(t.size, dtype=bool)
    keep[1:] = np.diff(t) > 0.0

    quat = np.stack([_column(log["body_quat"][axis]) for axis in "wxyz"], axis=1)
    gyro = np.stack([_column(log["gyro"][axis]) for axis in "xyz"], axis=1)

    pitch = np.array([pitch_of(quat_to_matrix(q)) for q in quat[keep]])
    roll = np.array([roll_of(quat_to_matrix(q)) for q in quat[keep]])
    # Trunk-frame angular velocity. Component 1 is the pitch rate; the raw channel is in
    # the sensor frame, where it is not (see odometry.gyro_to_trunk).
    rate = np.array([gyro_to_trunk(g)[1] for g in gyro[keep]])
    return {"t": t[keep], "pitch": pitch, "roll": roll, "rate": rate}


# ----------------------------------------------------------------------------------
# Signal conditioning


def detrend(x: np.ndarray, dt: float, window_s: float) -> np.ndarray:
    """Remove the mean and anything slower than `window_s`, without phase distortion."""
    x = x - x.mean()
    width = int(round(window_s / dt))
    width += 1 - width % 2  # odd, so the average is centred on a sample
    if width <= 1 or width >= x.size:
        return x
    pad = width // 2
    baseline = np.convolve(np.pad(x, pad, mode="edge"), np.ones(width) / width, mode="valid")
    return x - baseline


def smooth(x: np.ndarray, dt: float, window_s: float) -> np.ndarray:
    """Centred moving average — zero phase, so it cannot move a delay."""
    width = int(round(window_s / dt))
    width += 1 - width % 2
    if width <= 1 or width >= x.size:
        return x
    pad = width // 2
    return np.convolve(np.pad(x, pad, mode="edge"), np.ones(width) / width, mode="valid")


# ----------------------------------------------------------------------------------
# Estimators


def correlate_lag(late: np.ndarray, early: np.ndarray, dt: float,
                  lag_range_ms: tuple[float, float]) -> dict:
    """Delay of `late` behind `early`, by peak normalised cross-correlation.

    Positive means `late` lags. The peak is refined by fitting a parabola to its two
    neighbours, which is worth doing: at a 1 kHz grid the raw peak is quantised to 1 ms,
    and the underlying correlation is smooth on that scale.

    `gain` is the regression slope at the peak, and it is the honesty check on the whole
    measurement — a filtered signal is not just late, it is also smaller, and a gain far
    from 1 means something other than a delay is going on (rolling feet, most likely).
    """
    lo = int(round(lag_range_ms[0] * 1e-3 / dt))
    hi = int(round(lag_range_ms[1] * 1e-3 / dt))
    n = late.size

    lags, corrs, gains = [], [], []
    for shift in range(lo, hi + 1):
        if shift >= 0:
            a, b = late[shift:], early[: n - shift]
        else:
            a, b = late[: n + shift], early[-shift:]
        if a.size < n // 4:
            continue
        aa = float(a @ a)
        bb = float(b @ b)
        ab = float(a @ b)
        if aa <= 0.0 or bb <= 0.0:
            continue
        lags.append(shift * dt)
        corrs.append(ab / np.sqrt(aa * bb))
        gains.append(ab / bb)

    if not lags:
        return {"lag_s": float("nan"), "corr": 0.0, "gain": float("nan"),
                "lags": np.empty(0), "corrs": np.empty(0)}

    lags = np.array(lags)
    corrs = np.array(corrs)
    gains = np.array(gains)
    k = int(np.argmax(corrs))

    lag = lags[k]
    if 0 < k < corrs.size - 1:
        # Parabola through the three points around the peak; the correction is bounded by
        # half a grid step, so a bad fit cannot run away with the answer.
        denom = corrs[k - 1] - 2 * corrs[k] + corrs[k + 1]
        if denom < 0.0:
            lag += 0.5 * dt * (corrs[k - 1] - corrs[k + 1]) / denom

    return {"lag_s": float(lag), "corr": float(corrs[k]), "gain": float(gains[k]),
            "lags": lags, "corrs": corrs}


def bandpass(x: np.ndarray, dt: float, low: float, high: float) -> np.ndarray:
    """Keep only [low, high) Hz, with no phase distortion whatsoever.

    A real, symmetric mask applied to the spectrum of the *whole* record: the filter is
    zero-phase by construction, so it cannot contribute any delay of its own, and the
    frequency resolution is that of the full recording rather than of a short segment —
    which is what keeps a strong tone from bleeding into the band next door and dragging
    its estimate with it.
    """
    spectrum = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, dt)
    spectrum[(f < low) | (f >= high)] = 0.0
    return np.fft.irfft(spectrum, n=x.size)


def band_lags(late: np.ndarray, early: np.ndarray, dt: float,
              lag_range_ms: tuple[float, float],
              power_ref: np.ndarray | None = None) -> list[dict]:
    """The delay measured separately in each octave band.

    This is the test that says what kind of lag it is. A dead time delays every frequency
    by the same number of milliseconds; a low-pass filter delays the slow motion more than
    the fast. One correlation peak over the whole recording cannot tell those apart — it
    just returns whichever compromise fits the frequencies the operator happened to use.

    `share` is how much of the body's motion lives in the band, so a band nobody excited
    can be recognised rather than believed. It is deliberately measured on `power_ref` —
    the trunk pitch — for every channel: the rate traces are differentiated, which lifts
    the encoder's quantisation noise into the top band and would otherwise make that band
    look like the busiest part of the recording when it holds no real motion at all.
    """
    # Taper first. A recording is an arbitrary chunk of a longer motion, so its ends are
    # discontinuities, and a rectangular cut smears a strong tone across the entire
    # spectrum. The neighbouring band then fills with skirt that is coherent, carries real
    # power, and reports the strong tone's phase divided by the wrong frequency — a delay
    # wrong by the ratio of the two frequencies. The same taper goes on both traces, so it
    # cannot shift the lag between them.
    taper = np.hanning(late.size)
    late, early = late * taper, early * taper
    reference = early if power_ref is None else power_ref * taper

    power = np.abs(np.fft.rfft(reference)) ** 2
    f = np.fft.rfftfreq(reference.size, dt)
    total = float(np.sum(power[f > 0.0]))

    out = []
    for low, high in BANDS:
        inside = (f >= low) & (f < high)
        share = float(np.sum(power[inside]) / total) if total > 0.0 else 0.0
        result = correlate_lag(bandpass(late, dt, low, high),
                               bandpass(early, dt, low, high), dt, lag_range_ms)
        out.append({"low": low, "high": high, "power_share": share, **result})
    return out


def spectrum_of(x: np.ndarray, dt: float, fmax_hz: float) -> dict:
    """Where the motion actually was, for the report and the plot."""
    power = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, dt)
    keep = (f > 0.0) & (f <= fmax_hz)
    f, power = f[keep], power[keep]
    if power.size == 0 or power.max() <= 0.0:
        return {"f": f, "power": power, "dominant_hz": float("nan")}
    return {"f": f, "power": power / power.max(),
            "dominant_hz": float(f[int(np.argmax(power))])}


def windowed_lags(late: np.ndarray, early: np.ndarray, dt: float,
                  lag_range_ms: tuple[float, float], windows: int) -> list[dict]:
    """The correlation estimate repeated on consecutive slices of the recording."""
    size = late.size // max(1, windows)
    out = []
    if size < int(0.5 / dt):  # a slice shorter than half a second says nothing
        return out
    for k in range(windows):
        piece = slice(k * size, (k + 1) * size)
        result = correlate_lag(late[piece], early[piece], dt, lag_range_ms)
        if np.isfinite(result["lag_s"]):
            out.append(result)
    return out


# ----------------------------------------------------------------------------------
# Reporting


def _fmt_ms(value: float) -> str:
    return "   n/a " if not np.isfinite(value) else f"{value * 1e3:6.1f}"


def usable_bands(bands: list[dict]) -> list[dict]:
    """The bands that were excited hard enough, and cleanly enough, to be believed."""
    return [b for b in bands
            if b["power_share"] >= BAND_POWER_FLOOR
            and b["corr"] >= BAND_CORR_FLOOR
            and np.isfinite(b["lag_s"])]


def report_channel(name: str, description: str, result: dict, bands: list[dict],
                   per_window: list[dict], age_ms: float) -> None:
    print()
    print(f"=== {name} — {description} ===")
    print(f"  whole recording    {_fmt_ms(result['lag_s'])} ms      "
          f"correlation {result['corr']:.3f}   gain {result['gain']:.3f}")

    if per_window:
        lags = np.array([r["lag_s"] for r in per_window])
        corrs = np.array([r["corr"] for r in per_window])
        print(f"  across {len(per_window)} windows    "
              f"median {_fmt_ms(float(np.median(lags)))} ms   "
              f"range [{lags.min() * 1e3:.1f}, {lags.max() * 1e3:.1f}] ms   "
              f"worst correlation {corrs.min():.3f}")

    if bands:
        print()
        print("        band         delay    gain   correlation   share of motion")
        for band in bands:
            flags = ""
            if band["power_share"] < BAND_POWER_FLOOR:
                flags = "   (barely excited — ignore)"
            elif band["corr"] < BAND_CORR_FLOOR:
                flags = "   (poor fit — ignore)"
            print(f"    {band['low']:4.1f} - {band['high']:4.1f} Hz   "
                  f"{_fmt_ms(band['lag_s'])} ms  {band['gain']:6.3f}       "
                  f"{band['corr']:5.3f}       {100 * band['power_share']:5.1f} %{flags}")

    if np.isfinite(result["lag_s"]) and np.isfinite(age_ms):
        print()
        print(f"  as a control tick sees it: {result['lag_s'] * 1e3 + age_ms:.1f} ms "
              f"({result['lag_s'] * 1e3:.1f} ms of sensor delay plus {age_ms:.1f} ms of "
              f"median sample age in the reader thread)")


def sanity(fk: dict, imu_grid: dict, result: dict) -> None:
    print()
    print("=== Sanity ===")
    spread = np.degrees(np.std(fk["pitch_left"] - fk["pitch_right"]))
    verdict = "feet agree" if spread < 1.0 else "FEET DISAGREE — one lifted or twisted; the FK reference is unreliable"
    print(f"  left vs right FK pitch   {spread:.2f} deg rms      {verdict}")

    swing_fk = np.degrees(np.ptp(fk["pitch"]))
    swing_imu = np.degrees(np.ptp(imu_grid["pitch"]))
    print(f"  pitch travelled          FK {swing_fk:.1f} deg, IMU {swing_imu:.1f} deg peak-to-peak")
    if swing_fk < 5.0:
        print("                           thin — rock it further, the estimate scales with this")

    # The gyro channel equates the trunk's y angular rate with d(pitch)/dt, which only holds
    # while the robot stays close to upright in roll. Fore/aft rocking keeps it there; a run
    # with the robot leaning sideways does not, and the gyro number would quietly absorb the
    # roll and yaw rates through the Euler coupling.
    roll = np.degrees(np.abs(0.5 * (fk["roll_left"] + fk["roll_right"])))
    print(f"  roll during the run      {roll.max():.1f} deg worst, {roll.mean():.1f} deg mean")
    if roll.max() > 15.0:
        print("                           large — the gyro comparison assumes the motion is pitch")
        print("                           only; rock it fore/aft, not sideways")

    gain = result["gain"]
    if np.isfinite(gain) and abs(gain - 1.0) > 0.25:
        print(f"  gain {gain:.2f}                 the two traces are not the same size. Most likely the")
        print("                           feet rolled on heel and toe instead of staying flat, which")
        print("                           the encoders cannot see. Treat the delay as a lower bound.")
    if result["corr"] < 0.9:
        print(f"  correlation {result['corr']:.2f}          low — not enough clean motion to trust the peak.")
    if np.isfinite(result["lag_s"]) and result["lag_s"] < 0.0:
        print("  negative delay           the IMU cannot lead the encoders. Something is wrong with")
        print("                           the timestamps or the flat-foot assumption.")


def plot(grid, fk_pitch, imu_pitch, fk_rate, imu_rate, attitude, rate_result,
         attitude_bands, rate_bands, spectrum, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 11))
    fig.suptitle(f"IMU delay — {path.name}")

    ax = axes[0]
    ax.plot(grid, np.degrees(fk_pitch), label="FK pitch (encoders)", lw=1.0)
    ax.plot(grid, np.degrees(imu_pitch), label="IMU pitch (body_quat)", lw=1.0)
    shift = int(round(attitude["lag_s"] * (len(grid) - 1) / (grid[-1] - grid[0])))
    if shift > 0:
        ax.plot(grid[:-shift], np.degrees(imu_pitch[shift:]), "--", lw=0.9,
                label=f"IMU advanced by {attitude['lag_s'] * 1e3:.1f} ms")
    ax.set_ylabel("pitch [deg]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(grid, fk_rate, label="d(FK pitch)/dt", lw=1.0)
    ax.plot(grid, imu_rate, label="gyro (trunk pitch rate)", lw=1.0)
    ax.set_ylabel("pitch rate [rad/s]")
    ax.set_xlabel("time [s]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for result, label in ((attitude, "attitude"), (rate_result, "gyro")):
        if result["lags"].size:
            ax.plot(result["lags"] * 1e3, result["corrs"], lw=1.0, label=label)
            ax.axvline(result["lag_s"] * 1e3, ls=":", lw=0.8)
    ax.set_xlabel("lag [ms]")
    ax.set_ylabel("correlation")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[3]
    twin = ax.twinx()
    twin.fill_between(spectrum["f"], spectrum["power"], color="0.85", label="motion spectrum")
    twin.set_ylabel("encoder power (normalised)")
    twin.set_ylim(0, 1.05)
    for bands, colour, label in ((attitude_bands, "C0", "attitude"), (rate_bands, "C1", "gyro")):
        for first, band in enumerate(usable_bands(bands)):
            ax.hlines(band["lag_s"] * 1e3, band["low"], band["high"], color=colour, lw=2.5,
                      label=label if first == 0 else None)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("delay [ms]")
    ax.set_xlim(0, spectrum["f"][-1] if spectrum["f"].size else 1.0)
    ax.set_zorder(twin.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", type=Path, help="recording from src/imu_delay_record.py")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--grid-hz", type=float, default=DEFAULT_GRID_HZ)
    parser.add_argument("--highpass-s", type=float, default=DEFAULT_HIGHPASS_S,
                        help="drift slower than this is removed from both traces")
    parser.add_argument("--rate-smooth-ms", type=float, default=DEFAULT_RATE_SMOOTH_MS,
                        help="centred smoothing applied to both rate traces")
    parser.add_argument("--windows", type=int, default=DEFAULT_WINDOWS,
                        help="slices the correlation is repeated on, to show its spread")
    parser.add_argument("--fmax", type=float, default=DEFAULT_FMAX_HZ)
    parser.add_argument("--lag-range-ms", type=float, nargs=2, default=DEFAULT_LAG_RANGE_MS)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    log = json.loads(args.log.read_text(encoding="utf-8"))
    metadata = log.get("metadata", {})
    if "imu_time" not in log:
        print("Note: this log has no imu_time channel, so the IMU samples are dated by the")
        print("      tick that read them. Everything below then includes the reader thread's")
        print("      sample age instead of separating it out.")

    t_motor = np.asarray(log["time"], dtype=float)
    imu = imu_series(log)
    fk = fk_series(log, args.model)

    dt = 1.0 / args.grid_hz
    t0 = max(t_motor[0], imu["t"][0])
    t1 = min(t_motor[-1], imu["t"][-1])
    grid = t0 + dt * np.arange(int((t1 - t0) / dt) + 1)

    fk["pitch"] = 0.5 * (fk["pitch_left"] + fk["pitch_right"])
    fk_pitch = np.interp(grid, t_motor, fk["pitch"])
    imu_pitch = np.interp(grid, imu["t"], imu["pitch"])
    imu_rate = np.interp(grid, imu["t"], imu["rate"])

    print(f"Log            {args.log}")
    print(f"               {len(t_motor)} motor samples, {imu['t'].size} distinct IMU samples, "
          f"{t1 - t0:.1f} s")
    if metadata:
        print(f"               recorded at {metadata.get('target_hz', '?')} Hz target, "
              f"torque {'off' if metadata.get('torque') is False else 'UNKNOWN'}")

    # The gyro measures a rate, so the encoders have to be differentiated to meet it.
    # np.gradient is a centred difference — zero phase, unlike a one-sided one, which would
    # put half a grid step of fake delay into the comparison.
    fk_rate = np.gradient(fk_pitch, dt)
    smooth_s = args.rate_smooth_ms * 1e-3
    fk_rate_s = smooth(fk_rate, dt, smooth_s)
    imu_rate_s = smooth(imu_rate, dt, smooth_s)

    fk_pitch_d = detrend(fk_pitch, dt, args.highpass_s)
    imu_pitch_d = detrend(imu_pitch, dt, args.highpass_s)
    fk_rate_d = detrend(fk_rate_s, dt, args.highpass_s)
    imu_rate_d = detrend(imu_rate_s, dt, args.highpass_s)

    lag_range = tuple(args.lag_range_ms)
    attitude = correlate_lag(imu_pitch_d, fk_pitch_d, dt, lag_range)
    rate_result = correlate_lag(imu_rate_d, fk_rate_d, dt, lag_range)
    attitude_bands = band_lags(imu_pitch_d, fk_pitch_d, dt, lag_range)
    rate_bands = band_lags(imu_rate_d, fk_rate_d, dt, lag_range, power_ref=fk_pitch_d)
    spectrum = spectrum_of(fk_pitch_d, dt, args.fmax)

    age_ms = float(metadata.get("imu_age_median_ms", float("nan")))
    if not np.isfinite(age_ms) and "imu_time" in log:
        age_ms = float(np.median(t_motor - np.asarray(log["imu_time"], dtype=float)) * 1e3)

    print(f"               rocked mainly at {spectrum['dominant_hz']:.2f} Hz")

    report_channel("GYRO", "raw rate, what the policies' gyro observation carries",
                   rate_result, rate_bands,
                   windowed_lags(imu_rate_d, fk_rate_d, dt, lag_range, args.windows), age_ms)
    report_channel("ATTITUDE", "fused quaternion — body_quat and projected gravity",
                   attitude, attitude_bands,
                   windowed_lags(imu_pitch_d, fk_pitch_d, dt, lag_range, args.windows), age_ms)

    sanity(fk, {"pitch": imu_pitch}, attitude)

    print()
    print("=== Verdict ===")
    print(f"  gyro      {_fmt_ms(rate_result['lag_s'])} ms behind the encoders")
    print(f"  attitude  {_fmt_ms(attitude['lag_s'])} ms behind the encoders")
    # Only worth splitting out when both estimates are sound: the difference of two bad
    # numbers is a worse number, and it would come out negative, which is impossible.
    if min(rate_result["corr"], attitude["corr"]) >= BAND_CORR_FLOOR:
        print(f"  of which  {_fmt_ms(attitude['lag_s'] - rate_result['lag_s'])} ms is the fusion "
              f"filter on top of the gyro path")

    good = usable_bands(attitude_bands)
    lags = [b["lag_s"] for b in good]
    # A pure delay passes every frequency at full size and with the same lag. Either
    # signature breaking means a filter, and the distinction matters beyond wording: a
    # filter's lag does not stay put once the robot is excited outside the band measured
    # here, so the single number stops being usable.
    spread_ms = (max(lags) - min(lags)) * 1e3 if len(good) >= 2 else float("nan")
    droops = len(good) >= 2 and good[-1]["gain"] < good[0]["gain"] - 0.05

    if len(good) < 2:
        print("  only one frequency band carried real motion, so there is no way to tell a dead")
        print("  time from a filter here. Redo the run with both slow and brisk rocking.")
    elif spread_ms >= 5.0 or droops:
        if spread_ms >= 5.0:
            print(f"  the delay varies by {spread_ms:.1f} ms across {len(good)} bands "
                  f"({min(lags) * 1e3:.0f} ms at "
                  f"{good[int(np.argmin(lags))]['low']:.1f}-{good[int(np.argmin(lags))]['high']:.1f} Hz, "
                  f"{max(lags) * 1e3:.0f} ms at "
                  f"{good[int(np.argmax(lags))]['low']:.1f}-{good[int(np.argmax(lags))]['high']:.1f} Hz),")
        if droops:
            print(f"  the amplitude falls with frequency ({good[0]['gain']:.2f} at "
                  f"{good[0]['low']:.1f}-{good[0]['high']:.1f} Hz down to {good[-1]['gain']:.2f} at "
                  f"{good[-1]['low']:.1f}-{good[-1]['high']:.1f} Hz),")
        print("  so this is a low-pass rather than a pure dead time: take the number from the band")
        print("  your controller actually operates in, and expect more lag below it.")
    else:
        print(f"  flat to within {spread_ms:.1f} ms across {len(good)} frequency bands, at full "
              f"amplitude — it behaves")
        print("  like a true dead time, so the single number above is the one to use.")

    if args.plot:
        plot(grid, fk_pitch, imu_pitch, fk_rate_s, imu_rate_s, attitude, rate_result,
             attitude_bands, rate_bands, spectrum, args.log)


if __name__ == "__main__":
    main()
