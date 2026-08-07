# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Estimate the command -> motion dead time from a benchmark move log.

`BenchmarkMove` (src/moves/benchmark.py) steps the head with a square wave whose edges are
instantaneous by construction, so every edge in the log is one dead-time measurement. This
script turns those edges into a number with an error bar, instead of a gap eyeballed off
the plot in plot_log.py.

Reading the answer off the plot overestimates: the joint accelerates rather than stepping,
so back-extrapolating the visible slope lands well after motion actually began. And the
naive fix -- averaging the first post-edge sample over all the edges -- is worse than it
looks, because the head settles on the *same* encoder count before every edge, so the
sub-count rounding error is one shared unknown rather than noise that averages away.

Independent estimators are run over two channels, and they fail in different ways, which is
the point -- the answer is only as good as their agreement:

  position  the joint accelerates from rest, so the onset can be read from the curvature of
            the rise. Encoder rounding is modelled exactly rather than averaged (see
            `_batch_loglik`) and the shared sub-count phase is profiled out. The catch is
            that the onset and the *shape* of the rise trade off strongly across the two or
            three samples that carry any information, so the shape is swept and reported
            rather than assumed, under two response laws (see SHAPE_GRIDS). A single
            confident-looking number here is almost always the assumed shape talking.

  current   the first post-edge current sample is ~600x its own quantum, so rounding is a
            non-issue -- but the reading looks time-averaged by the firmware, which trades
            off against the onset. Broken by regressing on the control loop's own tick
            jitter (18-23 ms), which dithers the sample instant relative to the onset and
            separates the two. Limited by having only ~2 ms of dither and ~13 edges.

The reported band spans the response laws rather than intersecting the estimators: which
law holds is a real systematic, and intersecting a wide interval with a narrow one would
just relabel the narrow one's precision.

WHERE t=0 IS, AND WHY THIS SCRIPT CANNOT FINISH THE JOB. The log stamps `time` at the tick
start, before the read; `position` is what came back from that read; `target_position` is
the command computed after it and sent at the end of the tick. So the fitted onset is NOT
referenced to the command going out -- it is referenced to the tick start, and the whole
read sits in between. Writing R for the read, M for the move computation and W for the write:

    the command reaches the head at   t[i] + R + M + W
    => dead time = (fitted onset - t[i]) - (R + M + W)

R is several ms of sync read over 19 motors, it is not recorded in the log, and it does NOT
cancel. An earlier version of this script argued that it did, on the grounds that the head
is last in MOTOR_TO_ID and so is read at the very end of the burst. Direct measurement says
otherwise: src/debug/bench_head_delay.py polls the head alone at ~4 kHz and puts the servo's
dead time at about 6 ms, against ~14 ms of fitted onset here, on a response whose measured
acceleration is the same in both. Roughly 6-8 ms of what this script would otherwise call
servo delay is in fact control-loop timing between the position sample and the write.

So treat the onset below as an onset relative to the tick start, and subtract R + M + W to
get a dead time. Their sum is what [p] reports as `read` + `moves` + `send` in a live
session; pass it via --send-latency-ms. The default is 0, which reports the raw onset rather
than quietly applying a correction that this log cannot supply.

For the servo's dead time itself, prefer bench_head_delay.py: one motor, no sync read to
sit in front of the write, and a round trip short enough to see the onset directly instead
of inferring it from a fitted response shape.

    uv run --group debug src/debug/bench_delay.py                      # newest log in logs/
    uv run --group debug src/debug/bench_delay.py logs/bench_real.json
    uv run --group debug src/debug/bench_delay.py logs/a.json --plot
    uv run --group debug src/debug/bench_delay.py logs/a.json --joint head --send-latency-ms 2
"""

import sys
import json
import math
import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOG_DIR = "logs"

# XL330 register units. Position is what limits the position estimator, so it is spelled
# out rather than imported: one count is 0.088 deg, and the whole early response is a
# handful of them.
ENCODER_COUNTS = 4096
COUNT_RAD = 2.0 * math.pi / ENCODER_COUNTS
CURRENT_UNIT_A = 0.001

# An edge is a commanded jump of at least this much. Well clear of both the noise floor and
# the 0.5 s lerp that BenchmarkMove uses to enter and leave the move.
EDGE_THRESHOLD_RAD = math.radians(5.0)

# Only edges within this fraction of the modal amplitude are kept. The first edge of a run
# leaves 0 rather than -A, so it is half height and has visibly different dynamics; it is
# dropped rather than pooled with the rest.
AMPLITUDE_TOLERANCE = 0.1

# Onset grid. 0.25 ms is already finer than anything the data can resolve; the range starts
# below zero deliberately, so a fit that wants a physically impossible onset (motion before
# the command) says so instead of piling up against a boundary.
ONSET_GRID_MS = np.arange(-4.0, 24.0, 0.25)

# Two response laws, both swept jointly with the onset rather than assumed. The onset and
# the shape of the response trade off strongly over so few samples, so which law is used is
# a real part of the answer, and running both is how that shows up rather than hiding.
#
#   saturated   the physical one, and the default. A 40 deg error saturates the P
#               controller's PWM, so the joint sees a fixed voltage and its speed rises as
#               1 - exp(-u/tau) against back-EMF. Displacement is then
#               v * [u - tau * (1 - exp(-u/tau))], which is quadratic in u while u << tau
#               and straightens to linear after -- exactly the bend the data shows. Because
#               it covers both regimes it can use samples further out, where the encoder
#               rounding no longer dominates.
#
#   power       plain (t - onset)^p. Only valid while the joint is still accelerating under
#               constant torque (p = 2), so it must be held to the first samples; kept as a
#               cross-check because it assumes much less about the servo.
#
# The grids below are the shape parameter of each: tau in ms, and p dimensionless.
# tau is geometric rather than linear: it needs fine resolution where the fit actually
# lands (tens of ms) and only coarse coverage out at the long-tau end, where the response
# is effectively still quadratic and tau stops being identifiable at all. A linear grid
# either rails at its upper bound or leaves the profile too ragged to read an interval off.
SHAPE_GRIDS = {
    "saturated": np.geomspace(3.0, 400.0, 44),
    "power": np.arange(1.6, 3.41, 0.2),
}
# Windows differ because the models are valid over different spans, per the note above.
DEFAULT_WINDOWS = {"saturated": (1, 4), "power": (1, 2)}
# Slices of the surface printed to show how much of the answer is the assumed shape.
REPORTED_SHAPES = {"saturated": (10.0, 20.0, 40.0), "power": (2.0, 2.5, 3.0)}

# Nuisance parameters, all profiled out. `phase` is the shared sub-count offset of the rest
# position (see `_batch_loglik`). The noise has an absolute floor plus a term proportional
# to the response, because the edges do not merely scale -- peak speed varies ~20% across a
# run and the shape varies with it, so a 200-count sample is nowhere near +/-0.2 counts of
# the model even though its own rounding error is that small.
PHASE_GRID = np.linspace(-0.5, 0.5, 9)
SIGMA_ABS_GRID_COUNTS = np.array([0.25, 0.5, 1.0])
SIGMA_REL_GRID = np.array([0.02, 0.05, 0.10, 0.20])
AMPLITUDE_GRID_SIZE = 17
AMPLITUDE_GRID_HALF_WIDTH = 0.45  # fractional, around the value the last sample implies

BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20260806


class Edge(NamedTuple):
    """One commanded step, with everything the estimators need already signed."""

    index: int            # log row where target_position jumps
    time_s: float         # tick start of that row
    amplitude_rad: float  # signed commanded jump
    rel_times_s: np.ndarray   # sample times after the edge, relative to the tick start
    counts: np.ndarray        # position deviation from the pre-edge rest, in counts, signed toward the step
    currents_a: np.ndarray    # current, signed toward the step
    rest_counts: float        # pre-edge position, in counts


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorised erf (Abramowitz & Stegun 7.1.26), good to 1.5e-7.

    math.erf is scalar and scipy is not in the debug group, and the likelihood below wants
    this over whole arrays.
    """
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return sign * y


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def load_log(path: Path) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def newest_log() -> Path:
    logs = sorted(Path(LOG_DIR).glob("*.json"))
    if not logs:
        raise SystemExit(f"No logs found in {LOG_DIR}/")
    return logs[-1]


def extract_edges(log: dict, joint: str, n_samples: int) -> tuple[list[Edge], list[Edge]]:
    """Find the commanded steps. Returns (full amplitude, rejected)."""
    missing = {"time", "target_position", "position"} - set(log)
    if missing:
        raise SystemExit(f"Log is missing {sorted(missing)} — is it a session log from [l]?")
    if joint not in log["position"]:
        raise SystemExit(f"Joint {joint!r} not in log; have {sorted(log['position'])}")

    time_s = np.asarray(log["time"], dtype=float)
    target = np.asarray([np.nan if v is None else v for v in log["target_position"][joint]], dtype=float)
    position = np.asarray([np.nan if v is None else v for v in log["position"][joint]], dtype=float)
    raw_current = log.get("current", {}).get(joint)
    current = (
        np.asarray([np.nan if v is None else v for v in raw_current], dtype=float)
        if raw_current is not None
        else np.full_like(position, np.nan)
    )

    jumps = [
        i
        for i in range(1, len(target) - n_samples)
        if abs(target[i] - target[i - 1]) > EDGE_THRESHOLD_RAD
    ]
    if not jumps:
        raise SystemExit(f"No commanded steps on {joint!r} — was this recorded with the benchmark move?")

    amplitudes = np.abs([target[i] - target[i - 1] for i in jumps])
    modal = float(np.median(amplitudes))

    kept: list[Edge] = []
    rejected: list[Edge] = []
    for i in jumps:
        step = target[i] - target[i - 1]
        sign = math.copysign(1.0, step)
        window = slice(i, i + n_samples + 1)
        edge = Edge(
            index=i,
            time_s=float(time_s[i]),
            amplitude_rad=float(step),
            rel_times_s=time_s[window] - time_s[i],
            counts=(position[window] - position[i]) / COUNT_RAD * sign,
            currents_a=current[window] * sign,
            rest_counts=float(position[i] / COUNT_RAD),
        )
        target_list = kept if abs(abs(step) - modal) <= AMPLITUDE_TOLERANCE * modal else rejected
        target_list.append(edge)

    return kept, rejected


def _shape(elapsed_ms: np.ndarray, model: str, param: float) -> np.ndarray:
    """Unit-amplitude displacement `param` defines, as a function of time since onset.

    Both laws are normalised only up to a scale, since the per-edge amplitude is profiled
    out anyway; all that matters is the shape.
    """
    if model == "power":
        return elapsed_ms ** param
    # Saturated motor: speed rises as 1 - exp(-u/tau) against back-EMF, so displacement is
    # its integral. Written with expm1 so the u << tau limit stays accurate.
    u = elapsed_ms / param
    return param * param * (u + np.expm1(-u))


def _batch_loglik(
    counts: np.ndarray,
    rel_times_ms: np.ndarray,
    onset_ms: float,
    model: str,
    param: float,
) -> float:
    """Profiled log-likelihood of every edge at one candidate (onset, shape parameter).

    The motor reports whole encoder counts, and the joint sits on the *same* count before
    every edge, so the sub-count offset of that rest position is one unknown shared by the
    whole run rather than per-edge noise. Write it as `phase`: with the true displacement
    x (in counts) the reported deviation is exactly round(x - phase). So the likelihood of
    seeing integer d is the probability that x - phase falls in [d - 0.5, d + 0.5), which
    is what makes this an interval probability rather than a squared residual -- and is
    what keeps a 1-count observation from pretending to be more precise than it is.

    Profiled out here: the response amplitude, separately per edge; the shared phase; and
    both noise scales. Everything is one broadcast over
    (edge, amplitude, phase, sigma_abs, sigma_rel, sample), because the onset and exponent
    grids are swept outside it and a Python loop per cell is far too slow.
    """
    elapsed = np.clip(rel_times_ms - onset_ms, 0.0, None)
    shape = _shape(elapsed, model, param)                              # (E, S)

    # Amplitude is pinned to within a few percent by the last (largest) sample, so the grid
    # is centred on what that sample implies at this onset rather than spanning everything.
    # It has to be rebuilt per onset: that implied value moves as the onset moves.
    last = shape[:, -1]
    hint = np.where(last > 0, counts[:, -1] / np.maximum(last, 1e-30), 1.0)
    hint = np.maximum(hint, 1e-9)
    frac = np.linspace(1.0 - AMPLITUDE_GRID_HALF_WIDTH, 1.0 + AMPLITUDE_GRID_HALF_WIDTH, AMPLITUDE_GRID_SIZE)
    amplitudes = hint[:, None] * frac[None, :]                         # (E, A)

    model = amplitudes[:, :, None, None, None, None] * shape[:, None, None, None, None, :]
    model = model - PHASE_GRID[None, None, :, None, None, None]        # (E, A, P, 1, 1, S)

    sigma = np.sqrt(
        SIGMA_ABS_GRID_COUNTS[None, None, None, :, None, None] ** 2
        + (SIGMA_REL_GRID[None, None, None, None, :, None] * np.abs(model)) ** 2
    )
    observed = counts[:, None, None, None, None, :]

    hi = _norm_cdf((observed + 0.5 - model) / sigma)
    lo = _norm_cdf((observed - 0.5 - model) / sigma)
    cell = np.log(np.clip(hi - lo, 1e-300, None)).sum(axis=-1)         # (E, A, P, Ga, Gr)

    per_edge = cell.max(axis=1)                                        # profile amplitude
    return float(per_edge.sum(axis=0).max())                           # profile phase and sigmas


def fit_onset_from_position(edges: list[Edge], model: str, window: tuple[int, int]) -> dict:
    """Log-likelihood surface over (onset, shape parameter) for the position channel.

    Returns the surface plus two readings of it: the interval with the shape profiled out,
    which is the honest one, and slices at fixed shapes, which show how much of any
    tight-looking answer is really just the shape that was assumed.
    """
    first, last = window
    grid = SHAPE_GRIDS[model]
    rel_times = np.stack([e.rel_times_s[first:last + 1] for e in edges]) * 1e3
    counts = np.stack([e.counts[first:last + 1] for e in edges])

    surface = np.array([
        [_batch_loglik(counts, rel_times, ms, model, param) for param in grid]
        for ms in ONSET_GRID_MS
    ])                                                                 # (onset, shape)

    best = np.unravel_index(surface.argmax(), surface.shape)
    return {
        "model": model,
        "window": window,
        "shape_grid": grid,
        "surface": surface,
        "profiled": _interval(ONSET_GRID_MS, surface.max(axis=1)),
        "slices": {
            param: _interval(ONSET_GRID_MS, surface[:, int(np.argmin(np.abs(grid - param)))])
            for param in REPORTED_SHAPES[model]
        },
        # Best-fitting shape at each onset: this is the degeneracy, made visible.
        "ridge": grid[surface.argmax(axis=1)],
        "best_shape": float(grid[best[1]]),
    }


def _interval(grid_ms: np.ndarray, profile: np.ndarray) -> dict:
    """Point estimate and interval from a profile log-likelihood.

    The interval is the usual 1.92 log-likelihood drop (a 95% interval for one parameter).
    """
    peak = int(np.argmax(profile))
    inside = np.flatnonzero(profile - profile[peak] > -1.92)
    return {
        "onset_ms": float(grid_ms[peak]),
        "lo_ms": float(grid_ms[inside[0]]),
        "hi_ms": float(grid_ms[inside[-1]]),
        "profile": profile,
    }


def fit_onset_from_current(edges: list[Edge]) -> dict | None:
    """Onset from the current channel, using the loop's own tick jitter as dither.

    The first post-edge current sample is large (~0.7 A against a 1 mA quantum), so unlike
    the position channel it is not rounding-limited. What it is limited by is the firmware
    reporting something that behaves like a time average: a single sample cannot separate
    "the command arrived late" from "the averaging window swallowed most of the step".

    The control loop does not tick at exactly 20 ms -- it jitters over ~18-23 ms -- which
    moves the sample instant relative to the onset from edge to edge, at no cost. Modelling
    the reading as the fraction of an averaging window W that lies after the onset,

        I(dt) = I_sat * (dt - onset) / W        while the window straddles the onset

    makes it a straight line in dt whose x-intercept is the onset and whose slope gives W.
    I_sat is read off the following sample, by when the window is clear of the onset.
    """
    dt_ms = np.array([e.rel_times_s[1] * 1e3 for e in edges])
    first = np.array([e.currents_a[1] for e in edges])
    saturated = np.array([e.currents_a[2] for e in edges])
    if not np.isfinite(first).all():
        return None

    def fit(idx: np.ndarray) -> tuple[float, float]:
        slope, intercept = np.polyfit(dt_ms[idx], first[idx], 1)
        return float(-intercept / slope), float(slope)

    onset_ms, slope = fit(np.arange(len(edges)))
    i_sat = float(np.median(saturated))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        idx = rng.integers(0, len(edges), len(edges))
        if len(np.unique(dt_ms[idx])) < 3:
            continue
        try:
            value, _ = fit(idx)
        except (np.linalg.LinAlgError, ValueError):
            continue
        draws.append(value)
    draws = np.array(draws)

    return {
        "onset_ms": onset_ms,
        "lo_ms": float(np.percentile(draws, 2.5)),
        "hi_ms": float(np.percentile(draws, 97.5)),
        "slope_a_per_ms": slope,
        "correlation": float(np.corrcoef(dt_ms, first)[0, 1]),
        "i_sat_a": i_sat,
        "window_ms": i_sat / slope if slope > 0 else float("nan"),
        "dither_ms": float(dt_ms.max() - dt_ms.min()),
    }


def report(
    log: dict,
    path: Path,
    joint: str,
    edges: list[Edge],
    rejected: list[Edge],
    fits: dict[str, dict],
    current_fit: dict | None,
    send_latency_ms: float,
) -> None:
    time_s = np.asarray(log["time"], dtype=float)
    dt_ms = np.diff(time_s) * 1e3

    print(f"Log            {path}")
    meta = log.get("metadata", {})
    if meta:
        print(f"               {meta.get('policy', '?')} policy, {meta.get('ticks', '?')} ticks, "
              f"{meta.get('duration_s', '?')} s")
    print(f"Joint          {joint}")
    print(f"Tick           {dt_ms.mean():.2f} ms mean ({1e3 / dt_ms.mean():.1f} Hz), "
          f"sd {dt_ms.std():.2f}, range {dt_ms.min():.2f}-{dt_ms.max():.2f}")
    print(f"Edges          {len(edges)} at full amplitude "
          f"({math.degrees(abs(edges[0].amplitude_rad)):.0f} deg)"
          + (f", {len(rejected)} rejected as off-amplitude" if rejected else ""))
    rests = sorted({round(e.rest_counts, 2) for e in edges})
    print(f"Rest           {rests} counts — "
          f"{'one shared sub-count phase, profiled out' if len(rests) <= 2 else 'varied'}")
    print()

    units = {"saturated": ("tau", "ms"), "power": ("p", "")}
    for model, fit in fits.items():
        name, unit = units[model]
        first, last = fit["window"]
        print(f"Position channel, {model} response law, samples +{first}..+{last}")
        print(f"  shape                   onset      95% interval")
        for param, sl in fit["slices"].items():
            flag = "   <- before the command: unphysical" if sl["onset_ms"] < 0.0 else ""
            print(f"  {name} fixed at {param:5.1f}{unit:<3}  {sl['onset_ms']:5.1f} ms   "
                  f"[{sl['lo_ms']:5.1f}, {sl['hi_ms']:5.1f}]{flag}")
        prof = fit["profiled"]
        print(f"  {name} profiled out{'':<8}  {prof['onset_ms']:5.1f} ms   "
              f"[{prof['lo_ms']:5.1f}, {prof['hi_ms']:5.1f}]   <- best fit at "
              f"{name}={fit['best_shape']:.1f}{unit}")
        print(f"  degeneracy: best {name} runs {fit['ridge'][0]:.1f} -> {fit['ridge'][-1]:.1f} "
              f"across the onset grid")
        print()

    if current_fit:
        print("Current channel, tick-jitter dither regression")
        print(f"  onset                  {current_fit['onset_ms']:5.1f} ms   "
              f"[{current_fit['lo_ms']:5.1f}, {current_fit['hi_ms']:5.1f}]  (bootstrap over edges)")
        print(f"  slope {current_fit['slope_a_per_ms']:.3f} A/ms, r={current_fit['correlation']:.2f} over "
              f"{current_fit['dither_ms']:.1f} ms of dither")
        print(f"  implies a {current_fit['window_ms']:.0f} ms averaging window at "
              f"I_sat={current_fit['i_sat_a']:.2f} A")
        print()

    # The band spans both response laws rather than intersecting them: which law is right is
    # a genuine systematic, not something to average away, and the two windows make the fits
    # near enough independent. Clipped at zero, since the joint cannot move before the
    # command reaches it. The current channel is a consistency check only -- its interval is
    # far too wide to narrow anything, and intersecting a wide interval with a narrow one
    # would report the narrow one's precision under a second estimator's name.
    lo = max(0.0, min(f["profiled"]["lo_ms"] for f in fits.values()))
    hi = max(f["profiled"]["hi_ms"] for f in fits.values())
    onset = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)

    if current_fit:
        agrees = current_fit["lo_ms"] <= onset <= current_fit["hi_ms"]
        print(f"Cross-check    current channel {'agrees' if agrees else 'DISAGREES'}: "
              f"{current_fit['onset_ms']:.1f} ms [{current_fit['lo_ms']:.1f}, {current_fit['hi_ms']:.1f}] "
              f"vs {onset:.1f} ms from position")
        print()

    print(f"Onset          {onset:.0f} +/- {half:.0f} ms after the edge tick start "
          f"(spans both response laws)")
    if send_latency_ms > 0.0:
        print(f"  send path    -{send_latency_ms:.1f} ms  (sync read + move computation + write)")
        print()
        print(f"DEAD TIME      {onset - send_latency_ms:.0f} +/- {half:.0f} ms, from the goal position")
        print(f"               reaching the motor to the joint starting to move")
        print(f"               ({max(0.0, onset - send_latency_ms - half):.0f} to "
              f"{onset - send_latency_ms + half:.0f} ms)")
    else:
        print()
        print("               This is an onset, not a dead time. The sync read, the move")
        print("               computation and the write all sit between the position sample")
        print("               above and the command reaching the motor, and none of them are")
        print("               in this log. Subtract them with --send-latency-ms ([p] reports")
        print("               them as read + moves + send), or measure the servo directly with")
        print("               bench_head_delay.py, which does not have the problem.")
    print()
    print("Caveats        the readback carries the servo's own sensing lag, so this is an upper")
    print(f"               bound on the true mechanical onset. Sampling at {dt_ms.mean():.1f} ms means the")
    print("               log cannot show motion sooner than one tick whatever the truth is;")
    print("               the estimate leans on the shape of the response to get inside that.")


def plot(edges: list[Edge], fits: dict[str, dict], current_fit: dict | None) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    ax = axes[0]
    for e in edges:
        ax.plot(e.rel_times_s * 1e3, e.counts, "o-", color="tab:blue", alpha=0.35, ms=4, lw=1)
    for model, fit in fits.items():
        prof = fit["profiled"]
        ax.axvline(prof["onset_ms"], lw=2, label=f"{model}: {prof['onset_ms']:.1f} ms")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("time after edge tick start (ms)")
    ax.set_ylabel("displacement (encoder counts)")
    ax.set_title("Step responses, all edges overlaid")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for model, fit in fits.items():
        prof = fit["profiled"]
        ax.plot(ONSET_GRID_MS, prof["profile"] - prof["profile"].max(), lw=2, label=f"{model}, shape profiled")
    ax.axhline(-1.92, color="black", ls="--", lw=1, label="95% (1.92 drop)")
    ax.axvline(0.0, color="tab:grey", lw=1)
    ax.set_ylim(-12, 0.5)
    ax.set_xlabel("onset (ms)")
    ax.set_ylabel("profile log-likelihood")
    ax.set_title("Position channel: profile likelihood")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    fit = fits["saturated"]
    surface = fit["surface"]
    im = ax.pcolormesh(
        ONSET_GRID_MS, fit["shape_grid"], (surface - surface.max()).T, vmin=-12, vmax=0, shading="auto"
    )
    ax.plot(ONSET_GRID_MS, fit["ridge"], "r-", lw=1.5, label="best tau at each onset")
    ax.axvline(0.0, color="white", lw=1)
    ax.set_xlabel("onset (ms)")
    ax.set_ylabel("motor time constant tau (ms)")
    ax.set_title("Onset vs response shape: the degeneracy")
    ax.legend(fontsize=8, loc="lower right")
    fig.colorbar(im, ax=ax, label="log-likelihood drop")

    ax = axes[3]
    if current_fit:
        dt_ms = np.array([e.rel_times_s[1] * 1e3 for e in edges])
        first = np.array([e.currents_a[1] for e in edges])
        ax.plot(dt_ms, first, "o", color="tab:green")
        xs = np.linspace(current_fit["onset_ms"], dt_ms.max() + 0.5, 50)
        ax.plot(xs, current_fit["slope_a_per_ms"] * (xs - current_fit["onset_ms"]), "-", color="tab:red",
                label=f"onset {current_fit['onset_ms']:.1f} ms")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("tick interval after the edge (ms)")
        ax.set_ylabel("first post-edge current (A)")
        ax.set_title("Current channel: tick-jitter dither")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no current channel in log\n(record with [u])", ha="center", va="center")

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", nargs="?", help="benchmark log to analyse (default: newest in logs/)")
    parser.add_argument("--joint", default="head", help="joint the square wave was played on")
    parser.add_argument(
        "--window",
        default=None,
        metavar="FIRST:LAST",
        help="override the post-edge samples used by both position fits "
             f"(defaults {DEFAULT_WINDOWS['saturated']} saturated, {DEFAULT_WINDOWS['power']} power)",
    )
    parser.add_argument(
        "--send-latency-ms",
        type=float,
        default=0.0,
        help="everything between the position sample and the write reaching the motor: the "
             "sync read, the move computation and the write ([p] reports these as read + "
             "moves + send). Subtracted from the onset. Default 0, i.e. report the raw onset",
    )
    parser.add_argument("--plot", action="store_true", help="show the fits")
    args = parser.parse_args()

    path = Path(args.log) if args.log else newest_log()
    log = load_log(path)

    windows = dict(DEFAULT_WINDOWS)
    if args.window:
        first, last = (int(v) for v in args.window.split(":"))
        windows = {model: (first, last) for model in windows}

    n_samples = max(w[1] for w in windows.values()) + 2
    edges, rejected = extract_edges(log, args.joint, n_samples=n_samples)
    if len(edges) < 3:
        raise SystemExit(f"Only {len(edges)} usable edges — need at least 3")

    fits = {model: fit_onset_from_position(edges, model, w) for model, w in windows.items()}
    current_fit = fit_onset_from_current(edges)

    report(log, path, args.joint, edges, rejected, fits, current_fit, args.send_latency_ms)

    if args.plot:
        plot(edges, fits, current_fit)


if __name__ == "__main__":
    main()
