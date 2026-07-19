# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Interactive view of the bam Stribeck coefficient against joint velocity.

The coefficient weights the Stribeck friction terms in bam's motor model:

    excess         = max(0, |dtheta| - dtheta_stiction)
    stribeck_coeff = exp(-(excess / dtheta_stribeck) ** alpha)

It is 1 at standstill and decays to 0 once the joint is moving, which is what makes
friction highest exactly where a joint is trying to break away.

`dtheta_stiction` widens a band around standstill where the coefficient stays flat at 1
before any decay begins: the joint is stuck, not merely slow. Setting it to 0 recovers
bam's current formula exactly. Drag the sliders to see how the three parameters reshape
the curve.

    uv run --group debug src/debug/plot_stribeck.py
    uv run --group debug src/debug/plot_stribeck.py --save stribeck.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Fitted XL330 m6 values, used as the starting point. Read from the installed bam package
# when available so this cannot drift away from what the simulation actually runs.
DEFAULT_DTHETA_STRIBECK = 2.890372094130307
DEFAULT_ALPHA = 8.683259907618984

# Slider ranges, matching the bounds bam allows when fitting these parameters
# (model.py: Parameter(0.2, 0.01, 5.0) and Parameter(1.35, 0.5, 10.0)).
DTHETA_RANGE = (0.01, 5.0)
ALPHA_RANGE = (0.5, 10.0)
# Not yet a fitted bam parameter, so there is no published bound to match. Zero is the
# meaningful default: it reduces the formula to the one bam ships today.
STICTION_RANGE = (0.0, 3.0)
DEFAULT_DTHETA_STICTION = 0.0

# Wide enough that the decay stays on screen across the whole dtheta_stribeck slider
# range: the coefficient only falls off around |dtheta| ~ dtheta_stribeck, which the
# fitted value already puts at 2.9 rad/s.
VELOCITY_MAX = 8.0  # rad/s shown on the x axis


def m6_defaults() -> tuple[float, float]:
    """(dtheta_stribeck, alpha) from the installed bam XL330 m6 fit, with a fallback."""
    try:
        import bam

        path = Path(bam.__file__).parent / "params" / "xl330" / "m6.json"
        params = json.loads(path.read_text(encoding="utf-8"))
        return float(params["dtheta_stribeck"]), float(params["alpha"])
    except Exception:
        return DEFAULT_DTHETA_STRIBECK, DEFAULT_ALPHA


def stribeck_coeff(
    dtheta, dtheta_stribeck: float, alpha: float, dtheta_stiction: float = 0.0
):
    """Stribeck weight: 1 when stopped, decaying to 0 once the joint is moving.

    `dtheta_stiction` holds the coefficient at 1 until the joint exceeds that speed, and
    the decay then runs on the excess only, so raising it shifts the whole knee outwards
    rather than reshaping it. At dtheta_stiction = 0 this is exactly bam/model.py today.
    """
    excess = np.maximum(0.0, np.abs(dtheta) - dtheta_stiction)
    return np.exp(-((excess / dtheta_stribeck) ** alpha))


def build_figure(dtheta_stribeck: float, alpha: float, dtheta_stiction: float = 0.0):
    """Returns (fig, sliders) — the caller must keep `sliders` alive, since a
    garbage-collected matplotlib widget silently stops responding."""
    velocity = np.linspace(-VELOCITY_MAX, VELOCITY_MAX, 2001)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.30)

    # The fitted curve stays visible as a reference while the sliders move the live one.
    ref_dtheta, ref_alpha = m6_defaults()
    ax.plot(
        velocity, stribeck_coeff(velocity, ref_dtheta, ref_alpha),
        color="0.7", linestyle=":", linewidth=1.4,
        label=f"XL330 m6 fit (dtheta={ref_dtheta:.2f}, alpha={ref_alpha:.2f})",
    )
    (curve,) = ax.plot(
        velocity, stribeck_coeff(velocity, dtheta_stribeck, alpha, dtheta_stiction),
        color="tab:blue", linewidth=2.0, label="current",
    )

    # The stiction band, where the coefficient is pinned at 1 and no decay has started.
    stiction_band = ax.axvspan(
        -dtheta_stiction, dtheta_stiction, color="tab:orange", alpha=0.15, linewidth=0
    )

    # The coefficient hits exp(-1) at |dtheta| = dtheta_stiction + dtheta_stribeck whatever
    # alpha is, so the markers show what those two parameters mean independently of it.
    knee_at = dtheta_stiction + dtheta_stribeck
    knee = ax.axvline(knee_at, color="tab:red", linestyle="--", linewidth=1.0)
    knee2 = ax.axvline(-knee_at, color="tab:red", linestyle="--", linewidth=1.0)
    ax.axhline(np.exp(-1.0), color="tab:red", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.text(
        0.02, np.exp(-1.0) + 0.02, "exp(-1) at |dtheta| = dtheta_stiction + dtheta_stribeck",
        transform=ax.get_yaxis_transform(), fontsize=8, color="tab:red",
    )

    ax.set_xlabel("joint velocity dtheta (rad/s)")
    ax.set_ylabel("stribeck_coeff")
    ax.set_title(
        "Stribeck coefficient:  exp(-(max(0, |dtheta| - dtheta_stiction) / dtheta_stribeck) ** alpha)"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-VELOCITY_MAX, VELOCITY_MAX)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    ax_stiction = fig.add_axes([0.12, 0.16, 0.76, 0.03])
    ax_dtheta = fig.add_axes([0.12, 0.10, 0.76, 0.03])
    ax_alpha = fig.add_axes([0.12, 0.04, 0.76, 0.03])
    s_stiction = Slider(
        ax_stiction, "dtheta_stiction", *STICTION_RANGE, valinit=dtheta_stiction, valfmt="%.3f"
    )
    s_dtheta = Slider(
        ax_dtheta, "dtheta_stribeck", *DTHETA_RANGE, valinit=dtheta_stribeck, valfmt="%.3f"
    )
    s_alpha = Slider(ax_alpha, "alpha", *ALPHA_RANGE, valinit=alpha, valfmt="%.3f")

    def update(_value) -> None:
        stiction = s_stiction.val
        curve.set_ydata(stribeck_coeff(velocity, s_dtheta.val, s_alpha.val, stiction))
        knee_at = stiction + s_dtheta.val
        knee.set_xdata([knee_at, knee_at])
        knee2.set_xdata([-knee_at, -knee_at])
        # axvspan gives back a Rectangle drawn in axes-y coordinates, so the height stays
        # 1 and only the x extent moves. A zero-width band simply disappears.
        stiction_band.set_bounds(-stiction, 0.0, 2.0 * stiction, 1.0)
        fig.canvas.draw_idle()

    s_stiction.on_changed(update)
    s_dtheta.on_changed(update)
    s_alpha.on_changed(update)
    return fig, (s_stiction, s_dtheta, s_alpha)


def main() -> None:
    ref_dtheta, ref_alpha = m6_defaults()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dtheta-stribeck", type=float, default=ref_dtheta, help="initial dtheta_stribeck")
    parser.add_argument("--alpha", type=float, default=ref_alpha, help="initial alpha")
    parser.add_argument(
        "--dtheta-stiction", type=float, default=DEFAULT_DTHETA_STICTION,
        help="initial dtheta_stiction (0 reproduces bam's current formula)",
    )
    parser.add_argument("--save", type=Path, metavar="PNG", help="write the figure to a file instead of showing it")
    args = parser.parse_args()

    # `sliders` must stay referenced while the window is open
    fig, sliders = build_figure(args.dtheta_stribeck, args.alpha, args.dtheta_stiction)
    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f"Wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
