# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Score a squat_rl run against the squat reference it was trained to track.

``Mjlab-SquatRef-Microban`` (mjlab_microban/tasks/microban_squatref_env_cfg.py) does not
command a trunk height: it hands the policy a phase and rewards it for matching a
placo-generated squat clip, joint by joint. The clip is one full cycle sampled at the
50 Hz control rate and lives in

    ~/mjlab_microban/src/mjlab_microban/robot/squat_reference.pkl

(regenerate with ``python3 src/mjlab_microban/scripts/make_squat_reference.py``). On the
real robot src/moves/squat_rl.py replays that clip's clock off the wall clock, restarted
at activation:

    phase = 2*pi*SQUAT_FREQUENCY*(t - policy_t0),   command = [cos(phase), sin(phase)]

so a log carrying ``policy_t0`` is enough to know, for every tick, which reference frame
the policy was being asked to be at. This script reconstructs that frame (linear
interpolation between clip rows, the clip loops) and reports how far the robot actually
was from it — the same quantity the training reward is built on.

Three errors are separated, because they have different fixes:

    reference error  read - reference    the thing the task grades: is the squat right?
    command error    goal - reference    the policy asking for the wrong pose
    servo error      read - goal         the servo not reaching the pose it was asked for

and next to the raw error, three shape diagnostics that say *how* a run is wrong:

    lag    the time shift that would minimise the error (robot behind the clock)
    gain   read amplitude / reference amplitude (< 1 = squat too shallow)
    bias   mean offset (posture sitting off the reference)

Angles are reported in degrees, unlike the rest of src/debug (the logs, the clip and the
reward all work in radians, and so does everything inside this script — only the printing
converts). A tenth of a degree is about the resolution the servos read back at, so it is
the scale the numbers here are worth reading to.

Two comparisons, one window. By default each log is scored against the reference clip.
``--sim-vs-real`` instead pairs every real log with its ``_sim`` twin among the logs given
and reports how far the simulation drifted from the robot -- not "did the robot follow the
clip" but "did the simulator reproduce the robot". ``--joints`` narrows either one to the
pitch chain, the legs, or the arms.

Every run is scored over the same window: ``--skip`` seconds after the policy started
(2 by default, while the robot settles onto the reference), then exactly ``--cycles``
squat cycles (8 by default). The whole number of cycles matters -- error varies 2-3x
across the phase, so a window ending mid-squat weights whichever part it stopped in.

``--trunk`` adds the one number the joint table cannot show: how deep the robot actually
squatted, in mm. The clip carries its own ``trunk_z``, and the read joint angles run through
``src/model/mjcf/robot.xml`` give the height the robot really reached — see
:meth:`Run.measured_trunk_z`. It needs MuJoCo, hence ``--group sim``; the placo odometry is
not used (it does not import in this venv, and height needs no dead reckoning).

    uv run --group debug src/debug/squat_ref_error.py                      # newest log
    uv run --group debug src/debug/squat_ref_error.py logs/a.json --plot
    uv run --group debug src/debug/squat_ref_error.py logs/a.json logs/b.json
    uv run --group debug --group sim src/debug/squat_ref_error.py logs/a.json --trunk
    uv run --group debug src/debug/squat_ref_error.py logs/a.json logs/b.json --joints pitch
    uv run --group debug src/debug/squat_ref_error.py logs/*_43*.json --sim-vs-real
"""

import re
import sys
import json
import argparse
from pathlib import Path

import numpy as np

# constants.py lives one directory up; this script is run by path, so nothing else puts
# src/ on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import OBSERVATION_DOF_ORDER  # noqa: E402

LOG_DIR = "logs"

DEFAULT_REFERENCE = Path.home() / "mjlab_microban/src/mjlab_microban/robot/squat_reference.pkl"

# The clock src/moves/squat_rl.py runs the clip at. The clip carries its own `frequency`;
# if the two disagree the robot is playing the motion at a speed it was never trained on,
# which is worth a warning rather than a silent rescale.
MOVE_FREQUENCY = 0.25  # Hz, squat_rl.SQUAT_FREQUENCY

# Per-joint tolerance of the tracking reward, from the `reference_std` dict in
# microban_squatref_env_cfg.py. Keyed by the joint name with the left_/right_ prefix
# dropped. Legs are tight (the clip actually specifies their motion), arms are loose (the
# clip just pins them at home).
REFERENCE_STD = {
    "shoulder_pitch": 0.2,
    "shoulder_roll": 0.2,
    "elbow": 0.2,
    "hip_roll": 0.1,
    "hip_yaw": 0.1,
    "hip_pitch": 0.15,
    "knee": 0.15,
    "ankle_pitch": 0.1,
    "ankle_roll": 0.1,
}

# Joints the policy actuates, i.e. what it is graded on. The head is in the clip but not
# in the action space, so it is reported separately and left out of every total.
POLICY_JOINTS = list(OBSERVATION_DOF_ORDER)
LEG_JOINTS = [name for name in POLICY_JOINTS if "hip" in name or "knee" in name or "ankle" in name]
ARM_JOINTS = [name for name in POLICY_JOINTS if name not in LEG_JOINTS]

# The sagittal chain — hip pitch, knee, ankle pitch. These six are the squat: the clip
# swings them through 46-96 deg while every other joint is held within ~3 deg, so their
# error is the motion being tracked, where the full leg average mixes it with roll and yaw
# joints that are only being held still.
PITCH_JOINTS = [
    name for name in LEG_JOINTS if name.endswith(("hip_pitch", "knee", "ankle_pitch"))
]

# Joint sets the score can be restricted to with --joints. "pitch" is the squat itself;
# "legs" adds the roll/yaw joints the clip only holds; "all" adds the arms, which the clip
# also only holds. Narrowing the set changes what the headline number means, so the report
# always names the group it scored.
SCORE_GROUPS: dict[str, list[str] | None] = {
    "all": None,
    "legs": LEG_JOINTS,
    "pitch": PITCH_JOINTS,
    "arms": ARM_JOINTS,
}

# A joint whose reference barely moves is being *held*, not tracked; its gain is the ratio
# of two noise floors and means nothing, so it is printed as "-".
MOVING_JOINT_P2P = 0.05  # rad (2.9 deg)

# Lag search range and step. Half a period each way is more than enough: past that the
# search would start locking onto the previous cycle.
LAG_RANGE = 0.6  # s
LAG_STEP = 0.005  # s

PHASE_BINS = 12

# --trunk measures the trunk height off this model rather than the placo odometry: placo
# does not import in this venv, and height needs forward kinematics only — no integration,
# no drift. The corner sites are the ones src/odometry.py plants its support anchor on.
MODEL_PATH = "src/model/mjcf/robot.xml"
TRUNK_BODY = "trunk"
SOLE_CORNER_SITES = (
    "left_foot_front_left",
    "left_foot_front_right",
    "left_foot_back_left",
    "left_foot_back_right",
    "right_foot_front_left",
    "right_foot_front_right",
    "right_foot_back_left",
    "right_foot_back_right",
)

# A tick whose goal is bit-for-bit the previous tick's is not a tick the policy wrote: once
# the move is stopped (or its fall safety returns early) nothing recomputes the targets and
# they are simply held. A trailing run of those is the operator ending the run and is cut
# from the window; a run in the middle is a fall and is reported instead of hidden.
FROZEN_GOAL_EPS = 1e-9  # rad
MIN_FREEZE_S = 0.5


def latest_log() -> Path:
    """Newest log in logs/, the same default plot_log.py uses."""
    logs = sorted(Path(LOG_DIR).glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise SystemExit(f"No logs in {LOG_DIR}/")
    return logs[-1]


def load_reference(path: Path) -> dict:
    """Read the clip pickle written by make_squat_reference.py."""
    import pickle

    if not path.exists():
        raise SystemExit(
            f"No squat reference at {path}.\n"
            "Generate it with (system interpreter, placo does not import in mjlab's venv):\n"
            "  cd ~/mjlab_microban && python3 src/mjlab_microban/scripts/make_squat_reference.py"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


class Unscorable(Exception):
    """This log cannot be lined up against the reference — say why and move on.

    Raised rather than exiting so that a glob over a whole logs/ directory reports the
    handful of files it has to skip instead of dying on the first one.
    """


class Run:
    """One log lined up against the reference over the scored window.

    Everything downstream reads the three [T, J] arrays — `read`, `goal`, `reference` —
    which are all in the same joint order (`self.joints`) and the same units the log and
    the clip already share: motor angles in rad, one column per joint.
    """

    def __init__(
        self,
        path: Path,
        log: dict,
        reference: dict,
        skip_s: float,
        duration_s: float | None,
        cycles: float | None = None,
        group: str = "all",
    ):
        self.path = path
        self.policy = log["metadata"].get("policy")

        t0 = log["metadata"].get("policy_t0")
        if t0 is None:
            raise Unscorable(
                "no policy_t0 — the phase is anchored on it, so there is nothing to line the "
                "clip up against. Start the log before the policy."
            )

        self.frequency = MOVE_FREQUENCY
        clip_frequency = float(reference["frequency"])
        self.frequency_mismatch = abs(clip_frequency - MOVE_FREQUENCY) > 1e-9
        self.clip_frequency = clip_frequency

        # Joints the clip and the log have in common, restricted to what the policy drives.
        clip_names = list(reference["joint_names"])
        self.joints = [name for name in POLICY_JOINTS if name in clip_names and name in log["position"]]
        missing = [name for name in POLICY_JOINTS if name not in self.joints]
        if missing:
            print(f"{path.name}: not in the clip or the log, skipped: {', '.join(missing)}")

        columns = [clip_names.index(name) for name in self.joints]
        self._clip = np.asarray(reference["joint_pos"], dtype=float)[:, columns]  # [S, J]
        self._n_frames = self._clip.shape[0]

        time = np.asarray(log["time"], dtype=float)
        elapsed = time - t0
        window = elapsed >= skip_s
        # The log usually outlives the policy: [l] is pressed after the move is stopped, and
        # the reconstructed clock keeps turning over a robot standing still at home, which
        # would read as a catastrophic tracking failure that never happened.
        self.stopped_at, self.freezes = self._policy_end(log, elapsed)
        if self.stopped_at is not None:
            window &= elapsed <= self.stopped_at
        if duration_s is not None:
            window &= elapsed <= skip_s + duration_s
            cycles = None  # An explicit duration is the caller overriding the cycle count.

        # Score a whole number of squats. The error varies 2-3x across the phase, so a
        # window ending mid-cycle weights whichever part it stopped in: measured over these
        # runs that is worth up to 1.5% of the MAE, small but systematic, and it makes runs
        # of different lengths not quite comparable. Taking an exact multiple of the period
        # removes it -- every phase is then visited the same number of times.
        self.cycles = cycles
        if cycles:
            span = cycles / self.frequency
            first = float(elapsed[window].min())
            last = float(elapsed[window].max())
            tick = float(np.median(np.diff(elapsed[window]))) if window.sum() > 1 else 0.0
            if last < first + span - tick:
                raise Unscorable(
                    f"only {(last - first) * self.frequency:.2f} squat cycles between "
                    f"{first:.1f} and {last:.1f} s; {cycles:g} were asked for."
                )
            window &= elapsed < first + span

        if window.sum() < 2:
            raise Unscorable(
                f"nothing left to score after skipping {skip_s} s of a "
                f"{min(elapsed[-1], self.stopped_at if self.stopped_at is not None else elapsed[-1]):.1f} s "
                "policy run."
            )

        self.time = elapsed[window]
        self.read = self._channel(log["position"], window)
        self.goal = self._channel(log["target_position"], window)

        # A failed servo read is logged as null. Keeping it would turn every metric into
        # NaN, so the tick goes; it is worth saying how many did.
        good = ~(np.isnan(self.read).any(axis=1) | np.isnan(self.goal).any(axis=1))
        if not good.all():
            print(f"{path.name}: {int((~good).sum())} of {len(good)} ticks have a missing read, dropped")
            self.time, self.read, self.goal = self.time[good], self.read[good], self.goal[good]
            if len(self.time) < 2:
                raise Unscorable("almost every tick is missing a servo read.")

        # Which log ticks survived, so --trunk can go back for channels the joint scoring
        # does not need (every joint including the head, and the trunk attitude).
        self.rows = np.flatnonzero(window)[good]
        self._log = log
        self._clip_trunk_z = np.asarray(reference["trunk_z"], dtype=float)
        self._trunk_z: np.ndarray | None = None  # filled on the first --trunk call

        self.reference = self.reference_at(0.0)

        self.phase = (self.time * self.frequency) % 1.0  # cycle fraction in [0, 1)
        self.cycle = np.floor(self.time * self.frequency).astype(int)

        self.std = np.array([REFERENCE_STD[self._base(name)] for name in self.joints])
        self.leg_cols = [i for i, name in enumerate(self.joints) if name in LEG_JOINTS]
        self.arm_cols = [i for i, name in enumerate(self.joints) if name in ARM_JOINTS]
        self.pitch_cols = [i for i, name in enumerate(self.joints) if name in PITCH_JOINTS]

        # The joints every aggregate below is taken over. `--joints all` keeps the full
        # policy set; a narrower group makes the headline MAE, the lag fit, the phase and
        # per-cycle profiles and the comparison table all speak about that group only.
        self.group = group
        names = SCORE_GROUPS[group]
        self.score_cols = (
            list(range(len(self.joints)))
            if names is None
            else [i for i, name in enumerate(self.joints) if name in names]
        )
        if not self.score_cols:
            raise Unscorable(f"no joints of group '{group}' are in this log.")

        # Kept out of every total (not actuated), but worth a line: a head drifting off
        # neutral is the clearest sign the whole robot is leaning.
        self.head_error = None
        if "head" in clip_names and "head" in log["position"]:
            head_read = np.asarray(log["position"]["head"], dtype=float)[window]
            head_clip = np.asarray(reference["joint_pos"], dtype=float)[:, clip_names.index("head")]
            low, alpha = self._frames(0.0)
            high = (low + 1) % self._n_frames
            head_ref = head_clip[low] * (1.0 - alpha[:, 0]) + head_clip[high] * alpha[:, 0]
            self.head_error = head_read - head_ref

    def _policy_end(self, log: dict, elapsed: np.ndarray) -> tuple[float | None, list[tuple[float, float]]]:
        """When the policy stopped writing goals, and any freeze in the middle of the run.

        Returns the elapsed time of the last tick the policy actually authored (None if it
        was still running when the log ended), plus the ``(start, end)`` of every freeze
        longer than `MIN_FREEZE_S` that is *not* that trailing one — those are the run's
        own failures (squat_rl.step returns early on a fall) and stay in the score.
        """
        goals = np.stack([np.asarray(log["target_position"][name], dtype=float) for name in self.joints], axis=1)
        held = np.zeros(len(elapsed), dtype=bool)
        held[1:] = np.abs(np.diff(goals, axis=0)).max(axis=1) < FROZEN_GOAL_EPS
        # Ticks logged before the policy went active are held by whatever ran before it;
        # that is not a freeze of this run.
        held[elapsed < 0.0] = False

        stopped_at = None
        tail = len(held)
        while tail > 1 and held[tail - 1]:
            tail -= 1
        if tail < len(held) and elapsed[-1] - elapsed[tail] >= MIN_FREEZE_S:
            stopped_at = float(elapsed[tail - 1])
            held = held[:tail]

        freezes, start = [], None
        for i, frozen in enumerate(held):
            if frozen and start is None:
                start = i
            elif not frozen and start is not None:
                if elapsed[i - 1] - elapsed[start] >= MIN_FREEZE_S:
                    freezes.append((float(elapsed[start]), float(elapsed[i - 1])))
                start = None
        return stopped_at, freezes

    @staticmethod
    def _base(name: str) -> str:
        return name.split("_", 1)[1] if name.startswith(("left_", "right_")) else name

    def _channel(self, channel: dict, window: np.ndarray) -> np.ndarray:
        """One [T, J] array from a log channel, missing reads (logged null) as NaN."""
        columns = [
            np.array([np.nan if value is None else value for value in channel[name]], dtype=float)[window]
            for name in self.joints
        ]
        return np.stack(columns, axis=1)

    def _frames(self, shift_s: float):
        """Clip rows bracketing each tick, plus the interpolation weight.

        The clip is a loop, so the index wraps; `shift_s` moves the reference forward in
        time, which is what the lag search sweeps.
        """
        index = ((self.time - shift_s) * self.frequency * self._n_frames) % self._n_frames
        low = np.floor(index).astype(int) % self._n_frames
        return low, (index - np.floor(index))[:, None]

    def reference_at(self, shift_s: float) -> np.ndarray:
        """Reference pose per tick, [T, J], the clip linearly interpolated at the phase."""
        low, alpha = self._frames(shift_s)
        high = (low + 1) % self._n_frames
        return self._clip[low] * (1.0 - alpha) + self._clip[high] * alpha

    # Trunk height.

    def reference_trunk_z(self, shift_s: float = 0.0) -> np.ndarray:
        """Reference trunk height per tick [m], the clip's own ``trunk_z`` at the phase."""
        low, alpha = self._frames(shift_s)
        high = (low + 1) % self._n_frames
        alpha = alpha[:, 0]
        return self._clip_trunk_z[low] * (1.0 - alpha) + self._clip_trunk_z[high] * alpha

    def measured_trunk_z(self, model_path: str = MODEL_PATH) -> np.ndarray:
        """Trunk height per tick [m], from forward kinematics on the read joint angles.

        Defined exactly as the clip defines it — the trunk body origin above the ground the
        feet stand on — by putting the read angles into the MuJoCo model, orienting the
        floating base with the logged ``body_quat`` (so a leaning robot is measured
        vertically, as the clip is), and taking the height above the *lowest sole corner*.
        The corner sites are the ones src/odometry.py anchors on. Checked against the clip
        itself: replaying the clip's own frames through this reproduces its ``trunk_z`` to
        0.06 mm, a constant offset, so the two models agree on where the trunk frame is.

        No integration and no dead reckoning is involved, unlike the odometry — height only
        needs the current pose. What it does assume is that a sole corner is on the floor:
        with a foot in the air this measures height above that foot, not above the ground.
        A double-support squat holds to that, but a run that lifts a foot does not.

        NaN on ticks with no IMU reading. Raises `Unscorable` if the log has no
        ``body_quat`` at all (it predates that channel).
        """
        if self._trunk_z is not None:
            return self._trunk_z

        try:
            import mujoco
        except ImportError as error:  # pragma: no cover - depends on the invocation
            raise Unscorable(f"--trunk needs MuJoCo: run with `uv run --group sim --group debug` ({error})")

        quat = self._log.get("body_quat")
        if not quat or all(value is None for value in quat["w"]):
            raise Unscorable("no body_quat in this log — the trunk attitude is unknown, so its height is too.")

        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TRUNK_BODY)
        corners = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in SOLE_CORNER_SITES]
        if trunk < 0 or any(site < 0 for site in corners):
            raise Unscorable(f"{model_path} has no '{TRUNK_BODY}' body or is missing the sole corner sites.")

        # Every joint the model has and the log recorded, the head included: it is part of
        # the kinematic chain even though the policy never drives it.
        addresses = {}
        for name in self._log["position"]:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint >= 0:
                addresses[name] = model.jnt_qposadr[joint]

        angles = {name: np.asarray(self._log["position"][name], dtype=object) for name in addresses}
        attitude = np.stack([np.array([np.nan if v is None else v for v in quat[a]]) for a in "wxyz"], axis=1)

        heights = np.full(len(self.rows), np.nan)
        for i, row in enumerate(self.rows):
            if not np.isfinite(attitude[row]).all():
                continue
            data.qpos[:] = 0.0
            data.qpos[3:7] = attitude[row]
            for name, address in addresses.items():
                value = angles[name][row]
                if value is None:
                    break
                data.qpos[address] = value
            else:
                mujoco.mj_kinematics(model, data)
                heights[i] = data.xpos[trunk][2] - min(data.site_xpos[site][2] for site in corners)
        self._trunk_z = heights
        return heights

    # Errors.

    @property
    def error(self) -> np.ndarray:
        """read - reference, [T, J]. What the task actually grades."""
        return self.read - self.reference

    @property
    def command_error(self) -> np.ndarray:
        """goal - reference, [T, J]. The policy asking for the wrong pose."""
        return self.goal - self.reference

    @property
    def servo_error(self) -> np.ndarray:
        """read - goal, [T, J]. The servo not reaching the pose it was asked for."""
        return self.read - self.goal

    # Aggregates.

    def mae(self, error: np.ndarray, cols: list[int] | None = None) -> float:
        """Mean absolute error over `cols`, defaulting to the scored group."""
        columns = self.score_cols if cols is None else cols
        return float(np.abs(error[:, columns]).mean())

    def reward(self) -> np.ndarray:
        """The training reward per tick: ``exp(-mean(error^2 / std^2))``, in [0, 1].

        Identical kernel to squat_reference.squat_reference_pose, so a run can be read on
        the same scale the reward curves were trained on — bar the arm/head columns mjlab
        weights slightly differently only if the action set changes.
        """
        return np.exp(-np.mean(np.square(self.error) / np.square(self.std), axis=1))

    def lag(self, signal: np.ndarray | None = None) -> tuple[float, float]:
        """Time shift of `signal` against the reference, and the error at that shift.

        A positive lag means the signal is *behind* the clock — it reaches a given pose
        that many seconds after the reference asked for it. Against the zero-shift error
        it says how much of the error is pure delay rather than a wrong shape. Run on the
        goals instead of the reads it measures the other direction: how far the policy
        *leads* the clip to cover the servos' own delay.
        """
        signal = self.read if signal is None else signal
        sel = self.score_cols
        shifts = np.arange(-LAG_RANGE, LAG_RANGE + LAG_STEP, LAG_STEP)
        errors = [np.abs(signal[:, sel] - self.reference_at(shift)[:, sel]).mean() for shift in shifts]
        best = int(np.argmin(errors))
        return float(shifts[best]), float(errors[best])

    def gain_bias(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-joint amplitude ratio and mean offset against the reference.

        The gain is the least-squares slope of read against reference once both are
        centred, so it is the amplitude the robot really produced over the amplitude it
        was asked for: 0.9 is a squat 10% too shallow. NaN where the reference barely
        moves (the ratio would be noise over noise).
        """
        gains = np.full(len(self.joints), np.nan)
        for j in range(len(self.joints)):
            reference = self.reference[:, j]
            if np.ptp(reference) < MOVING_JOINT_P2P:
                continue
            centred = reference - reference.mean()
            gains[j] = float(np.dot(centred, self.read[:, j] - self.read[:, j].mean()) / np.dot(centred, centred))
        return gains, self.read.mean(axis=0) - self.reference.mean(axis=0)

    def mean_gain(self) -> float:
        """One amplitude ratio for the run, weighted by how much each joint travels.

        A plain mean would give the 0.05 rad roll joints the same vote as the 1.67 rad
        knees, and their slope is mostly noise over noise — enough of it to swing the
        headline number by 4% while the squat itself is on depth. Weighting by reference
        travel makes this the question it looks like: was the squat as deep as asked?
        """
        gains, _ = self.gain_bias()
        travel = np.array([np.ptp(self.reference[:, j]) for j in range(len(self.joints))])
        moving = np.isfinite(gains)
        outside = np.ones(len(gains), dtype=bool)
        outside[self.score_cols] = False
        moving &= ~outside  # Only the scored group votes.
        if not moving.any():
            return float("nan")
        return float(np.sum(gains[moving] * travel[moving]) / np.sum(travel[moving]))

    def per_cycle_mae(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean absolute error of each *complete* squat cycle in the window.

        Cycle by cycle rather than one number for the run: a policy that tracks well for
        three cycles and then settles into a wrong posture, or one that is slowly drifting
        off, reads as a perfectly good average.
        """
        cycles, maes = [], []
        # 90% of the ticks a full cycle should hold, so a cycle clipped by the window edge
        # (or by a few dropped ticks) is dropped rather than scored on a fragment.
        expected = 0.9 / (self.frequency * np.median(np.diff(self.time)))
        for cycle in np.unique(self.cycle):
            rows = self.cycle == cycle
            if rows.sum() < expected:  # Partial cycle at either end of the window.
                continue
            cycles.append(int(cycle))
            maes.append(float(np.abs(self.error[rows][:, self.score_cols]).mean()))
        return np.array(cycles), np.array(maes)

    def phase_profile(self, bins: int = PHASE_BINS) -> np.ndarray:
        """Mean absolute error in `bins` slices of the cycle, NaN where nothing landed.

        Phase 0 is the top of the squat (the clip starts at the home pose), 0.5 the
        bottom, so the profile says whether the robot loses the reference on the way down,
        at the bottom, or coming back up.
        """
        edges = np.clip((self.phase * bins).astype(int), 0, bins - 1)
        profile = np.full(bins, np.nan)
        for b in range(bins):
            rows = edges == b
            if rows.any():
                profile[b] = float(np.abs(self.error[rows][:, self.score_cols]).mean())
        return profile


def deg(radians):
    """Radians to degrees, for printing only — every computation above stays in radians."""
    return np.degrees(radians)


def bar(value: float, scale: float, width: int = 24) -> str:
    if not np.isfinite(value) or scale <= 0:
        return ""
    return "#" * max(1, min(width, round(width * value / scale)))


def trunk_report(run: Run, model_path: str) -> None:
    """The squat as a height rather than a set of angles, in mm."""
    try:
        measured = run.measured_trunk_z(model_path)
    except Unscorable as reason:
        print(f"\n-- trunk height -----------------------------------------------------------------------")
        print(f"unavailable: {reason}")
        return

    reference = run.reference_trunk_z()
    ok = np.isfinite(measured)
    if not ok.any():
        print("\n-- trunk height: no tick has both a pose and an IMU reading ---------------------------")
        return

    measured, reference, phase = measured[ok], reference[ok], run.phase[ok]
    error = (measured - reference) * 1000.0  # mm

    # Same regression as the joint gain: how much of the commanded travel the trunk really
    # made, independent of any constant offset.
    centred = reference - reference.mean()
    gain = float(np.dot(centred, measured - measured.mean()) / np.dot(centred, centred))

    shifts = np.arange(-LAG_RANGE, LAG_RANGE + LAG_STEP, LAG_STEP)
    lags = [np.abs(measured - run.reference_trunk_z(shift)[ok]).mean() for shift in shifts]
    lag = float(shifts[int(np.argmin(lags))])

    print("\n-- trunk height (mm, trunk origin above the lowest sole corner) ------------------------")
    if not ok.all():
        print(f"              {int((~ok).sum())} of {len(ok)} ticks skipped (no IMU reading)")
    print(
        f"reference     {reference.min() * 1000:.1f} .. {reference.max() * 1000:.1f} mm "
        f"(travel {np.ptp(reference) * 1000:.1f})"
    )
    print(
        f"measured      {measured.min() * 1000:.1f} .. {measured.max() * 1000:.1f} mm "
        f"(travel {np.ptp(measured) * 1000:.1f})"
    )
    print(f"MAE           {np.abs(error).mean():.2f} mm   RMSE {np.sqrt(np.mean(error**2)):.2f}   max {np.abs(error).max():.2f}")
    bias = error.mean()
    verdict = "on the reference" if abs(bias) < 0.5 else ("standing tall" if bias > 0 else "sitting low")
    print(f"bias          {bias:+.2f} mm   ({verdict})")
    print(f"gain          {gain:.3f}   lag {lag * 1000:+.0f} ms")

    profile = np.full(PHASE_BINS, np.nan)
    edges = np.clip((phase * PHASE_BINS).astype(int), 0, PHASE_BINS - 1)
    for b in range(PHASE_BINS):
        rows = edges == b
        if rows.any():
            profile[b] = float(error[rows].mean())  # signed: too high or too low, not just off
    print("  signed error over the cycle (phase 0 = top, 0.5 = bottom):")
    scale = np.nanmax(np.abs(profile))
    for b, value in enumerate(profile):
        print(f"  {b / PHASE_BINS:.2f}-{(b + 1) / PHASE_BINS:.2f}  {value:+6.2f}  {bar(abs(value), scale)}")


def report(run: Run, model_path: str | None = None) -> None:
    print(f"\n{'=' * 92}")
    print(f"{run.path.name}   policy={run.policy}")
    print("=" * 92)

    if run.policy not in (None, "squat_rl"):
        print(
            f"WARNING: this log was recorded with '{run.policy}', not squat_rl — the phase "
            "reconstructed below is fiction unless that move runs the same clock."
        )
    if run.frequency_mismatch:
        print(
            f"WARNING: clip was generated at {run.clip_frequency} Hz but squat_rl replays it at "
            f"{MOVE_FREQUENCY} Hz — the robot is squatting at a speed it was not trained on."
        )

    for start, end in run.freezes:
        print(
            f"WARNING: the policy wrote no new goals from {start:.2f} to {end:.2f} s — squat_rl "
            "returns early once projected gravity says the robot is down. Those ticks are "
            "kept in the score; the run fell over."
        )

    cycles, cycle_maes = run.per_cycle_mae()
    reward = run.reward()
    lag, lag_mae = run.lag()
    goal_lag, _ = run.lag(run.goal)
    mae = run.mae(run.error)

    # With --cycles the span is an exact multiple of the period, which is not the same as
    # the count of phase-ALIGNED cycles the per-cycle table below can score (the window
    # starts mid-squat), so the two numbers differ by one and both are worth stating.
    span = run.time[-1] - run.time[0]
    if run.cycles:
        covered = f"{span:.2f} s = exactly {run.cycles:g} squat cycles of {1.0 / run.frequency:.1f} s"
    else:
        covered = f"{span:.2f} s, {len(cycles)} whole cycles of {1.0 / run.frequency:.1f} s"
    print(
        f"\nwindow        {run.time[0]:.2f} .. {run.time[-1]:.2f} s after policy start "
        f"({len(run.time)} ticks, {covered})"
    )
    if run.stopped_at is not None:
        print(f"              policy stopped writing goals at {run.stopped_at:.2f} s; the rest of the log is cut")
    print(
        f"joints        {len(run.joints)} policy joints ({len(run.leg_cols)} leg of which "
        f"{len(run.pitch_cols)} pitch, {len(run.arm_cols)} arm)"
    )
    if run.group != "all":
        print(f"scored on     '{run.group}' only -- {len(run.score_cols)} joints; every number below is that group")

    print("\n-- tracking error (deg, mean absolute) ------------------------------------------------")
    label = "all policy joints" if run.group == "all" else f"'{run.group}' joints"
    print(f"reference     {deg(mae):.3f}   {label}   <- the score")
    print(f"                {deg(run.mae(run.error, run.leg_cols)):.3f}   legs")
    print(f"                {deg(run.mae(run.error, run.arm_cols)):.3f}   arms")
    print(
        f"                {deg(run.mae(run.error, run.pitch_cols)):.3f}   pitch chain "
        f"({len(run.pitch_cols)} joints: hip pitch, knee, ankle pitch -- the squat itself)"
    )
    print(f"command       {deg(run.mae(run.command_error)):.3f}   goal vs reference (policy's own choice)")
    print(f"servo         {deg(run.mae(run.servo_error)):.3f}   read vs goal (servo shortfall)")
    if run.mae(run.command_error) > mae:
        print(
            "              command and servo both exceed the reference error: the policy is "
            "deliberately overdriving the goals and the servo lag brings them back."
        )
    if run.head_error is not None:
        print(f"head          {deg(np.abs(run.head_error).mean()):.3f}   (not actuated, not in the totals)")

    print("\n-- training reward exp(-mean(e^2/std^2)) ----------------------------------------------")
    print(
        f"mean {reward.mean():.3f}   p10 {np.percentile(reward, 10):.3f}   "
        f"min {reward.min():.3f}   (1.0 = perfect, the reward the run would have earned; "
        "unitless, its per-joint std are the env's, in rad)"
        + ("\n              always over all 18 joints -- it is the env's reward, --joints does not narrow it"
           if run.group != "all" else "")
    )

    print("\n-- shape ------------------------------------------------------------------------------")
    print(
        f"lag           {lag * 1000:+.0f} ms   read behind the clip (error {deg(lag_mae):.3f} deg there "
        f"vs {deg(mae):.3f} at zero shift -> {100 * (1 - lag_mae / mae):.0f}% of the error is pure delay)"
    )
    print(f"lead          {-goal_lag * 1000:+.0f} ms   goal ahead of the clip (what the policy adds to beat the servos)")

    gains, biases = run.gain_bias()
    moving = np.isfinite(gains)
    if moving.any():
        mean_gain = run.mean_gain()
        verdict = "on target" if abs(mean_gain - 1.0) < 0.02 else ("too shallow" if mean_gain < 1.0 else "overshooting")
        print(
            f"gain          {mean_gain:.3f} over the {int(moving.sum())} joints the clip really "
            f"moves, weighted by travel ({verdict})"
        )

    print("\n-- per joint (deg) --------------------------------------------------------------------")
    print(f"{'joint':<22}{'ref p2p':>9}{'MAE':>9}{'RMSE':>9}{'max':>9}{'bias':>9}{'gain':>8}{'cmd':>9}{'servo':>9}")
    order = sorted(run.score_cols, key=lambda j: -np.abs(run.error[:, j]).mean())
    for j in order:
        error = run.error[:, j]
        print(
            f"{run.joints[j]:<22}"
            f"{deg(np.ptp(run.reference[:, j])):>9.2f}"
            f"{deg(np.abs(error).mean()):>9.3f}"
            f"{deg(np.sqrt(np.mean(error**2))):>9.3f}"
            f"{deg(np.abs(error).max()):>9.3f}"
            f"{deg(biases[j]):>+9.3f}"
            f"{(f'{gains[j]:.3f}' if np.isfinite(gains[j]) else '-'):>8}"
            f"{deg(np.abs(run.command_error[:, j]).mean()):>9.3f}"
            f"{deg(np.abs(run.servo_error[:, j]).mean()):>9.3f}"
        )

    profile = run.phase_profile()
    print("\n-- error over the cycle, deg (phase 0 = top of the squat, 0.5 = bottom) ---------------")
    scale = np.nanmax(profile)
    for b, value in enumerate(profile):
        print(f"  {b / len(profile):.2f}-{(b + 1) / len(profile):.2f}  {deg(value):.3f}  {bar(value, scale)}")

    if model_path is not None:
        trunk_report(run, model_path)

    if len(cycles) > 1:
        print("\n-- error per cycle (deg) --------------------------------------------------------------")
        scale = cycle_maes.max()
        for cycle, value in zip(cycles, cycle_maes):
            print(f"  cycle {cycle:>3}  {deg(value):.3f}  {bar(value, scale)}")
        drift = np.polyfit(np.arange(len(cycle_maes)), cycle_maes, 1)[0]
        print(f"  spread {deg(cycle_maes.std()):.3f} deg, trend {deg(drift):+.4f} deg/cycle")


def pair_key(path: Path) -> tuple[str, bool]:
    """Split a log name into its ``(run identity, is_sim)``.

    The convention these logs follow is ``<timestamp>_squat_<model>_<seed>[_sim]``, so the
    identity is the name with the timestamp and a *trailing* ``_sim`` removed. Anything
    after ``_sim`` (``_sim_blend``, ``_sim71``, ``_sim_1ohms``) is a one-off variant and
    keeps that suffix in its identity, so it never silently pairs with a plain real run.
    """
    stem = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_?", "", path.stem)
    if stem.endswith("_sim"):
        return stem[: -len("_sim")], True
    return stem, False


def pair_logs(paths: list[Path]) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Match each real log with the sim log of the same identity.

    Returns the pairs and whatever could not be matched, so an unpaired file is reported
    rather than dropped -- a missing counterpart is the kind of thing that would otherwise
    quietly shrink a comparison table.
    """
    real: dict[str, Path] = {}
    sim: dict[str, Path] = {}
    for path in paths:
        key, is_sim = pair_key(path)
        (sim if is_sim else real)[key] = path
    pairs = [(real[k], sim[k]) for k in real if k in sim]
    orphans = [p for k, p in {**real, **sim}.items() if not (k in real and k in sim)]
    return pairs, orphans


def sim_vs_real(real: Run, sim: Run) -> tuple[np.ndarray, np.ndarray]:
    """Sim minus real, ``[T, J]``, on the real run's ticks, plus those tick times.

    Both logs are anchored on their own ``policy_t0`` and both replay the same
    ``MOVE_FREQUENCY`` clock off the wall clock, so equal elapsed time is the same moment
    of the same squat -- that is what makes the two runs comparable without any alignment
    fitting. The sim is linearly resampled onto the real ticks (it samples ~1.5% denser)
    and the comparison is restricted to the overlap of the two windows.

    This measures something different from the reference error: not "did the robot follow
    the clip" but "did the simulator reproduce the robot", which is the question a sim2real
    gap is actually about.
    """
    if real.joints != sim.joints:
        raise Unscorable("the two logs do not carry the same joints, so they cannot be compared.")
    low = max(real.time[0], sim.time[0])
    high = min(real.time[-1], sim.time[-1])
    keep = (real.time >= low) & (real.time <= high)
    if keep.sum() < 2:
        raise Unscorable("the real and sim windows do not overlap.")
    times = real.time[keep]
    resampled = np.stack(
        [np.interp(times, sim.time, sim.read[:, j]) for j in range(len(sim.joints))], axis=1
    )
    return resampled - real.read[keep], times


def pair_report(real: Run, sim: Run) -> np.ndarray:
    """Print how far the simulation drifted from the robot it is meant to reproduce."""
    delta, times = sim_vs_real(real, sim)
    print(f"\n{'=' * 92}")
    print(f"SIM vs REAL   {sim.path.name}\n              against {real.path.name}")
    print("=" * 92)
    print(
        f"window        {times[0]:.2f} .. {times[-1]:.2f} s after policy start "
        f"({len(times)} ticks, {(times[-1] - times[0]) * real.frequency:.2f} squat cycles)"
    )

    def m(cols):
        return deg(np.abs(delta[:, cols]).mean())

    label = "all policy joints" if real.group == "all" else f"'{real.group}' joints"
    print("\n-- simulator error (deg, mean absolute |sim - real|) ----------------------------------")
    print(f"scored        {m(real.score_cols):.3f}   {label}   <- how well the sim reproduces the robot")
    print(f"                {m(real.pitch_cols):.3f}   pitch chain")
    print(f"                {m(real.leg_cols):.3f}   legs")
    print(f"                {m(real.arm_cols):.3f}   arms")
    print(
        f"\nfor scale     real vs reference {deg(real.mae(real.error)):.3f}, "
        f"sim vs reference {deg(sim.mae(sim.error)):.3f} -- the simulator's own error is "
        f"{100 * m(real.score_cols) / deg(real.mae(real.error)):.0f}% of the tracking error it is modelling"
    )

    print("\n-- per joint (deg) --------------------------------------------------------------------")
    print(f"{'joint':<22}{'|sim-real|':>11}{'bias':>9}{'max':>9}{'real vs ref':>13}{'sim vs ref':>12}")
    for j in sorted(real.score_cols, key=lambda j: -np.abs(delta[:, j]).mean()):
        print(
            f"{real.joints[j]:<22}"
            f"{deg(np.abs(delta[:, j]).mean()):>11.3f}"
            f"{deg(delta[:, j].mean()):>+9.3f}"
            f"{deg(np.abs(delta[:, j]).max()):>9.3f}"
            f"{deg(np.abs(real.error[:, j]).mean()):>13.3f}"
            f"{deg(np.abs(sim.error[:, j]).mean()):>12.3f}"
        )
    return delta


def pair_compare(results: list[tuple[Run, Run, np.ndarray]]) -> None:
    """One row per pair: what the robot did, and how well the sim reproduced it."""
    print(f"\n{'=' * 92}")
    print("sim vs real (best first)")
    print("=" * 92)
    print(
        f"{'run':<26}{'sim-real':>10}{'pitch':>8}{'legs':>8}{'arms':>8}   "
        f"{'real/ref':>9}{'sim/ref':>9}"
    )
    rows = []
    for real, sim, delta in results:
        rows.append(
            (
                deg(np.abs(delta[:, real.score_cols]).mean()),
                short_label(real.path),
                deg(np.abs(delta[:, real.pitch_cols]).mean()),
                deg(np.abs(delta[:, real.leg_cols]).mean()),
                deg(np.abs(delta[:, real.arm_cols]).mean()),
                deg(real.mae(real.error)),
                deg(sim.mae(sim.error)),
            )
        )
    for scored, name, pitch, legs, arms, rr, sr in sorted(rows):
        print(f"{name:<26}{scored:>10.3f}{pitch:>8.3f}{legs:>8.3f}{arms:>8.3f}   {rr:>9.3f}{sr:>9.3f}")


def short_label(path: Path) -> str:
    """`2026-08-26_15-01-38_squat_m6_sim.json` -> `m6_sim`, for a column heading.

    Falls back to the stem when the name does not carry the logger's date stamp, so a
    hand-renamed log still gets a heading rather than an empty one.
    """
    stem = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_?", "", path.stem)
    stem = re.sub(r"^squat_?", "", stem)
    return (stem or path.stem)[:8]


def trunk_mae_mm(run: Run, model_path: str) -> float:
    """Trunk height MAE in mm, NaN when this log cannot give one."""
    try:
        measured = run.measured_trunk_z(model_path)
    except Unscorable:
        return float("nan")
    ok = np.isfinite(measured)
    if not ok.any():
        return float("nan")
    return float(np.abs(measured[ok] - run.reference_trunk_z()[ok]).mean() * 1000.0)


def compare(runs: list[Run], model_path: str | None = None) -> None:
    print(f"\n{'=' * 92}")
    print("comparison (best first)")
    print("=" * 92)
    trunk_header = f"{'trunk mm':>10}" if model_path else ""
    # Name the group in the header when it is not the whole set, so a narrowed run cannot be
    # mistaken for a full-set one in a pasted table.
    scored = "MAE deg" if runs[0].group == "all" else f"MAE({runs[0].group})"
    print(
        f"{'log':<44}{scored:>9}{'pitch':>9}{'legs':>9}{'arms':>9}"
        f"{'reward':>9}{'lag ms':>9}{'gain':>8}{trunk_header}"
    )
    rows = []
    for run in runs:
        rows.append(
            (
                run.mae(run.error),
                run.path.name,
                run.mae(run.error, run.pitch_cols),
                run.mae(run.error, run.leg_cols),
                run.mae(run.error, run.arm_cols),
                run.reward().mean(),
                run.lag()[0] * 1000,
                run.mean_gain(),
                trunk_mae_mm(run, model_path) if model_path else None,
            )
        )
    for mae, name, pitch, legs, arms, reward, lag, gain, trunk in sorted(rows):
        print(
            f"{name:<44}{deg(mae):>9.3f}{deg(pitch):>9.3f}{deg(legs):>9.3f}{deg(arms):>9.3f}"
            f"{reward:>9.3f}{lag:>+9.0f}{gain:>8.3f}"
            + (f"{trunk:>10.2f}" if trunk is not None else "")
        )

    # Per joint, side by side: the summary says which run is better, this says where. The
    # joints are ordered by how far the clip moves them, so the ones the squat is actually
    # made of come first and the pinned arms fall to the bottom.
    print("\nMAE per joint (deg), reference unshifted and unscaled:")
    print(f"{'joint':<22}{'ref p2p':>9}" + "".join(f"{short_label(run.path):>9}" for run in runs))
    order = sorted(range(len(runs[0].joints)), key=lambda j: -np.ptp(runs[0].reference[:, j]))
    for j in order:
        name = runs[0].joints[j]
        cells = "".join(f"{deg(np.abs(run.error[:, run.joints.index(name)]).mean()):>9.3f}" for run in runs)
        print(f"{name:<22}{deg(np.ptp(runs[0].reference[:, j])):>9.2f}{cells}")
    print(
        f"{'all policy joints':<22}{'':>9}" + "".join(f"{deg(run.mae(run.error)):>9.3f}" for run in runs)
    )


def plot(runs: list[Run], model_path: str | None = None) -> None:
    """Reference against read for the joints the clip moves most, plus the two profiles.

    With `model_path` set the squat also gets a panel in millimetres of trunk height, which
    is the one panel that shows the motion rather than the joints it is made of.
    """
    import matplotlib.pyplot as plt

    # The joints the clip moves most, one per mirrored pair — the clip is symmetric, so
    # plotting both knees would spend a panel saying the same thing twice. Picked off the
    # first run and reused for the rest so the panels stay comparable.
    first = runs[0]
    names, seen = [], set()
    for j in sorted(range(len(first.joints)), key=lambda j: -np.ptp(first.reference[:, j])):
        base = Run._base(first.joints[j])
        if base in seen:
            continue
        seen.add(base)
        names.append(first.joints[j])
        if len(names) == 3:
            break

    trunk = {}
    if model_path is not None:
        for run in runs:
            try:
                heights = run.measured_trunk_z(model_path)
            except Unscorable:
                continue
            if np.isfinite(heights).any():
                trunk[run.path] = heights

    rows = len(names) + 2 + (1 if trunk else 0)
    fig, axes = plt.subplots(rows, 1, figsize=(13, 3 + 2 * rows))

    if trunk:
        ax = axes[0]
        ax.plot(first.time, first.reference_trunk_z() * 1000.0, color="black", ls="--", lw=1.2, label="reference")
        for run in runs:
            if run.path in trunk:
                ax.plot(run.time, trunk[run.path] * 1000.0, lw=1.0, label=run.path.stem[:24])
        ax.set_ylabel("trunk\n(mm)", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=3)
        axes = axes[1:]

    for row, name in enumerate(names):
        ax = axes[row]
        column = first.joints.index(name)
        ax.plot(first.time, deg(first.reference[:, column]), color="black", ls="--", lw=1.2, label="reference")
        for run in runs:
            column = run.joints.index(name)
            ax.plot(run.time, deg(run.read[:, column]), lw=1.0, label=f"read {run.path.stem[:24]}")
        ax.set_ylabel(f"{name}\n(deg)", fontsize=8)
        ax.grid(alpha=0.3)
        if row == 0:
            ax.legend(fontsize=7, ncol=3)
        if row == len(names) - 1:
            ax.set_xlabel("time since policy start (s)", fontsize=8)
        ax.sharex(fig.axes[0])

    ax = axes[-2]
    for run in runs:
        profile = run.phase_profile()
        centres = (np.arange(len(profile)) + 0.5) / len(profile)
        ax.plot(centres, deg(profile), marker="o", ms=3, label=run.path.stem[:24])
    ax.set_xlabel("phase in the squat cycle (0 = top, 0.5 = bottom)", fontsize=8)
    ax.set_ylabel("MAE (deg)", fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[-1]
    for run in runs:
        cycles, maes = run.per_cycle_mae()
        ax.plot(cycles, deg(maes), marker="o", ms=3, label=run.path.stem[:24])
    ax.set_xlabel("squat cycle", fontsize=8)
    ax.set_ylabel("MAE (deg)", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("squat reference tracking", fontsize=10)
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="*", type=Path, help=f"squat_rl logs to score (default: newest in {LOG_DIR}/)")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="squat_reference.pkl to score against")
    parser.add_argument(
        "--skip",
        type=float,
        default=2.0,
        help="seconds after policy start to drop, while the robot is still settling onto "
        "the reference from wherever it was standing (default: 2)",
    )
    parser.add_argument("--duration", type=float, default=None, help="score only this many seconds after --skip (overrides --cycles)")
    parser.add_argument(
        "--cycles",
        type=float,
        default=8.0,
        help="score exactly this many squat cycles from the start of the window, so runs of "
        "different lengths are compared over the same phase coverage (0 = use the whole "
        "window; default: 8)",
    )
    parser.add_argument(
        "--trunk",
        action="store_true",
        help="also score the trunk height in mm, by running the read joint angles through "
        f"{MODEL_PATH} (needs --group sim)",
    )
    parser.add_argument(
        "--joints",
        choices=tuple(SCORE_GROUPS),
        default="all",
        help="restrict every score to a joint group: 'pitch' is the squat itself (hip pitch, "
        "knee, ankle pitch), 'legs' adds the roll/yaw joints the clip only holds, 'arms' is "
        "the upper body, 'all' is the whole policy set (default: all)",
    )
    parser.add_argument(
        "--sim-vs-real",
        action="store_true",
        help="pair each real log with the '<same name>_sim' log among those given and report "
        "how far the simulation drifted from the robot, instead of scoring each against the "
        "reference",
    )
    parser.add_argument("--plot", action="store_true", help="also show the traces (needs --group debug)")
    args = parser.parse_args()

    reference = load_reference(args.reference)
    paths = args.logs or [latest_log()]

    print(
        f"reference {args.reference}\n"
        f"  {reference['joint_pos'].shape[0]} frames, dt {reference['dt']:.3f} s, "
        f"{reference['frequency']} Hz, amplitude {reference['amplitude']} m, "
        f"trunk_z {reference['trunk_z'].min():.3f}..{reference['trunk_z'].max():.3f} m"
    )

    def build(path: Path) -> Run:
        return Run(
            path, json.loads(path.read_text()), reference,
            args.skip, args.duration, args.cycles, args.joints,
        )

    if args.sim_vs_real:
        pairs, orphans = pair_logs(paths)
        for path in orphans:
            print(f"\nskipping {path.name}: no counterpart among the logs given.")
        if not pairs:
            raise SystemExit(
                "Nothing to pair. Give both a real log and its '_sim' twin, e.g.\n"
                "  logs/..._squat_m1_43.json logs/..._squat_m1_43_sim.json"
            )
        results = []
        for real_path, sim_path in sorted(pairs):
            try:
                real, sim = build(real_path), build(sim_path)
                results.append((real, sim, pair_report(real, sim)))
            except Unscorable as reason:
                print(f"\nskipping {real_path.name} / {sim_path.name}: {reason}")
        if not results:
            raise SystemExit("Nothing to compare.")
        if len(results) > 1:
            pair_compare(results)
        return

    runs = []
    for path in paths:
        try:
            runs.append(build(path))
        except Unscorable as reason:
            print(f"\nskipping {path.name}: {reason}")
    if not runs:
        raise SystemExit("Nothing to score.")

    for run in runs:
        report(run, MODEL_PATH if args.trunk else None)
    if len(runs) > 1:
        compare(runs, MODEL_PATH if args.trunk else None)
    if args.plot:
        plot(runs, MODEL_PATH if args.trunk else None)


if __name__ == "__main__":
    main()
