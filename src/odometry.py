# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Kinematic odometry from joint angles and the IMU, using placo.

This is a Python transcription of the estimator the rhoban humanoid runs online
(``humanoid/services/model_service.cpp``). Every tick:

1. the measured joint positions and velocities go into the placo model;
2. the floating base *angular* velocity is taken straight from the gyroscope;
3. the sole corner that sits lowest becomes the new support (anchor) frame;
4. the base *linear* velocity is solved for by requiring that anchor to be immobile;
5. the whole robot is re-hung from that anchor, oriented by the IMU.

Steps 3-5 are what makes it an odometry: the anchor is only re-planted where the
kinematic chain says it currently is, so the trunk's world position accumulates from
step to step. Nothing observes absolute position, so it drifts — every slip or
mis-modelled contact is integrated and never corrected. Heading likewise comes from
the IMU's own yaw, which is gyro-integrated (the BMI088 has no magnetometer).

The corner frames come from the sites added per foot in ``src/model/mjcf/robot.xml``
(``left_foot_front_left`` and friends). A single ``left_foot``/``right_foot`` anchor
would pin the whole sole flat on the floor and lose the roll-off through a step; the
corners let the anchor migrate heel-to-toe the way the real foot does.

Typical use is offline, replaying a recorded session log:

    from odometry import replay_log
    samples = replay_log(json.loads(Path("logs/run.json").read_text()))
    print(samples[-1].T_world_trunk[:3, 3])   # where the robot ended up
"""

from dataclasses import dataclass, field

import numpy as np
import placo

from constants import MOTOR_TO_ID, NEUTRAL_POSE, IMU_MOUNT_QUAT

DEFAULT_MODEL_PATH = "src/model/mjcf/robot.xml"

# Candidate anchor points, tried every tick — the lowest one wins. Only the sole corners
# are listed: the mid-foot frames would never be the lowest point of a tilted foot, and a
# foot flat on the floor keeps whichever corner it already had (the test is strict).
CONTACT_FRAMES: tuple[str, ...] = (
    "left_foot_front_left",
    "left_foot_front_right",
    "left_foot_back_left",
    "left_foot_back_right",
    "right_foot_front_left",
    "right_foot_front_right",
    "right_foot_back_left",
    "right_foot_back_right",
)

# placo's HumanoidRobot starts anchored on the whole left sole; the first tick that finds
# a lower corner takes over from it.
INITIAL_SUPPORT_FRAME = "left_foot"
INITIAL_SUPPORT_SIDE = "left"

# Logs are recorded at the 50 Hz control rate, and the odometry is replayed on a finer grid
# than that. The anchor only ever moves *between* ticks — it is re-planted wherever forward
# kinematics says the new corner is at the moment the switch is noticed — so a coarse grid
# bakes up to a full tick of unmodelled foot travel into every transfer. At 20 ms a swinging
# foot covers centimetres; at 5 ms it covers a quarter of that, and the transfer lands much
# closer to the instant the corner actually touched down.
ODOMETRY_DT = 0.005


def quat_to_matrix(quat) -> np.ndarray:
    """Rotation matrix from a (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in quat)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


# The IMU is bolted on rotated, and its readings are logged raw, in the sensor frame (the
# `body_quat` channel is already converted, the `gyro` one is not — see imu_reader.py).
# IMU_MOUNT_QUAT is the `imu` site orientation inside the trunk body, i.e. R_trunk_imu.
R_TRUNK_IMU = quat_to_matrix(IMU_MOUNT_QUAT)


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z) quaternion from a rotation matrix — the inverse of `quat_to_matrix`.

    Branches on which diagonal term is largest so the divisor is never near zero, which a
    single formula would hit for rotations near 180 deg.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return np.array(q)


def gyro_to_trunk(gyro) -> np.ndarray:
    """Raw gyroscope reading (IMU frame, rad/s) as an angular velocity in the trunk frame."""
    return R_TRUNK_IMU @ np.asarray(gyro, dtype=float)


def yaw_of(R: np.ndarray) -> float:
    """Heading of a rotation matrix, in radians."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


class Odometry:
    """placo model kept anchored on the robot's lowest contact point.

    One instance carries the integrated state, so ticks must be fed in order.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self.robot = placo.HumanoidRobot(model_path, placo.Flags.mjcf)
        self._support_frame = INITIAL_SUPPORT_FRAME
        self._support_side = INITIAL_SUPPORT_SIDE
        self._body_velocity = np.zeros(3)

    @property
    def support_frame(self) -> str:
        """Frame currently assumed immobile in the world."""
        return self._support_frame

    @property
    def support_side(self) -> str:
        """`"left"` or `"right"` — which foot the support frame belongs to."""
        return self._support_side

    @property
    def T_world_trunk(self) -> np.ndarray:
        return self.robot.get_T_world_frame("trunk")

    @property
    def body_velocity(self) -> np.ndarray:
        """Trunk linear velocity, expressed in the trunk frame (m/s)."""
        return self._body_velocity

    @property
    def omega_trunk(self) -> np.ndarray:
        """Trunk angular velocity in the trunk frame (rad/s) — roll, pitch, yaw rates.

        This is the gyroscope with the mount rotation taken out, which is what the
        estimator actually uses. Reading a yaw rate off the raw log channel means knowing
        that `gyro y` is the *negative* of it; this component is the thing itself.
        """
        return self.robot.state.qd[3:6].copy()

    @property
    def world_velocity(self) -> np.ndarray:
        """Trunk linear velocity, expressed in the world frame (m/s)."""
        return self.T_world_trunk[:3, :3] @ self._body_velocity

    def reset(self, joints: dict[str, float], R_world_trunk: np.ndarray | None = None) -> None:
        """Put the robot in `joints`, standing on its left sole at the world origin."""
        for name, angle in joints.items():
            self.robot.set_joint(name, float(angle))
        self.robot.state.qd[:] = 0.0
        self.robot.update_kinematics()

        self._support_frame = INITIAL_SUPPORT_FRAME
        self._support_side = INITIAL_SUPPORT_SIDE
        self.robot.update_support_side_with_frame(INITIAL_SUPPORT_SIDE, INITIAL_SUPPORT_FRAME)
        self.robot.set_T_world_support(np.eye(4))
        self.robot.ensure_on_floor_oriented(
            np.eye(3) if R_world_trunk is None else np.asarray(R_world_trunk, dtype=float)
        )
        self._body_velocity = np.zeros(3)

    def update(
        self,
        joints: dict[str, float],
        joint_velocities: dict[str, float],
        R_world_trunk: np.ndarray,
        omega_trunk: np.ndarray,
    ) -> np.ndarray:
        """Feed one tick of measurements; returns the resulting T_world_trunk.

        `omega_trunk` is the gyroscope in the *trunk* frame — pass raw readings through
        `gyro_to_trunk` first. `R_world_trunk` is the IMU attitude, already in the trunk
        frame (the `body_quat` log channel).
        """
        for name, angle in joints.items():
            self.robot.set_joint(name, float(angle))
        for name, velocity in joint_velocities.items():
            self.robot.set_joint_velocity(name, float(velocity))

        # qd = [base linear (3), base angular (3), joints...], all in the base (trunk)
        # frame for a free-flyer. The gyroscope *is* the base angular velocity; the linear
        # part is solved for below.
        self.robot.state.qd[3:6] = np.asarray(omega_trunk, dtype=float)

        # Forward kinematics with the new joints but the previous anchoring, which is what
        # the support test below compares heights in.
        self.robot.update_kinematics()

        self._update_support()
        self._body_velocity = self.estimate_body_velocity()

        # Re-hang the robot from its support, at the attitude the IMU reports. This is
        # where the odometry actually accumulates: the support keeps the world pose it was
        # planted with, so the trunk moves to wherever the leg says it must be.
        self.robot.ensure_on_floor_oriented(np.asarray(R_world_trunk, dtype=float))

        return self.T_world_trunk

    def estimate_body_velocity(self) -> np.ndarray:
        """Trunk linear velocity implied by an immobile support and the measured motion.

        The support frame is assumed fixed in the world, so its spatial velocity is zero:

            Jc_base @ v_base + Jc_rest @ qd_rest = 0

        where `qd_rest` holds the base angular velocity (the gyroscope) and the joint
        velocities, both already measured. Only the three base linear DoFs are unknown, and
        `Jc_base` — the translation block of the support Jacobian — is square, so this
        inverts directly. Same computation as ModelService::estimate_body_vel.

        Returns the velocity in the trunk frame, and writes it back into `state.qd` so the
        model carries a consistent full velocity vector.
        """
        # Translation rows of the support Jacobian, in the support's own frame.
        Jc = self.robot.frame_jacobian(self._support_frame, "local")[:3, :]
        Jc_base = Jc[:, :3]
        Jc_rest = Jc[:, 3:]

        body_velocity = -np.linalg.solve(Jc_base, Jc_rest @ self.robot.state.qd[3:])
        self.robot.state.qd[:3] = body_velocity
        return body_velocity

    def _update_support(self) -> None:
        """Hand the anchor over to the lowest sole corner, if one dropped below it.

        Strictly lower, so a foot resting flat keeps the corner it already had rather than
        flickering between four coplanar candidates.
        """
        lowest_z = float(self.robot.get_T_world_frame(self._support_frame)[2, 3])
        lowest_frame: str | None = None

        for frame in CONTACT_FRAMES:
            z = float(self.robot.get_T_world_frame(frame)[2, 3])
            if z < lowest_z:
                lowest_z = z
                lowest_frame = frame

        if lowest_frame is not None:
            side = "right" if lowest_frame.startswith("right") else "left"
            # placo re-plants the anchor flat on the floor, where forward kinematics says
            # this frame currently is — that transfer is the odometry step.
            self.robot.update_support_side_with_frame(side, lowest_frame)
            self._support_frame = lowest_frame
            self._support_side = side


@dataclass
class OdometrySample:
    """One tick of estimated state."""

    t: float
    T_world_trunk: np.ndarray  # 4x4
    body_velocity: np.ndarray  # trunk-frame linear velocity (m/s)
    world_velocity: np.ndarray  # world-frame linear velocity (m/s)
    omega_trunk: np.ndarray  # trunk-frame angular velocity (rad/s): roll, pitch, yaw rates
    support_frame: str
    support_side: str
    joints: dict[str, float] = field(default_factory=dict)

    @property
    def position(self) -> np.ndarray:
        return self.T_world_trunk[:3, 3]

    @property
    def yaw(self) -> float:
        return yaw_of(self.T_world_trunk[:3, :3])


def _valid_samples(times: list, values: list) -> tuple[np.ndarray, np.ndarray]:
    """The (t, v) pairs of a log channel, with the dropped reads (nulls) removed."""
    pairs = [(t, float(v)) for t, v in zip(times, values) if v is not None]
    if not pairs:
        return np.empty(0), np.empty(0)
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def _resample(grid: np.ndarray, times: list, values: list, default: float = 0.0) -> np.ndarray:
    """A log channel linearly interpolated onto `grid`.

    Dropped reads are interpolated straight across rather than held, which is also what
    makes gaps harmless. Outside the recorded span `np.interp` holds the end values, so a
    channel that starts or ends short does not swing off to zero.
    """
    t, v = _valid_samples(times, values)
    if t.size == 0:
        return np.full(grid.shape, default)
    if t.size == 1:
        return np.full(grid.shape, v[0])
    return np.interp(grid, t, v)


def _resample_quat(grid: np.ndarray, times: list, channel: dict) -> np.ndarray:
    """The `body_quat` channel interpolated onto `grid`, as normalised (w, x, y, z) rows.

    Component-wise interpolation followed by renormalisation (nlerp). Between two ticks
    20 ms apart the rotation is small enough that this is indistinguishable from slerp, and
    it stays well away from slerp's near-antipodal blowup.

    The sign is made continuous first: q and -q are the same rotation, so a log is free to
    flip between them from one tick to the next, and interpolating across a flip would
    sweep the quaternion through zero and out the other side.
    """
    rows = [
        (t, [float(channel[axis][i]) for axis in "wxyz"])
        for i, t in enumerate(times)
        if all(channel[axis][i] is not None for axis in "wxyz")
    ]
    if not rows:
        raise ValueError("Log has no body_quat channel — cannot estimate orientation.")

    t = np.array([r[0] for r in rows])
    q = np.array([r[1] for r in rows])
    for i in range(1, len(q)):
        if float(q[i] @ q[i - 1]) < 0.0:
            q[i] = -q[i]

    if t.size == 1:
        out = np.repeat(q, grid.size, axis=0)
    else:
        out = np.stack([np.interp(grid, t, q[:, k]) for k in range(4)], axis=1)

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0.0, 1.0, norms)


def replay_log(
    log: dict, model_path: str = DEFAULT_MODEL_PATH, dt: float = ODOMETRY_DT
) -> list[OdometrySample]:
    """Run a recorded session log (see src/robot_logger.py) through the odometry.

    Uses the *read* joint positions and velocities, and the `body_quat` / `gyro` channels
    — the same signals the robot estimates from online. Raises if the log predates
    `body_quat`, since without an attitude there is no odometry to speak of.

    Every channel is linearly interpolated onto a uniform `dt` grid first (see ODOMETRY_DT
    for why). Interpolation invents no information about the *robot* — the joint traces
    between two log ticks are straight lines, not the truth — but it does let the anchor
    switch be noticed at a `dt` resolution instead of a 20 ms one, and it is the transfer
    instant, not the joint detail, that the integrated position is sensitive to.

    The returned samples are therefore on the `dt` grid, not on the log's own ticks.
    """
    times = log.get("time") or []
    if not times:
        raise ValueError("Log has no samples.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")

    position = log.get("position", {})
    velocity = log.get("velocity", {})
    body_quat = log.get("body_quat") or {}
    gyro = log.get("gyro") or {}

    if not all(axis in body_quat for axis in "wxyz"):
        raise ValueError("Log has no body_quat channel — cannot estimate orientation.")

    # Uniform grid spanning the log, built by index so accumulated float error cannot make
    # the step drift or the last point overshoot the recorded span.
    span = float(times[-1]) - float(times[0])
    grid = float(times[0]) + dt * np.arange(int(round(span / dt)) + 1)

    names = [name for name in MOTOR_TO_ID if name in position]
    joint_grid = {
        name: _resample(grid, times, position[name], NEUTRAL_POSE.get(name, 0.0))
        for name in names
    }
    # A dropped velocity read means "unknown"; zero is the least harmful stand-in for a
    # channel that was never recorded at all.
    velocity_grid = {
        name: _resample(grid, times, velocity.get(name, [None] * len(times))) for name in names
    }
    quat_grid = _resample_quat(grid, times, body_quat)
    omega_grid = np.stack(
        [_resample(grid, times, gyro.get(axis, [None] * len(times))) for axis in "xyz"], axis=1
    )

    odometry = Odometry(model_path)
    odometry.reset(
        {name: float(joint_grid[name][0]) for name in names}, quat_to_matrix(quat_grid[0])
    )

    samples: list[OdometrySample] = []
    for i, t in enumerate(grid):
        joints = {name: float(joint_grid[name][i]) for name in names}
        velocities = {name: float(velocity_grid[name][i]) for name in names}

        T = odometry.update(
            joints, velocities, quat_to_matrix(quat_grid[i]), gyro_to_trunk(omega_grid[i])
        ).copy()
        samples.append(
            OdometrySample(
                t=float(t),
                T_world_trunk=T,
                body_velocity=odometry.body_velocity.copy(),
                world_velocity=odometry.world_velocity,
                omega_trunk=odometry.omega_trunk,
                support_frame=odometry.support_frame,
                support_side=odometry.support_side,
                joints=joints,
            )
        )

    return samples
