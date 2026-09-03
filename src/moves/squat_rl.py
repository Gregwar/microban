# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import math
from pathlib import Path

import onnxruntime as ort

from constants import MOTOR_TO_ID, KP_DEFAULT, KP_RL, OBSERVATION_DOF_ORDER
from controller import ControllerProtocol
from observer import Observation
from moves.move import MotorCommand, Move, MoveState, onnx_run_name

# Policy name
AGENT_NAME = "squat_m1_45.onnx"

# Squat clock. This policy does not track a trunk height: it tracks a placo-generated
# squat reference — the same motion src/moves/squat.py runs — and the only thing it is
# told is where it is in that cycle:
#
#     phase = 2*pi*SQUAT_FREQUENCY*t,   command = [cos(phase), sin(phase)]
#
# The depth and the posture of the squat therefore come from the reference the policy
# learned, not from anything set here; the frequency only says how fast to play it, and
# it MUST match the clip the policy was trained against (one full period sampled at the
# 50 Hz control rate, mjlab_microban/scripts/make_squat_reference.py FREQUENCY = 0.25 Hz).
# Phase 0 is the top of the squat, where the reference sits at the home pose — which is
# why the clock restarts at every activation instead of free-running.
SQUAT_FREQUENCY = 0.25  # Hz

# gyro(3) + projected gravity(3) + joint pos/vel/last action(3 * 18) + phase(2).
_OBSERVATION_SIZE = 3 + 3 + 3 * len(OBSERVATION_DOF_ORDER) + 2


class SquatRlMove(Move):
    """Squat using an RL policy trained in simulation.

    Environment: mjlab_microban/tasks/microban_squatref_env_cfg.py
    (``Mjlab-SquatRef-Microban``). The observation is gyro(3) + projected gravity(3) +
    joint_pos(18) + joint_vel(18) + last action(18) + phase(2) = 62, and the 18 actions
    are joint offsets from the reference pose for every joint except the head — which
    this move therefore leaves at neutral.

    Note the observation is *not* the one the older ``Mjlab-Squat-Microban`` policies
    take (61, ending in a trunk-height target): a policy trained on that task cannot be
    driven by this move, and the size check in ``__init__`` refuses it rather than
    letting it run on a silently wrong last observation.
    """

    is_policy = True

    def __init__(self, controller: ControllerProtocol | None = None) -> None:
        super().__init__()
        self._controller = controller
        self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)

        # Load ONNX policy
        agent_path = Path(f"src/agents/{AGENT_NAME}")
        if not agent_path.exists():
            raise FileNotFoundError(
                f"{agent_path} not found. Export one from the squat reference task:\n"
                "  cd ~/mjlab_microban && uv run python src/mjlab_microban/scripts/export_onnx.py "
                "--checkpoint logs/rsl_rl/mjlab_microban_squatref/<run>/model_<n>.pt\n"
                "(the script's TASK must be Mjlab-SquatRef-Microban)"
            )
        self._ort_session = ort.InferenceSession(str(agent_path))

        # Guard against loading a height-tracking squat policy by mistake: both are
        # called squat_*.onnx, and the only visible difference is the observation size.
        input_size = self._ort_session.get_inputs()[0].shape[1]
        if input_size != _OBSERVATION_SIZE:
            raise ValueError(
                f"{AGENT_NAME} takes {input_size} observations but this move builds "
                f"{_OBSERVATION_SIZE}: it is not a Mjlab-SquatRef-Microban policy "
                "(a height-tracking Mjlab-Squat-Microban one takes 61)."
            )

        self.action_scale = 1.0

        # Reference pose: read from ONNX metadata
        meta = self._ort_session.get_modelmeta().custom_metadata_map
        names = meta["joint_names"].split(",")
        positions = [float(v) for v in meta["default_joint_pos"].split(",")]
        self._default_pose: dict[str, float] = dict(zip(names, positions))

        # Phase clock, restarted at each activation so the squat always begins at the top
        # of the cycle instead of dropping into wherever a free-running phase happens to be.
        self._start_time_s = 0.0

        # Number of full squat cycles completed since activation, used only to report
        # progress: the phase clock is what drives the policy.
        self._squat_count = 0

        # Safety parameters
        self._projected_gravity_z_threshold = -0.5  # Threshold for detecting a fall based on projected gravity

        # NOTE: Move.height_target_command stays None for this move. The policy is given a
        # phase, not a height, so there is no commanded trunk height to log — the log's
        # `command.height` channel is simply null while this squat runs (the reference's
        # own trunk height lives in the training-side clip, not here).

    def describe(self) -> str:
        return onnx_run_name(self._ort_session, AGENT_NAME)

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        if self._controller is not None:
            ids = list(MOTOR_TO_ID.values())
            self._controller.sync_write_kp(ids, [KP_RL] * len(ids))
        self._start_time_s = obs.robot_state.time_s
        self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
        self._squat_count = 0
        self.state = MoveState.ACTIVE

    def phase(self, obs: Observation) -> float:
        """Position in the squat cycle [rad] for this tick, 0 at the top of the squat.

        Read off the clock rather than counted in ticks (as the training env does, one
        reference row per control step): the loop is real-time, so timing the phase keeps
        the squat at exactly SQUAT_FREQUENCY even when a tick runs late.
        """
        elapsed_s = obs.robot_state.time_s - self._start_time_s
        return 2.0 * math.pi * SQUAT_FREQUENCY * elapsed_s

    def step(self, obs: Observation, command: MotorCommand) -> None:
        # Safety check: if the robot is fallen, stop the policy
        if obs.robot_state.projected_gravity[2] > self._projected_gravity_z_threshold:
            return

        # Report each completed squat: the phase restarts at 0 at the top of the cycle,
        # so a full squat is done every time it passes another multiple of 2*pi.
        completed = int(self.phase(obs) / (2.0 * math.pi))
        if completed > self._squat_count:
            self._squat_count = completed
            print(f"{self._squat_count} squat{'s' if self._squat_count > 1 else ''} done\r")

        # Run policy
        input_obs = self.build_observation(obs)
        ort_inputs = {self._ort_session.get_inputs()[0].name: [input_obs]}
        ort_outs = self._ort_session.run(None, ort_inputs)
        action = ort_outs[0][0]
        self._last_action = action.tolist()

        # Update command
        for i, name in enumerate(OBSERVATION_DOF_ORDER):
            command.target_angles[name] = self._default_pose[name] + action[i] * self.action_scale

    def build_observation(self, obs: Observation) -> list[float]:
        """Build policy observation from robot state."""
        input_obs = []

        # IMU data: gyroscope and projected gravity in body frame
        input_obs.extend(obs.robot_state.gyro)
        input_obs.extend(obs.robot_state.projected_gravity)

        # Motor positions
        for name in OBSERVATION_DOF_ORDER:
            input_obs.append(obs.robot_state.motor_positions[name] - self._default_pose[name])

        # Motor velocities
        for name in OBSERVATION_DOF_ORDER:
            input_obs.append(obs.robot_state.motor_velocities[name])

        # Last action
        input_obs.extend(self._last_action)

        # Command: where in the squat cycle the policy is, as [cos, sin] so the clock has
        # no discontinuity when it wraps.
        phase = self.phase(obs)
        input_obs.append(math.cos(phase))
        input_obs.append(math.sin(phase))

        return input_obs

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        if self._controller is not None:
            ids = list(MOTOR_TO_ID.values())
            self._controller.sync_write_kp(ids, [KP_DEFAULT] * len(ids))
        self.state = MoveState.INACTIVE
