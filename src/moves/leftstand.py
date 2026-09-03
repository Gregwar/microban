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
AGENT_NAME = "leftstand_25.onnx"

# Single-leg clock. Like the squat reference policy, this one is not given a pose target
# but a position in a cycle:
#
#     phase = 2*pi*t/LEFTSTAND_CYCLE_S,   command = [cos(phase), sin(phase)]
#
# so the move loops — lift the right foot, stand on the left, put it back down, again —
# for as long as it is active, instead of rising once and holding there. What the robot
# does at each point of that cycle comes from the reference the policy was trained on;
# the period here only says how fast to play it, and it MUST match the length of that
# reference clip (6 s, i.e. 300 frames at the 50 Hz control rate). Phase 0 is the first
# frame of the clip, and the clock restarts at every activation so the loop always begins
# there rather than dropping in wherever a free-running phase happens to be.
LEFTSTAND_CYCLE_S = 6.0  # seconds per lift/lower cycle

# gyro(3) + projected gravity(3) + joint pos/vel/last action(3 * 18) + phase(2).
_OBSERVATION_SIZE = 3 + 3 + 3 * len(OBSERVATION_DOF_ORDER) + 2


class LeftStandMove(Move):
    """Balance on the left foot using an RL policy trained in simulation.

    Environment: the single-leg reference task of mjlab_microban (the policy's ONNX
    metadata names its command ``leftstand_ref``). The robot cycles between double
    support and standing on its *left* foot with the right one lifted; the observation is
    gyro(3) + projected gravity(3) + joint_pos(18) + joint_vel(18) + last action(18) +
    phase(2) = 62, and the 18 actions are joint offsets from the reference pose for every
    joint except the head — which this move therefore leaves at neutral.

    The policy commands no trunk height (its whole command is the phase), so
    Move.height_target_command stays None and the log's `command.height` channel is null
    while it runs.
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
                f"{agent_path} not found. Export one from the single-leg reference task:\n"
                "  cd ~/mjlab_microban && uv run python src/mjlab_microban/scripts/export_onnx.py "
                "--checkpoint logs/rsl_rl/<leftstand run>/model_<n>.pt"
            )
        self._ort_session = ort.InferenceSession(str(agent_path))

        input_size = self._ort_session.get_inputs()[0].shape[1]
        if input_size != _OBSERVATION_SIZE:
            raise ValueError(
                f"{AGENT_NAME} takes {input_size} observations but this move builds "
                f"{_OBSERVATION_SIZE}: it is not a phase-driven leftstand policy (the "
                "older two-height Mjlab-LeftStandFoot-Microban one also takes 62 but "
                "reads its command as [foot height, trunk height])."
            )

        self.action_scale = 1.0

        # Reference pose: read from ONNX metadata
        meta = self._ort_session.get_modelmeta().custom_metadata_map
        names = meta["joint_names"].split(",")
        positions = [float(v) for v in meta["default_joint_pos"].split(",")]
        self._default_pose: dict[str, float] = dict(zip(names, positions))

        # Phase clock, restarted at each activation (see LEFTSTAND_CYCLE_S).
        self._start_time_s = 0.0

        # Safety parameters
        self._projected_gravity_z_threshold = -0.5  # Threshold for detecting a fall based on projected gravity

    def describe(self) -> str:
        return onnx_run_name(self._ort_session, AGENT_NAME)

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        if self._controller is not None:
            ids = list(MOTOR_TO_ID.values())
            self._controller.sync_write_kp(ids, [KP_RL] * len(ids))
        self._start_time_s = obs.robot_state.time_s
        self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
        self.state = MoveState.ACTIVE

    def phase(self, obs: Observation) -> float:
        """Position in the lift cycle [rad] for this tick, 0 at the start of the clip.

        Read off the clock rather than counted in ticks (as the training env does, one
        reference row per control step): the loop is real-time, so timing the phase keeps
        the cycle at exactly LEFTSTAND_CYCLE_S even when a tick runs late.
        """
        elapsed_s = obs.robot_state.time_s - self._start_time_s
        return 2.0 * math.pi * elapsed_s / LEFTSTAND_CYCLE_S

    def step(self, obs: Observation, command: MotorCommand) -> None:
        # Safety check: if the robot is fallen, stop the policy
        if obs.robot_state.projected_gravity[2] > self._projected_gravity_z_threshold:
            return

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

        # Command: where in the lift cycle the policy is, as [cos, sin] so the clock has
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
