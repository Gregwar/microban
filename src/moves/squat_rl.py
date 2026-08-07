# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import math

import onnxruntime as ort

from constants import MOTOR_TO_ID, KP_DEFAULT, KP_RL, OBSERVATION_DOF_ORDER
from controller import ControllerProtocol
from observer import Observation
from moves.move import MotorCommand, Move, MoveState, onnx_run_name

# Policy name
AGENT_NAME = "squat.onnx"

# Squat trajectory. The policy tracks a trunk-height target; this move plays that target
# as the same sine the training command did (mjlab SquatHeightCommand):
#
#     z(t) = SQUAT_CENTER_HEIGHT + SQUAT_AMPLITUDE * sin(2*pi*SQUAT_FREQUENCY*t)
#
# Tune them here. The policy was trained with center sampled in [0.12, 0.16] m, amplitude
# in [0.0, 0.05] m and frequency in [0.2, 0.8] Hz — outside those ranges you are asking it
# for a target it never saw during training.
SQUAT_CENTER_HEIGHT = 0.14  # metres, mean trunk height the sine oscillates around
SQUAT_AMPLITUDE = 0.015      # metres, half the peak-to-peak squat depth
SQUAT_FREQUENCY = 0.25       # Hz


class SquatRlMove(Move):
    """Squat using an RL policy trained in simulation.

    Environment: mjlab_microban/tasks/microban_squat_env_cfg.py. The observation is
    gyro(3) + projected gravity(3) + joint_pos(18) + joint_vel(18) + last action(18) +
    height target(1) = 61, and the 18 actions are joint offsets from the reference pose
    for every joint except the head — which this move therefore leaves at neutral.
    """

    is_policy = True

    def __init__(self, controller: ControllerProtocol | None = None) -> None:
        super().__init__()
        self._controller = controller
        self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)

        # Load ONNX policy
        self._ort_session = ort.InferenceSession(f"src/agents/{AGENT_NAME}")

        self.action_scale = 1.0

        # Reference pose: read from ONNX metadata
        meta = self._ort_session.get_modelmeta().custom_metadata_map
        names = meta["joint_names"].split(",")
        positions = [float(v) for v in meta["default_joint_pos"].split(",")]
        self._default_pose: dict[str, float] = dict(zip(names, positions))

        # Sine clock, restarted at each activation so the squat always begins at the
        # centre height instead of jumping to wherever a free-running phase happens to be.
        self._start_time_s = 0.0

        # Last height target handed to the policy, picked up by the scheduler for the log
        # (see Move.height_target_command). None on any tick the policy did not run, so the
        # log never shows a commanded height the policy was not actually given.
        self.height_target_command: float | None = None

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
        self.height_target_command = None
        self.state = MoveState.ACTIVE

    def height_target(self, obs: Observation) -> float:
        """Commanded trunk height [m] for this tick, sampled from the sine."""
        elapsed_s = obs.robot_state.time_s - self._start_time_s
        return SQUAT_CENTER_HEIGHT + SQUAT_AMPLITUDE * math.sin(
            2.0 * math.pi * SQUAT_FREQUENCY * elapsed_s
        )

    def step(self, obs: Observation, command: MotorCommand) -> None:
        # Safety check: if the robot is fallen, stop the policy
        if obs.robot_state.projected_gravity[2] > self._projected_gravity_z_threshold:
            self.height_target_command = None
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

        # Command: the trunk height target the policy tracks. Kept on the move as well, so
        # the scheduler logs exactly the value that went into the observation.
        self.height_target_command = self.height_target(obs)
        input_obs.append(self.height_target_command)

        return input_obs

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        if self._controller is not None:
            ids = list(MOTOR_TO_ID.values())
            self._controller.sync_write_kp(ids, [KP_DEFAULT] * len(ids))
        # Cleared here too, so the log stops showing the last commanded height once the
        # policy is no longer the one driving the robot.
        self.height_target_command = None
        self.state = MoveState.INACTIVE
