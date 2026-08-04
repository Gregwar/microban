# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import onnxruntime as ort

from constants import MOTOR_TO_ID, KP_DEFAULT, KP_RL, OBSERVATION_DOF_ORDER
from controller import ControllerProtocol
from observer import Observation
from moves.move import MotorCommand, Move, MoveState, onnx_run_name

# Policy name
AGENT_NAME = "getup.onnx"

# The policy tracks a trunk-height target, the same command term the squat policy uses.
# Getting up is that command pinned high and held: the training config sampled it from
# (0.165, 0.165) with zero amplitude and zero frequency, so unlike the squat there is no
# trajectory to play — just a constant "stand up" target.
#
# This one is not a tuning knob. The command channel was constant throughout training, so
# its observation normalizer has a degenerate standard deviation (the variance floor).
# Any departure from the trained value is divided by that floor and reaches the network
# as a wildly out-of-distribution input.
STAND_HEIGHT = 0.165  # metres, target trunk height


class GetupMove(Move):
    """Get back on its feet from any fallen pose, using an RL policy trained in simulation.

    Environment: mjlab_microban/tasks/microban_getup_env_cfg.py. The observation layout is
    the squat one — gyro(3) + projected gravity(3) + joint_pos(18) + joint_vel(18) +
    last action(18) + height target(1) = 61 — and the 18 actions are joint offsets from the
    reference pose for every joint except the head, which this move leaves at neutral.

    Training started the robot prone, supine or standing, at a random yaw and with the
    trunk on the floor, so the policy expects to be handed an arbitrary orientation. It has
    no notion of being finished: once upright it keeps balancing, so the move stays ACTIVE
    until untoggled rather than stopping itself.
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

    def describe(self) -> str:
        return onnx_run_name(self._ort_session, AGENT_NAME)

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        if self._controller is not None:
            ids = list(MOTOR_TO_ID.values())
            self._controller.sync_write_kp(ids, [KP_RL] * len(ids))
        self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
        # Straight to ACTIVE, with no ramp to a start pose: the robot is on the ground and
        # lerping it through neutral on the way in would drive limbs into the floor.
        self.state = MoveState.ACTIVE

    def step(self, obs: Observation, command: MotorCommand) -> None:
        # Deliberately no fall check here. The policies that walk or squat bail out when
        # projected gravity says the robot is on its side, because for them that is a
        # failure; this one is the recovery from it and runs precisely in that state.
        #
        # What is worth refusing is a bad IMU read, which the observer reports as empty
        # lists. Building the observation from those would hand the network a short vector
        # and raise; holding the previous targets for a tick is the safer response, and
        # orientation is the one input a getup policy cannot do without.
        if len(obs.robot_state.gyro) != 3 or len(obs.robot_state.projected_gravity) != 3:
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

        # Command: the trunk height target, held at the standing value
        input_obs.append(STAND_HEIGHT)

        return input_obs

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        if self._controller is not None:
            ids = list(MOTOR_TO_ID.values())
            self._controller.sync_write_kp(ids, [KP_DEFAULT] * len(ids))
        self.state = MoveState.INACTIVE
