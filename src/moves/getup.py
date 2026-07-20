# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import onnxruntime as ort

from constants import MOTOR_TO_ID, KP_DEFAULT, KP_RL
from controller import ControllerProtocol
from observer import Observation
from moves.move import MotorCommand, Move, MoveState

# Policy name
AGENT_NAME = "getup.onnx"


class GetupMove(Move):
    """Get back on its feet from any fallen pose, using an RL policy trained in simulation.

    Environment: mjlab_microban/tasks/microban_getup_env_cfg.py, a fall-recovery task
    ported from mjlab_playground. Observation layout is
    gyro(3) + projected gravity(3) + joint_pos(19) + joint_vel(19) + last action(19) = 63,
    and there are 19 actions, one per joint including the head.

    Two things differ from the squat-derived policies and from this move's own earlier
    version, and both matter for deployment:

    - There is no command channel. The previous getup env reused the squat's trunk-height
      command pinned to a constant; this one has ``commands={}``. Difficulty comes from the
      reset (dropped from height at a random orientation) and an energy-termination
      curriculum instead, so there is nothing left to feed the network beyond state.

    - The action term is RELATIVE, not an offset from the reference pose. Training used
      SettleRelativeJointPositionActionCfg with scale 0.6, which drives
      ``target = measured joint position + action * scale`` every step. Adding the action
      to the default pose instead — what an absolute policy wants — would make the targets
      mean something entirely different to the robot.

    The joint set and its ordering come from the ONNX metadata rather than
    OBSERVATION_DOF_ORDER: that constant lists 18 joints in a different order and omits the
    head, which this policy both observes and drives.

    Training started the robot dropped at a random orientation, so the policy expects to be
    handed an arbitrary pose. It has no notion of being finished: once upright it keeps
    balancing, so the move stays ACTIVE until untoggled rather than stopping itself.
    """

    is_policy = True

    def __init__(self, controller: ControllerProtocol | None = None) -> None:
        super().__init__()
        self._controller = controller

        # Load ONNX policy
        self._ort_session = ort.InferenceSession(f"src/agents/{AGENT_NAME}")

        # Joint order, reference pose and action scale: read from ONNX metadata, so the
        # deployed layout tracks whatever the export produced instead of a hand-kept copy.
        meta = self._ort_session.get_modelmeta().custom_metadata_map
        self._joint_names = meta["joint_names"].split(",")
        positions = [float(v) for v in meta["default_joint_pos"].split(",")]
        self._default_pose: dict[str, float] = dict(zip(self._joint_names, positions))
        self.action_scale = float(meta["action_scale"])

        self._last_action = [0.0] * len(self._joint_names)

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        if self._controller is not None:
            ids = list(MOTOR_TO_ID.values())
            self._controller.sync_write_kp(ids, [KP_RL] * len(ids))
        self._last_action = [0.0] * len(self._joint_names)
        # Straight to ACTIVE, with no ramp to a start pose: the robot is on the ground and
        # lerping it through neutral on the way in would drive limbs into the floor.
        #
        # The env's 25-step settle window is not replayed here either. It exists so a robot
        # dropped in simulation lands before the policy takes over; a real robot that has
        # already fallen is settled by the time anyone toggles this move.
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

        # Update command. Relative action: the target is an increment on the position the
        # joint is measured at now, so each tick moves from wherever the robot actually is.
        for i, name in enumerate(self._joint_names):
            measured = obs.robot_state.motor_positions[name]
            command.target_angles[name] = measured + action[i] * self.action_scale

    def build_observation(self, obs: Observation) -> list[float]:
        """Build policy observation from robot state."""
        input_obs = []

        # IMU data: gyroscope and projected gravity in body frame
        input_obs.extend(obs.robot_state.gyro)
        input_obs.extend(obs.robot_state.projected_gravity)

        # Motor positions, relative to the reference pose
        for name in self._joint_names:
            input_obs.append(obs.robot_state.motor_positions[name] - self._default_pose[name])

        # Motor velocities
        for name in self._joint_names:
            input_obs.append(obs.robot_state.motor_velocities[name])

        # Last action
        input_obs.extend(self._last_action)

        return input_obs

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        if self._controller is not None:
            ids = list(MOTOR_TO_ID.values())
            self._controller.sync_write_kp(ids, [KP_DEFAULT] * len(ids))
        self.state = MoveState.INACTIVE
