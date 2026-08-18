# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import onnxruntime as ort
import numpy as np

from constants import MOTOR_TO_ID, KP_DEFAULT, KP_RL, OBSERVATION_DOF_ORDER
from controller import ControllerProtocol
from observer import Observation
from moves.move import MotorCommand, Move, MoveState, onnx_run_name


# Set to True to log motor positions and voltages during the walk move
# Note: requires to set observe_voltage = True in the Observer to log voltages
LOGGING = False

# Set to True to run the policy on a left/right mirrored robot: the observation is
# reflected across the sagittal plane before the policy sees it, and the resulting
# action is reflected back onto the real robot. On a perfectly symmetric robot this
# produces the exact mirror-image gait. Use it as a sim2real check: if the robot keeps
# drifting the SAME physical direction with mirror on, the bias is in the hardware;
# if the drift flips direction, the bias is in the policy.
MIRROR = False

# Policy name
AGENT_NAME = "walk_m6_l1.onnx"


class WalkMove(Move):
    """Walk using a RL policy trained in simulation."""

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

        # Detect reference phase from model input size:
        # base_obs = gyro(3) + proj_grav(3) + pos(N) + vel(N) + action(N) + cmd(3)
        # phase_obs = base_obs + phase(2)
        base_obs_size = 3 + 3 + 3 * len(OBSERVATION_DOF_ORDER) + 3
        self._use_reference_phase: bool = self._ort_session.get_inputs()[0].shape[1] > base_obs_size
        self._phase_step = 0
        self._phase_total_steps = 20

        # Mirror (sagittal reflection) support: precompute, for each joint in
        # OBSERVATION_DOF_ORDER, its left/right partner index and the sign it picks up
        # under the reflection (roll/yaw axes flip, pitch axes keep sign). The map is an
        # involution, so the same transform mirrors observations and un-mirrors actions.
        self.mirror: bool = MIRROR
        self._mirror_idx, self._mirror_sign = self._build_mirror_maps()

        # Safety parameters
        self._projected_gravity_z_threshold = -0.5  # Threshold for detecting a fall based on projected gravity

        # Logging
        self.position = {
            "head": [],
            "left_hip_yaw": [],
            "left_hip_roll": [],
            "left_hip_pitch": [],
            "left_knee": [],
            "left_ankle_pitch": [],
            "left_ankle_roll": [],
            "right_hip_yaw": [],
            "right_hip_roll": [],
            "right_hip_pitch": [],
            "right_knee": [],
            "right_ankle_pitch": [],
            "right_ankle_roll": [],
            "left_shoulder_pitch": [],
            "left_shoulder_roll": [],
            "left_elbow": [],
            "right_shoulder_pitch": [],
            "right_shoulder_roll": [],
            "right_elbow": [],
        }
        self.voltage = {
            "head": [],
            "left_hip_yaw": [],
            "left_hip_roll": [],
            "left_hip_pitch": [],
            "left_knee": [],
            "left_ankle_pitch": [],
            "left_ankle_roll": [],
            "right_hip_yaw": [],
            "right_hip_roll": [],
            "right_hip_pitch": [],
            "right_knee": [],
            "right_ankle_pitch": [],
            "right_ankle_roll": [],
            "left_shoulder_pitch": [],
            "left_shoulder_roll": [],
            "left_elbow": [],
            "right_shoulder_pitch": [],
            "right_shoulder_roll": [],
            "right_elbow": [],
        }
        
    def describe(self) -> str:
        return onnx_run_name(self._ort_session, AGENT_NAME)

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        if self._controller is not None:
            ids = list(MOTOR_TO_ID.values())
            self._controller.sync_write_kp(ids, [KP_RL] * len(ids))
        self.state = MoveState.ACTIVE

    def step(self, obs: Observation, command: MotorCommand) -> None:
        # Update reference phase
        if self._use_reference_phase:
            commanded_vel = np.mean([np.abs(obs.user_input.velocity["vx"]), np.abs(obs.user_input.velocity["vy"]), np.abs(obs.user_input.velocity["vtheta"])])
            if commanded_vel > 0.01:
                self._phase_step += 1
            else:
                self._phase_step = 0

        # Safety check: if the robot is fallen, stop the policy
        if obs.robot_state.projected_gravity[2] > self._projected_gravity_z_threshold:
            return
        
        # Run policy
        input_obs = self.build_observation(obs)
        ort_inputs = {self._ort_session.get_inputs()[0].name: [input_obs]}
        ort_outs = self._ort_session.run(None, ort_inputs)
        action = ort_outs[0][0]
        # Store the raw policy output as the "last action": it stays in the policy's
        # (mirrored, when self.mirror) frame, matching what build_observation feeds back.
        self._last_action = action.tolist()

        # In mirror mode the action is expressed for the mirrored robot; reflect it back
        # onto the real robot before applying it (no-op when self.mirror is False).
        applied = self._mirror_joints(self._last_action) if self.mirror else self._last_action

        # Update command
        for i, name in enumerate(OBSERVATION_DOF_ORDER):
            command.target_angles[name] = self._default_pose[name] + applied[i] * self.action_scale

        # Log positions and voltages
        if LOGGING:
            for name in MOTOR_TO_ID.keys():
                self.position[name].append(obs.robot_state.motor_positions[name])
                self.voltage[name].append(obs.robot_state.motor_voltages[name])

    def _build_mirror_maps(self) -> tuple[list[int], list[float]]:
        """Precompute the sagittal-reflection map over OBSERVATION_DOF_ORDER.

        Returns (idx, sign) where the reflected value of joint i is
        sign[i] * value[idx[i]]: idx[i] is the index of joint i's left/right partner,
        and sign[i] is -1 for roll/yaw axes (which flip under a left/right mirror) and
        +1 for pitch axes. The map is an involution.
        """
        # Base joint names (without the left_/right_ prefix) whose axis flips sign.
        antisymmetric = {"shoulder_roll", "hip_yaw", "hip_roll", "ankle_roll"}
        idx: list[int] = []
        sign: list[float] = []
        for name in OBSERVATION_DOF_ORDER:
            if name.startswith("left_"):
                partner = "right_" + name[len("left_"):]
                base = name[len("left_"):]
            elif name.startswith("right_"):
                partner = "left_" + name[len("right_"):]
                base = name[len("right_"):]
            else:
                partner = name  # central joint (e.g. head) maps to itself
                base = name
            idx.append(OBSERVATION_DOF_ORDER.index(partner))
            sign.append(-1.0 if base in antisymmetric else 1.0)
        return idx, sign

    def _mirror_joints(self, vec: list[float]) -> list[float]:
        """Reflect a per-joint vector (ordered like OBSERVATION_DOF_ORDER) across the
        sagittal plane by swapping left/right joints and flipping roll/yaw signs."""
        return [self._mirror_sign[i] * vec[self._mirror_idx[i]] for i in range(len(vec))]

    def build_observation(self, obs: Observation) -> list[float]:
        """Build policy observation from robot state."""
        mirror = self.mirror

        # IMU data: gyroscope (raw IMU sensor frame) and projected gravity (body frame).
        gyro = list(obs.robot_state.gyro)
        projected_gravity = list(obs.robot_state.projected_gravity)
        if mirror:
            # Sagittal mirror. The gyro is in the rotated IMU frame (IMU_MOUNT_QUAT), where
            # gyro = (body pitch, -body yaw, -body roll); a left/right mirror flips body
            # roll and yaw, which flips gyro_y and gyro_z (gyro_x = pitch is unchanged).
            gyro = [gyro[0], -gyro[1], -gyro[2]]
            # Projected gravity is a true vector in body frame (x fwd, y left, z up): flip y.
            projected_gravity = [projected_gravity[0], -projected_gravity[1], projected_gravity[2]]

        # Motor positions (as offsets from the default pose) and velocities.
        pos_obs = [obs.robot_state.motor_positions[name] - self._default_pose[name] for name in OBSERVATION_DOF_ORDER]
        vel_obs = [obs.robot_state.motor_velocities[name] for name in OBSERVATION_DOF_ORDER]
        if mirror:
            pos_obs = self._mirror_joints(pos_obs)
            vel_obs = self._mirror_joints(vel_obs)

        # Command
        vx = obs.user_input.velocity["vx"]
        vy = obs.user_input.velocity["vy"]
        vtheta = obs.user_input.velocity["vtheta"]
        if mirror:
            # Lateral velocity and yaw-rate commands flip under a left/right mirror.
            vy = -vy
            vtheta = -vtheta

        input_obs: list[float] = []
        input_obs.extend(gyro)
        input_obs.extend(projected_gravity)
        input_obs.extend(pos_obs)
        input_obs.extend(vel_obs)
        # Last action: already stored in the policy (mirrored) frame, so feed it as-is.
        input_obs.extend(self._last_action)
        input_obs.extend([vx, vy, vtheta])

        # Reference phase
        if self._use_reference_phase:
            reference_phase = (self._phase_step % self._phase_total_steps) / self._phase_total_steps * 2 * np.pi
            if mirror:
                # Mirroring swaps the stance/swing legs, i.e. a half-cycle phase shift.
                reference_phase += np.pi
            input_obs.append(np.cos(reference_phase))
            input_obs.append(np.sin(reference_phase))

        return input_obs

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        if self._controller is not None:
            ids = list(MOTOR_TO_ID.values())
            self._controller.sync_write_kp(ids, [KP_DEFAULT] * len(ids))
        self.state = MoveState.INACTIVE

        # Save json logs
        if LOGGING:
            import json
            with open("walk_log.json", "w") as f:
                json.dump({
                    "position": self.position,
                    "voltage": self.voltage,
                }, f, indent=4)