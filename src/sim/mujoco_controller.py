# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import mujoco.viewer
import numpy as np

if TYPE_CHECKING:
    from sim.debug_keys import SimDebugKeys

from bam.model import load_model as bam_load_model
from bam.mujoco import MujocoController as BamController

from constants import MOTOR_TO_ID, ID_TO_MOTOR, NEUTRAL_POSE, KP_DEFAULT, BAM_VIN, BAM_VOLTAGE_DROP_RESISTANCE, BAM_VIN_MIN, BAM_MAX_CURRENT


# Trunk height to spawn at. In the neutral pose the feet just graze the floor at 0.1727 m,
# so start a hair above that: any lower and the robot spawns interpenetrating the ground and
# the contact solver ejects it.
SPAWN_HEIGHT: float = 0.174

# Window used when velocity is estimated by finite differences instead of read straight
# from the simulator: v = (q(t) - q(t - dt)) / dt. The XL330 does not measure velocity
# directly either, so this models the smoothing and lag of the real read.
VELOCITY_FD_DT: float = 0.050

# Extra solver pass that removes residual tangential drift at contacts. MuJoCo's default
# is 0 (disabled).
NOSLIP_ITERATIONS: int = 1


class _DelayBuffer:
    """Returns values delayed by n_steps ticks (0 = no delay)."""

    def __init__(self, initial, n_steps: int) -> None:
        size = max(1, n_steps + 1)
        self._buf: deque = deque([initial] * size, maxlen=size)

    def push_and_read(self, value):
        self._buf.appendleft(value)
        return self._buf[-1]

    def fill(self, value) -> None:
        for i in range(len(self._buf)):
            self._buf[i] = value


class MuJoCoController:
    """MuJoCo-backed controller."""

    def __init__(
        self,
        mjcf_path: str,
        key_callback: Callable[[int, int, int, int], None] | None = None,
        stop_flag_path: str = "/tmp/microban_scheduler.stop",
        reset_source: "SimDebugKeys | None" = None,
        # Actuation delay (command → motor), in simulator steps (default timestep: 0.005 s)
        delay_act_steps: int = 0,
        # Sensor delays (motor/IMU → observation), in scheduler ticks (default: 0.02 s)
        delay_pos_ticks: int = 0,
        delay_vel_ticks: int = 0,
        delay_gyro_ticks: int = 0,
        delay_quat_ticks: int = 0,
        trunk_com_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        # Estimate joint velocity by finite-differencing position over VELOCITY_FD_DT
        # instead of reading the simulator's qvel.
        velocity_finite_difference: bool = False,
        # Launch the passive MuJoCo viewer. Set False for a headless run (e.g. batch log
        # replay), where there is no window to step in real time and nothing to sync to.
        enable_viewer: bool = True,
    ) -> None:
        self._stop_flag_path = Path(stop_flag_path)
        self._reset_source = reset_source
        self._model = mujoco.MjModel.from_xml_path(mjcf_path)
        self._data = mujoco.MjData(self._model)

        # Noslip is a CPU-only post-pass (MJX has no equivalent), so it lives here rather
        # than in the MJCF, which is shared with the GPU training setup. One iteration is
        # enough to stop the feet creeping under contact without softening the solve.
        self._model.opt.noslip_iterations = NOSLIP_ITERATIONS

        self._name_to_actuator_idx: dict[str, int] = {}
        self._name_to_qpos_idx: dict[str, int] = {}
        self._name_to_qvel_idx: dict[str, int] = {}
        for name in MOTOR_TO_ID:
            actuator_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if actuator_id < 0:
                raise ValueError(f"Actuator '{name}' not found in MJCF model {mjcf_path!r}")
            if joint_id < 0:
                raise ValueError(f"Joint '{name}' not found in MJCF model {mjcf_path!r}")
            self._name_to_actuator_idx[name] = actuator_id
            self._name_to_qpos_idx[name] = self._model.jnt_qposadr[joint_id]
            self._name_to_qvel_idx[name] = self._model.jnt_dofadr[joint_id]
        # Number of physics sub-steps per scheduler tick (scheduler runs at 50 Hz)
        self._steps_per_tick = max(1, round(0.02 / self._model.opt.timestep))
        self._torque_interval = 0.1
        self._last_torque_print = 0.0

        # Apply CoM offset on trunk body (simulates inertial model error)
        if any(v != 0.0 for v in trunk_com_offset):
            trunk_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
            self._model.body_ipos[trunk_id, 0] += trunk_com_offset[0]
            self._model.body_ipos[trunk_id, 1] += trunk_com_offset[1]
            self._model.body_ipos[trunk_id, 2] += trunk_com_offset[2]

        # BAM motor model — XL330 m6 (DC motor + Stribeck + load-dependent friction)
        # Built before the pose is set: its constructor calls mj_setConst(), which resets
        # the data back to qpos0. Set the pose first and it silently lands at the origin,
        # buried in the floor, and the contact solver ejects it on the first step.
        bam_model = bam_load_model(motor_name="xl330", model="m4")
        bam_model.actuator.kp = KP_DEFAULT
        bam_model.actuator.vin = BAM_VIN
        self._bam = BamController(
            bam_model,
            list(MOTOR_TO_ID.keys()),
            self._model,
            self._data,
            vin_drop_resistance=BAM_VOLTAGE_DROP_RESISTANCE,
            vin_min=BAM_VIN_MIN
        )

        # Supply voltage actually seen by the motors, kept in step with the bam model.
        self._vin_eff: float = BAM_VIN

        # Start from the same state [r] resets to, so a fresh run and a reset are identical.
        self._set_neutral_state()

        # Delay buffers — simulate sensor/communication latency. Seeded from the pose above.
        self._delay_pos = {
            mid: _DelayBuffer(
                self._data.qpos[self._name_to_qpos_idx[ID_TO_MOTOR[mid]]],
                delay_pos_ticks,
            )
            for mid in MOTOR_TO_ID.values()
        }
        self._delay_vel = {
            mid: _DelayBuffer(0.0, delay_vel_ticks)
            for mid in MOTOR_TO_ID.values()
        }
        # Position history backing the finite-difference velocity estimate. Sampled once
        # per physics step, so the lookback is VELOCITY_FD_DT rounded to a whole number
        # of steps (exact at the default 2 ms timestep).
        self._velocity_fd = velocity_finite_difference
        self._fd_steps = max(1, round(VELOCITY_FD_DT / self._model.opt.timestep))
        self._fd_dt = self._fd_steps * self._model.opt.timestep
        self._pos_history: deque = deque(maxlen=self._fd_steps + 1)
        self._reset_pos_history()

        self._delay_gyro = _DelayBuffer((0.0, 0.0, 0.0), delay_gyro_ticks)
        self._delay_quat = _DelayBuffer((1.0, 0.0, 0.0, 0.0), delay_quat_ticks)
        self._delay_act = {
            mid: _DelayBuffer(
                self._data.qpos[self._name_to_qpos_idx[ID_TO_MOTOR[mid]]],
                delay_act_steps,
            )
            for mid in MOTOR_TO_ID.values()
        }

        # The left panel is MuJoCo's own simulation options, none of which apply here (the
        # scheduler drives the stepping), and it covers the robot on a small window. Toggle
        # it back at runtime with Tab if needed. Skipped entirely when headless.
        if enable_viewer:
            self._viewer = mujoco.viewer.launch_passive(
                self._model, self._data, key_callback=key_callback, show_left_ui=False
            )
        else:
            self._viewer = None

        # Sensor indices for IMU readout
        self._sensor_orientation = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation")
        self._sensor_gyro = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")

    def _viewer_running(self) -> bool:
        """True while the run should continue. Always True when headless (no window to close)."""
        return self._viewer is None or self._viewer.is_running()

    def is_running(self) -> bool:
        """Whether a run should keep going: the viewer window is open, or always when headless."""
        return self._viewer_running()

    def close(self) -> None:
        """Close the viewer window, if any. Safe to call on a headless controller."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _viewer_sync(self) -> None:
        """Push the latest physics state to the viewer, a no-op when headless."""
        if self._viewer is not None:
            self._viewer.sync()

    def _joint_positions(self) -> dict[int, float]:
        """Current joint angles for every motor, keyed by motor id."""
        return {
            mid: float(self._data.qpos[self._name_to_qpos_idx[ID_TO_MOTOR[mid]]])
            for mid in MOTOR_TO_ID.values()
        }

    def _reset_pos_history(self) -> None:
        """Fill the whole history with the current pose, so the first differences are zero."""
        snapshot = self._joint_positions()
        self._pos_history.clear()
        for _ in range(self._pos_history.maxlen or 1):
            self._pos_history.append(snapshot)

    def _finite_difference_velocity(self, motor_id: int) -> float:
        """Velocity from the position change across the finite-difference window."""
        return (self._pos_history[-1][motor_id] - self._pos_history[0][motor_id]) / self._fd_dt

    def set_kp(self, kp: float) -> None:
        self._bam.model.actuator.kp = kp

    def sync_read_kp(self, ids: list[int]) -> list[int]:
        kp = int(self._bam.model.actuator.kp)
        return [kp] * len(ids)

    def sync_write_kp(self, ids: list[int], gains: list[int]) -> None:
        self._bam.model.actuator.kp = gains[0]

    def sync_write_torque_enable(self, ids: list[int], values: list[bool]) -> None:
        pass

    def sync_write_status_return_level(self, ids: list[int], levels: list[int]) -> None:
        pass

    def sync_write_goal_position(self, ids: list[int], positions: list[float]) -> None:
        if not self._viewer_running():
            self._stop_flag_path.write_text("stop\n", encoding="ascii")
            return

        cmd = dict(zip(ids, positions))

        for _ in range(self._steps_per_tick):
            for motor_id, pos in cmd.items():
                delayed_pos = self._delay_act[motor_id].push_and_read(pos)
                self._bam.set_q_target(ID_TO_MOTOR[motor_id], delayed_pos)
            self._sample_effective_vin()
            self._bam.update()
            mujoco.mj_step(self._model, self._data)
            if self._velocity_fd:
                self._pos_history.append(self._joint_positions())

        if self._reset_source is not None and self._reset_source.consume_reset():
            self.reset()
            return

        if self._reset_source is not None and self._reset_source.show_torque:
            now = time.monotonic()
            if now - self._last_torque_print >= self._torque_interval:
                total = float(sum(abs(f) for f in self._data.actuator_force))
                print(f"Torque sum: {total:.3f} Nm")
                self._last_torque_print = now

        self._viewer_sync()

    def sync_read_present_position(self, ids: list[int]) -> list[float]:
        return [
            self._delay_pos[mid].push_and_read(
                self._data.qpos[self._name_to_qpos_idx[ID_TO_MOTOR[mid]]]
            )
            for mid in ids
        ]

    def read_present_position(self, motor_id: int) -> float:
        name = ID_TO_MOTOR[motor_id]
        return float(self._delay_pos[motor_id].push_and_read(
            self._data.qpos[self._name_to_qpos_idx[name]]
        ))

    def _raw_velocity(self, motor_id: int) -> float:
        if self._velocity_fd:
            return self._finite_difference_velocity(motor_id)
        return float(self._data.qvel[self._name_to_qvel_idx[ID_TO_MOTOR[motor_id]]])

    def sync_read_present_velocity(self, ids: list[int]) -> list[float]:
        return [
            self._delay_vel[mid].push_and_read(self._raw_velocity(mid))
            for mid in ids
        ]

    def read_present_velocity(self, motor_id: int) -> float:
        return float(self._delay_vel[motor_id].push_and_read(self._raw_velocity(motor_id)))

    def _bus_current(self, motor_name: str) -> float:
        """Battery-side current drawn by one motor [A].

        The phase current is torque / kt, but the H-bridge is a buck stage: the motor draws
        that continuously while the battery only sources it during the PWM on-time, so the
        bus sees duty * torque / kt. That is what a supply-side measurement (and the voltage
        drop in _sample_effective_vin) sees, and it is well below the phase current whenever
        the duty cycle is small. The product is signed: duty and torque of opposite sign is a
        joint braking and pushing current back onto the bus.
        """
        torque = float(self._data.ctrl[self._name_to_actuator_idx[motor_name]])
        phase_current = torque / self._bam.model.kt.value
        duty_cycle = getattr(self._bam.model.actuator, "duty_cycle", None)
        if duty_cycle is None:  # no control step has run yet
            return 0.0
        return float(duty_cycle[self._bam.dof_to_q_target[motor_name]]) * phase_current

    def sync_read_present_current(self, ids: list[int]) -> list[float]:
        # Bus current, not phase current — see _bus_current. Both ctrl (the current-clipped
        # torque set by the last MujocoController.update()) and duty_cycle come from that
        # same update, so they pair up.
        return [self._bus_current(ID_TO_MOTOR[mid]) for mid in ids]

    def _sample_effective_vin(self) -> None:
        """Recompute the supply voltage bam is about to apply, just before it consumes it.

        bam derates vin by the drop across the battery + wire resistance, then restores the
        nominal value before update() returns, so the effective voltage cannot be read back
        afterwards. This mirrors the computation in BamController.update() on exactly the
        state that call will see (the previous step's torques and duty cycles), so what the
        voltage read reports is what the motors actually ran on rather than a constant.
        """
        act = self._bam.model.actuator
        duty_cycle = getattr(act, "duty_cycle", None)
        if duty_cycle is None:  # no control step has run yet
            self._vin_eff = BAM_VIN
            return
        # Battery current: the H-bridge only sources torque / kt during the PWM on-time, and
        # the joints share a supply, so sum the signed per-joint draw before clamping at zero.
        current = float(
            np.sum(
                duty_cycle
                * self._data.qfrc_actuator[self._bam.dof_indexes]
                / self._bam.model.kt.value
            )
        )
        vin = BAM_VIN - BAM_VOLTAGE_DROP_RESISTANCE * max(current, 0.0)
        self._vin_eff = max(vin, BAM_VIN_MIN)

    def sync_read_present_input_voltage(self, ids: list[int]) -> list[float]:
        # Volts, matching what RobotController returns after unit conversion. All motors share
        # the supply, so they all read the same sagged voltage.
        return [self._vin_eff] * len(ids)

    def read_present_input_voltage(self, motor_id: int) -> float:
        return self._vin_eff

    def read_acc(self) -> tuple[float, float, float]:
        """Return pseudo-accelerometer (ax, ay, az) in g from the 'orientation' sensor."""
        if self._sensor_orientation < 0:
            return 0.0, 0.0, -1.0
        adr = self._model.sensor_adr[self._sensor_orientation]
        w, x, y, z = self._data.sensordata[adr:adr + 4]
        # Gravity in world is (0, 0, -1) g; rotate into IMU frame using conjugate quat
        gx = 2 * (x * z - w * y)
        gy = 2 * (y * z + w * x)
        gz = w * w - x * x - y * y + z * z
        return float(gx), float(gy), float(-gz)

    def read_gyro(self) -> tuple[float, float, float]:
        if self._sensor_gyro < 0:
            current = (0.0, 0.0, 0.0)
        else:
            adr = self._model.sensor_adr[self._sensor_gyro]
            gx, gy, gz = self._data.sensordata[adr:adr + 3]
            current = (float(gx), float(gy), float(gz))
        return self._delay_gyro.push_and_read(current)

    def read_quat(self, dt: float) -> tuple[float, float, float, float]:
        if self._sensor_orientation < 0:
            current = (1.0, 0.0, 0.0, 0.0)
        else:
            adr = self._model.sensor_adr[self._sensor_orientation]
            w, x, y, z = self._data.sensordata[adr:adr + 4]
            current = (float(w), float(x), float(y), float(z))
        return self._delay_quat.push_and_read(current)

    def _set_neutral_state(self) -> None:
        """Put the model in the neutral standing pose, at rest and clear of the floor.

        Shared by startup and [r] so both land in exactly the same state.
        """
        self._data.qpos[:] = 0.0
        self._data.qvel[:] = 0.0
        self._data.ctrl[:] = 0.0
        self._data.qpos[2] = SPAWN_HEIGHT
        self._data.qpos[3] = 1.0  # free-joint quaternion (w, x, y, z), upright
        for name, angle in NEUTRAL_POSE.items():
            if name in self._name_to_qpos_idx:
                self._data.qpos[self._name_to_qpos_idx[name]] = angle
        mujoco.mj_forward(self._model, self._data)

    def reset(self) -> None:
        """Reset the simulation to the initial neutral standing pose."""
        self._set_neutral_state()
        # Flush the delay buffers too, so the first ticks after a reset don't replay
        # readings captured while the robot was falling.
        for mid in MOTOR_TO_ID.values():
            neutral = self._data.qpos[self._name_to_qpos_idx[ID_TO_MOTOR[mid]]]
            self._delay_act[mid].fill(neutral)
            self._delay_pos[mid].fill(neutral)
            self._delay_vel[mid].fill(0.0)
        self._delay_gyro.fill((0.0, 0.0, 0.0))
        self._delay_quat.fill((1.0, 0.0, 0.0, 0.0))
        self._vin_eff = BAM_VIN
        self._reset_pos_history()
        self._viewer_sync()