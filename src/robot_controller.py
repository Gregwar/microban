# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import os

from rustypot import Xl330PyController
import numpy as np

from constants import MOTOR_TO_ID, MOTOR_SIGN, IMU_I2C_BUS, MOTOR_BAUDRATE, PRESENT_CURRENT_UNIT_A
from imu_reader import ThreadedIMUReader
from xl330_state import (
    STATE_ADDR,
    STATE_LEN,
    VELOCITY_UNIT_RAD_S,
    decode_state_block,
    ticks_to_radians,
)

# A fused read must agree with the driver's own per-register reads to within this much,
# with the robot at rest, or it is not trusted. One tick is 2*pi/4096 ~= 1.5 mrad, so this
# allows a couple of ticks of jitter between the two reads and nothing more.
FUSED_POSITION_TOLERANCE_RAD = 0.01


class RobotController:
    """Wraps Xl330PyController."""

    def __init__(
        self,
        serial_port: str = "/dev/ttyAMA0",
        baudrate: int | None = None,
        timeout: float = 0.1,
        fused_read: str | None = None,
    ) -> None:
        """
        Args:
            baudrate: bus rate; defaults to MICROBAN_BAUD or constants.MOTOR_BAUDRATE.
                It must match the motors' EEPROM setting (see `make set-baud`).
            fused_read: "auto" (default) checks the fused state read against the driver's
                per-register reads at startup and uses it only if they agree; "off" always
                uses the per-register reads; "on" forces the fused read without checking
                (for benchmarking — do not run a policy on it unverified).
                Overridden by MICROBAN_FUSED_READ.
        """
        if baudrate is None:
            baudrate = int(os.environ.get("MICROBAN_BAUD", MOTOR_BAUDRATE))
        self.baudrate = baudrate
        self._controller = Xl330PyController(serial_port=serial_port, baudrate=baudrate, timeout=timeout)
        self._id_to_sign: dict[int, float] = {MOTOR_TO_ID[name]: MOTOR_SIGN[name] for name in MOTOR_TO_ID}
        self._imu_reader = ThreadedIMUReader(i2c_bus=IMU_I2C_BUS, frequency_hz=200.0)
        self._imu_reader.start()

        self._fused_fallbacks = 0
        mode = (fused_read or os.environ.get("MICROBAN_FUSED_READ", "auto")).lower()
        self.fused_read_enabled = self._resolve_fused_read(mode)

    # ------------------------------------------------------------------
    # Fused state read

    def _resolve_fused_read(self, mode: str) -> bool:
        """Decide whether the observer may use sync_read_state()."""
        if mode == "off":
            return False
        if mode == "on":
            print("Fused motor read forced on (unverified).")
            return True
        if mode != "auto":
            raise ValueError(f"MICROBAN_FUSED_READ must be auto|on|off, got {mode!r}")

        ok, detail = self.check_fused_read()
        if ok:
            print(f"Fused motor read enabled ({detail}).")
        else:
            print(f"Fused motor read disabled — {detail}")
        return ok

    def check_fused_read(self) -> tuple[bool, str]:
        """Compare one fused read against the driver's per-register reads.

        The fused read reaches into the control table by raw address, so a wrong offset or
        endianness would return plausible-looking nonsense to a running policy. Rather than
        trust the addresses, verify them once against the calls that are known to work; the
        robot is at rest at startup, so positions must agree.
        """
        ids = list(MOTOR_TO_ID.values())
        try:
            positions, _, _ = self.sync_read_state(ids)
            reference = self.sync_read_present_position(ids)
        except Exception as exc:
            return False, f"the check itself failed: {exc!r}"

        if len(positions) != len(reference):
            return False, f"got {len(positions)} motors, expected {len(reference)}"

        deviations = [abs(a - b) for a, b in zip(positions, reference)]
        worst = max(deviations)
        if worst > FUSED_POSITION_TOLERANCE_RAD:
            motor = list(MOTOR_TO_ID)[deviations.index(worst)]
            return False, (
                f"positions disagree with the driver by up to {worst:.4f} rad on {motor} "
                f"(tolerance {FUSED_POSITION_TOLERANCE_RAD}); falling back to separate reads"
            )
        return True, f"matches the driver within {worst:.4f} rad"

    def sync_read_state(self, ids: list[int]) -> tuple[list[float], list[float], list[float]]:
        """Positions [rad], velocities [rad/s] and currents [A] in ONE bus transaction.

        Replaces the separate position and velocity sync reads, which cost a full round
        trip each over all 19 motors. Current comes along for the ride, for a few extra
        bytes rather than a third round trip.

        Blocks come back longer than STATE_LEN when Protocol 2.0 stuffs an extra 0xFD into
        the payload (the negative velocity/current values seen while moving produce the
        0xFF 0xFF 0xFD trigger). decode_state_block un-stuffs those on the fast path, so
        stuffing no longer forces a fallback — verified against real motor frames. The
        fallback below now only catches genuinely malformed reads (a short frame, a motor
        that missed its reply), which stay rare.
        """
        blocks = self._controller.sync_read_raw_data(ids, STATE_ADDR, STATE_LEN)

        positions: list[float] = []
        velocities: list[float] = []
        currents: list[float] = []
        for motor_id, block in zip(ids, blocks):
            try:
                current_raw, velocity_raw, position_ticks = decode_state_block(block)
            except ValueError:
                return self._read_state_separately(ids)
            sign = self._id_to_sign[motor_id]
            positions.append(ticks_to_radians(position_ticks) * sign)
            velocities.append(velocity_raw * VELOCITY_UNIT_RAD_S * sign)
            currents.append(current_raw * PRESENT_CURRENT_UNIT_A)

        if len(positions) != len(ids):  # short read (e.g. a motor missed the reply)
            return self._read_state_separately(ids)
        return positions, velocities, currents

    def _read_state_separately(self, ids: list[int]) -> tuple[list[float], list[float], list[float]]:
        """The safe fallback: the driver's typed reads, which handle byte stuffing.

        Slower than the fused read (three round trips instead of one), but only taken on
        the ticks whose data happened to need stuffing, so the fast path still carries the
        majority of reads.
        """
        self._fused_fallbacks += 1
        if self._fused_fallbacks in (1, 100) or self._fused_fallbacks % 1000 == 0:
            print(
                f"Note: fused read fell back to separate reads (byte-stuffed frame); "
                f"{self._fused_fallbacks} so far this session.",
                end="\r\n",
                flush=True,
            )
        positions = self.sync_read_present_position(ids)
        velocities = self.sync_read_present_velocity(ids)
        currents = self.sync_read_present_current(ids)
        return positions, velocities, currents

    def sync_write_torque_enable(self, ids: list[int], values: list[bool]) -> None:
        self._controller.sync_write_torque_enable(ids, values)

    def sync_write_status_return_level(self, ids: list[int], levels: list[int]) -> None:
        self._controller.sync_write_status_return_level(ids, levels)

    def sync_write_goal_position(self, ids: list[int], positions: list[float]) -> None:
        hw_positions = [pos * self._id_to_sign[motor_id] for motor_id, pos in zip(ids, positions)]
        self._controller.sync_write_goal_position(ids, hw_positions)

    def sync_read_present_position(self, ids: list[int]) -> list[float]:
        raw = self._controller.sync_read_present_position(ids)
        return [r * self._id_to_sign[motor_id] for motor_id, r in zip(ids, raw)]

    def read_present_position(self, motor_id: int) -> float:
        raw = self._controller.read_present_position(motor_id)
        return raw * self._id_to_sign[motor_id]

    def sync_read_present_velocity(self, ids: list[int]) -> list[float]:
        raw = np.array(self._controller.sync_read_present_velocity(ids)) * 0.229 * np.pi / 30
        return [r * self._id_to_sign[motor_id] for motor_id, r in zip(ids, raw)]
    
    def read_present_velocity(self, motor_id: int) -> float:
        raw = self._controller.read_present_velocity(motor_id) * 0.229 * np.pi / 30
        return raw * self._id_to_sign[motor_id]

    def sync_read_present_current(self, ids: list[int]) -> list[float]:
        """Present current per motor, in Amps (signed). Magnitude is what matters for the BMS budget."""
        raw = self._controller.sync_read_present_current(ids)
        return [
            (r[0] if isinstance(r, (list, tuple)) else float(r)) * PRESENT_CURRENT_UNIT_A
            for r in raw
        ]

    def sync_read_present_input_voltage(self, ids: list[int]) -> list[float]:
        raw = self._controller.sync_read_present_input_voltage(ids)
        return [r[0] if isinstance(r, (list, tuple)) else float(r) for r in raw]

    def read_present_input_voltage(self, motor_id: int) -> float:
        raw = self._controller.read_present_input_voltage(motor_id)
        return raw[0] if isinstance(raw, (list, tuple)) else float(raw)

    def sync_read_kp(self, ids: list[int]) -> list[int]:
        return [int(v) for v in self._controller.sync_read_position_p_gain(ids)]

    def sync_write_kp(self, ids: list[int], gains: list[int]) -> None:
        self._controller.sync_write_position_p_gain(ids, gains)

    def read_acc(self) -> tuple[float, float, float]:
        """Return raw accelerometer (ax, ay, az) in g."""
        return self._imu_reader.get_latest().acc

    def read_gyro(self) -> tuple[float, float, float]:
        """Return (gx, gy, gz) in rad/s."""
        return self._imu_reader.get_latest().gyro

    def read_quat(self, dt: float) -> tuple[float, float, float, float]:
        """Return orientation quaternion (w, x, y, z)."""
        _ = dt
        return self._imu_reader.get_latest().quat

    def get_imu_status(self) -> dict[str, float | int | bool]:
        return self._imu_reader.get_status()

    def shutdown(self) -> None:
        self._imu_reader.stop()

    def close(self) -> None:
        self.shutdown()