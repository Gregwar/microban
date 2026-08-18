# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import os

from rustypot import Xl330PyController
import numpy as np

from bus_monitor import BusMonitor
from constants import MOTOR_TO_ID, MOTOR_SIGN, IMU_I2C_BUS, MOTOR_BAUDRATE, PRESENT_CURRENT_UNIT_A, PRESENT_VOLTAGE_UNIT_V
from imu_reader import ThreadedIMUReader
from xl330_state import (
    STATE_ADDR,
    STATE_EXT_LEN,
    STATE_LEN,
    VELOCITY_UNIT_RAD_S,
    decode_state_block,
    decode_state_block_ext,
    ticks_to_radians,
)

# A fused read must agree with the driver's own per-register reads to within this much,
# with the robot at rest, or it is not trusted. One tick is 2*pi/4096 ~= 1.5 mrad, so this
# allows a couple of ticks of jitter between the two reads and nothing more.
FUSED_POSITION_TOLERANCE_RAD = 0.01

# Same idea for the widened block's voltage field. The register quantises to 0.1 V and the
# supply moves between the two reads, so this only has to catch a wrong offset — which
# would land in a trajectory register and be wrong by volts, not by tenths.
FUSED_VOLTAGE_TOLERANCE_V = 0.5


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

        self.stats = BusMonitor()
        # Ping every motor after an unattributable failure to name the silent one. Off by
        # default: it costs 19 extra round trips, inside a tick that has already overrun.
        self.probe_on_failure = os.environ.get("MICROBAN_BUS_PROBE", "0").lower() in ("1", "on", "true")
        self._fused_fallbacks = 0
        mode = (fused_read or os.environ.get("MICROBAN_FUSED_READ", "auto")).lower()
        self.fused_read_enabled = self._resolve_fused_read(mode)
        # Whether the widened block (reaching Present Input Voltage) is trustworthy too.
        # Checked separately: the base block can be sound while the wider offset is not,
        # and in that case voltage should cost a second read rather than return nonsense.
        self.fused_voltage_enabled = (
            self._resolve_fused_voltage() if self.fused_read_enabled else False
        )

    # ------------------------------------------------------------------
    # Bus accounting

    def _read(self, fn, ids, *args):
        """Run one read transaction, counting the packets it sent and got back.

        Every bus read goes through here so the [p] report reflects the whole session
        rather than whichever call sites were remembered. A failure is handed to the
        monitor along with the ids it was asking for, which is what lets the report name
        the motor at fault — see bus_monitor for how the id is recovered.
        """
        self.stats.record_sent(len(ids))
        try:
            result = fn(ids, *args)
        except Exception as exc:
            self._record_failure(getattr(fn, "__name__", "read"), ids, exc)
            raise
        # rustypot's sync_read aborts on the first motor that does not answer rather than
        # returning a short list, so a full-length result means every motor replied.
        self.stats.record_received(len(result) if hasattr(result, "__len__") else 1)
        return result

    def _read_one(self, fn, motor_id, *args):
        """Same, for the single-motor reads."""
        self.stats.record_sent(1)
        try:
            result = fn(motor_id, *args)
        except Exception as exc:
            # One id, so there is nothing to disambiguate: whatever went wrong was this
            # motor's, including the timeouts a sync read could not have attributed.
            self.stats.record_exception(
                getattr(fn, "__name__", "read"), [motor_id], exc, attribute_to=motor_id
            )
            raise
        self.stats.record_received(1)
        return result

    def _write(self, fn, ids, *args) -> None:
        """Run one write transaction. No status packet is expected (return level 1)."""
        self.stats.record_sent()
        try:
            fn(ids, *args)
        except Exception as exc:
            self._record_failure(getattr(fn, "__name__", "write"), ids, exc)
            raise

    def _record_failure(self, op: str, ids: list[int], exc: Exception) -> None:
        """Log the fault, and optionally ping to find out whose it was.

        The ping only runs when the frame carried no id to blame (a timeout, a corrupted
        frame) and MICROBAN_BUS_PROBE is set — it is the difference between "something on
        the bus timed out" and "left_knee is not answering", but it is far too expensive to
        do on every tick of a run that is faulting continuously.
        """
        fault = self.stats.record_exception(op, list(ids), exc)
        if fault.motor is None and self.probe_on_failure:
            self.stats.record_probe(self._ping_silent(ids), op)

    def _ping_silent(self, ids: list[int]) -> list[int]:
        """Which of `ids` do not answer a ping. Never raises: this runs inside error handling."""
        silent = []
        for motor_id in ids:
            try:
                if not self._controller.ping(motor_id):
                    silent.append(motor_id)
            except Exception:
                silent.append(motor_id)
        return silent

    def get_bus_stats(self) -> dict:
        """Snapshot of the traffic counters and per-motor faults, for the [p] report."""
        return self.stats.snapshot()

    def get_bus_report_lines(self) -> list[str]:
        """The per-motor fault breakdown [p] prints under the packet counters."""
        return self.stats.report_lines()

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

    def _resolve_fused_voltage(self) -> bool:
        """Check the widened block's voltage field against the driver's own read."""
        ok, detail = self.check_fused_voltage()
        if ok:
            print(f"Fused voltage read enabled ({detail}).")
        else:
            print(f"Fused voltage read disabled — {detail}")
        return ok

    def check_fused_voltage(self) -> tuple[bool, str]:
        """Verify Present Input Voltage really is where the widened block expects it.

        A wrong offset here would land in Velocity/Position Trajectory and quietly report
        those as volts, so the address is checked rather than trusted — the same treatment
        the base block gets.
        """
        ids = list(MOTOR_TO_ID.values())
        try:
            _, _, _, voltages = self.sync_read_state(ids, include_voltage=True)
            reference = self.sync_read_present_input_voltage(ids)
        except Exception as exc:
            return False, f"the check itself failed: {exc!r}"

        if len(voltages) != len(reference):
            return False, f"got {len(voltages)} motors, expected {len(reference)}"

        deviations = [abs(a - b) for a, b in zip(voltages, reference)]
        worst = max(deviations)
        if worst > FUSED_VOLTAGE_TOLERANCE_V:
            motor = list(MOTOR_TO_ID)[deviations.index(worst)]
            return False, (
                f"voltages disagree with the driver by up to {worst:.2f} V on {motor} "
                f"(tolerance {FUSED_VOLTAGE_TOLERANCE_V}); voltage will cost a second read"
            )
        return True, f"matches the driver within {worst:.2f} V"

    def check_fused_read(self) -> tuple[bool, str]:
        """Compare one fused read against the driver's per-register reads.

        The fused read reaches into the control table by raw address, so a wrong offset or
        endianness would return plausible-looking nonsense to a running policy. Rather than
        trust the addresses, verify them once against the calls that are known to work; the
        robot is at rest at startup, so positions must agree.
        """
        ids = list(MOTOR_TO_ID.values())
        try:
            positions, _, _, _ = self.sync_read_state(ids)
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

    def sync_read_state(
        self, ids: list[int], include_voltage: bool = False
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """Positions [rad], velocities [rad/s], currents [A] and voltages [V] in ONE read.

        Replaces the separate position and velocity sync reads, which cost a full round
        trip each over all 19 motors. Current comes along for the ride, for a few extra
        bytes rather than a third round trip.

        With `include_voltage`, the block is widened to reach Present Input Voltage as
        well — still one transaction, at the cost of reading through the two trajectory
        registers in between. Cheaper than a second round trip; see xl330_state. Voltages
        come back empty when not asked for.

        Blocks come back longer than expected when Protocol 2.0 stuffs an extra 0xFD into
        the payload (the negative velocity/current values seen while moving produce the
        0xFF 0xFF 0xFD trigger). The decoder un-stuffs those on the fast path, so stuffing
        no longer forces a fallback — verified against real motor frames. The fallback
        below now only catches genuinely malformed reads (a short frame, a motor that
        missed its reply), which stay rare.
        """
        length = STATE_EXT_LEN if include_voltage else STATE_LEN
        decode = decode_state_block_ext if include_voltage else decode_state_block
        blocks = self._read(self._controller.sync_read_raw_data, ids, STATE_ADDR, length)

        positions: list[float] = []
        velocities: list[float] = []
        currents: list[float] = []
        voltages: list[float] = []
        for motor_id, block in zip(ids, blocks):
            try:
                decoded = decode(block)
            except ValueError as exc:
                # The one fault that never needs detective work: we are holding the block
                # and the id it came from, so [p] can name the motor outright.
                self.stats.record_malformed("sync_read_state", motor_id, str(exc))
                return self._read_state_separately(ids, include_voltage)
            current_raw, velocity_raw, position_ticks = decoded[:3]
            sign = self._id_to_sign[motor_id]
            positions.append(ticks_to_radians(position_ticks) * sign)
            velocities.append(velocity_raw * VELOCITY_UNIT_RAD_S * sign)
            currents.append(current_raw * PRESENT_CURRENT_UNIT_A)
            if include_voltage:
                voltages.append(decoded[3] * PRESENT_VOLTAGE_UNIT_V)

        if len(positions) != len(ids):  # short read (e.g. a motor missed the reply)
            return self._read_state_separately(ids, include_voltage)
        return positions, velocities, currents, voltages

    def _read_state_separately(
        self, ids: list[int], include_voltage: bool = False
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """The safe fallback: the driver's typed reads, which handle byte stuffing.

        Slower than the fused read (three round trips instead of one), but only taken on
        the ticks whose data happened to need stuffing, so the fast path still carries the
        majority of reads.
        """
        self._fused_fallbacks += 1
        self.stats.record_fallback()
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
        voltages = self.sync_read_present_input_voltage(ids) if include_voltage else []
        return positions, velocities, currents, voltages

    def sync_write_torque_enable(self, ids: list[int], values: list[bool]) -> None:
        self._write(self._controller.sync_write_torque_enable, ids, values)

    def sync_write_status_return_level(self, ids: list[int], levels: list[int]) -> None:
        self._write(self._controller.sync_write_status_return_level, ids, levels)

    def sync_write_goal_position(self, ids: list[int], positions: list[float]) -> None:
        hw_positions = [pos * self._id_to_sign[motor_id] for motor_id, pos in zip(ids, positions)]
        self._write(self._controller.sync_write_goal_position, ids, hw_positions)

    def sync_read_present_position(self, ids: list[int]) -> list[float]:
        raw = self._read(self._controller.sync_read_present_position, ids)
        return [r * self._id_to_sign[motor_id] for motor_id, r in zip(ids, raw)]

    def read_present_position(self, motor_id: int) -> float:
        raw = self._read_one(self._controller.read_present_position, motor_id)
        return raw * self._id_to_sign[motor_id]

    def sync_read_present_velocity(self, ids: list[int]) -> list[float]:
        raw = np.array(self._read(self._controller.sync_read_present_velocity, ids)) * 0.229 * np.pi / 30
        return [r * self._id_to_sign[motor_id] for motor_id, r in zip(ids, raw)]
    
    def read_present_velocity(self, motor_id: int) -> float:
        raw = self._read_one(self._controller.read_present_velocity, motor_id) * 0.229 * np.pi / 30
        return raw * self._id_to_sign[motor_id]

    def sync_read_present_current(self, ids: list[int]) -> list[float]:
        """Present current per motor, in Amps (signed). Magnitude is what matters for the BMS budget."""
        raw = self._read(self._controller.sync_read_present_current, ids)
        return [
            (r[0] if isinstance(r, (list, tuple)) else float(r)) * PRESENT_CURRENT_UNIT_A
            for r in raw
        ]

    def sync_read_present_input_voltage(self, ids: list[int]) -> list[float]:
        """Supply voltage per motor, in Volts. The register counts 0.1 V/LSB."""
        raw = self._read(self._controller.sync_read_present_input_voltage, ids)
        return [
            (r[0] if isinstance(r, (list, tuple)) else float(r)) * PRESENT_VOLTAGE_UNIT_V
            for r in raw
        ]

    def read_present_input_voltage(self, motor_id: int) -> float:
        """Supply voltage for one motor, in Volts."""
        raw = self._read_one(self._controller.read_present_input_voltage, motor_id)
        value = raw[0] if isinstance(raw, (list, tuple)) else float(raw)
        return value * PRESENT_VOLTAGE_UNIT_V

    def sync_read_kp(self, ids: list[int]) -> list[int]:
        return [int(v) for v in self._read(self._controller.sync_read_position_p_gain, ids)]

    def sync_write_kp(self, ids: list[int], gains: list[int]) -> None:
        self._write(self._controller.sync_write_position_p_gain, ids, gains)

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

    def get_imu_sample(self):
        """The latest IMUSnapshot, with the timestamp of the sample it was built from.

        read_acc/read_gyro/read_quat hand back the values alone, which is all a policy
        needs. Anything timing the IMU also needs to know *when* the sample was taken —
        the reader runs on its own thread, so the freshest sample is already a few ms old
        by the time a tick asks for it. See src/imu_delay_record.py.
        """
        return self._imu_reader.get_latest()

    def get_imu_status(self) -> dict[str, float | int | bool]:
        return self._imu_reader.get_status()

    def shutdown(self) -> None:
        self._imu_reader.stop()

    def close(self) -> None:
        self.shutdown()