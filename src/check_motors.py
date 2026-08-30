# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Audit every motor on the bus: who answers, and is its EEPROM set up the way we assume.

    make check-motors                  # the whole fleet
    make check-motors ID=24            # one motor
    make check-motors ARGS=--verbose   # print every register, not just the deviations
    make check-motors ARGS=--full-scan # also sweep IDs 0-252 for strays (a fresh motor is id 1)

Strictly read-only: no torque, no goal position, not even the LED. Safe with the robot
sitting on the bench, and safe to run between two control sessions.

Two things go wrong that this catches. A motor stops answering — then the question is
whether it is dead, unplugged, or simply at another baud rate, so a silent bus is
re-scanned at every rate we might have left it at. Or a motor answers but was never
configured: a replacement swapped in with factory EEPROM has Return Delay Time 254, the
default PWM Slope and Shutdown still trips on input voltage. It walks, badly, and the
cause is invisible in the logs. The expectations below come from the motor setup checklist
in docs/assembly.md and from what the rest of this repo assumes about the hardware.
"""

import argparse
import os
import sys

from rustypot import Xl330PyController

from constants import (
    BAM_MAX_CURRENT,
    BAM_VIN,
    ID_TO_MOTOR,
    MOTOR_BAUDRATE,
    MOTOR_TO_ID,
    PRESENT_CURRENT_UNIT_A,
    PRESENT_VOLTAGE_UNIT_V,
)
from set_baud import BAUD_CODES, SCAN_BAUDS

SERIAL_PORT = "/dev/ttyAMA0"
XL330_M288_MODEL_NUMBER = 1200

# The input-voltage bit latches on every motor of this robot, and it is not a fault: the
# 2S pack sits near BAM_VIN (7.9 V) while the XL330's Max Voltage Limit is 7.0 V, so the
# firmware flags an overvoltage the moment it powers up. Shutdown is set to 52 precisely
# to unmask that bit (docs/assembly.md, step 1) so the motors keep running anyway. Hardware
# Error Status reports it regardless of the Shutdown mask, so it is expected here -- only
# the other bits mean something has actually gone wrong.
EXPECTED_ERRORS = 1 << 0

# Error/shutdown bit names, shared by Shutdown (63) and Hardware Error Status (70).
ERROR_BITS = {
    0: "input voltage",
    2: "overheating",
    3: "motor encoder",
    4: "electrical shock",
    5: "overload",
}

# (register, expected, why it matters). `expected` None means "report it, never judge it".
# MUST: a deviation changes how the robot behaves, or means the motor was never set up.
MUST = [
    ("model_number", XL330_M288_MODEL_NUMBER, "not an XL330-M288: gearing and limits differ"),
    ("return_delay_time", 0, "delays every reply; at 50 Hz x 19 motors it eats the tick"),
    ("pwm_slope", 255, "ramps the PWM, adding lag the sim does not model"),
    ("shutdown", 52, "the default (53) shuts the motor down on battery sag"),
    ("drive_mode", 0, "reverses the joint; direction belongs in MOTOR_SIGN, not EEPROM"),
    ("operating_mode", 3, "not position control: goal positions would be ignored"),
    ("homing_offset", 0, "silently shifts the zero; offsets belong in the code"),
    ("raw_min_position_limit", 0, "clips the joint range the policy commands"),
    ("raw_max_position_limit", 4095, "clips the joint range the policy commands"),
    ("current_limit", round(BAM_MAX_CURRENT / PRESENT_CURRENT_UNIT_A),
     "the torque cap BAM_MAX_CURRENT and the sim actuator model assume"),
]

# ADVISORY: factory defaults we never change. A deviation is not always wrong, but it was
# not us, so it is worth seeing before chasing a behaviour difference between two motors.
ADVISORY = [
    ("pwm_limit", 885, "caps available torque"),
    ("temperature_limit", 70, "when the motor shuts itself down"),
    ("max_voltage_limit", 70, "0.1 V units"),
    ("min_voltage_limit", 35, "0.1 V units"),
    ("protocol_type", 2, "protocol 2.0"),
    ("secondary_id", 255, "255 = no secondary id, which is what we want"),
    ("startup_configuration", 0, "torque on boot / RAM restore"),
]

# RUNTIME: RAM, written by the control loop at startup. Printed, never judged — reading a
# stale value here says only what the last session left behind.
RUNTIME = ["torque_enable", "status_return_level", "position_p_gain"]


def scalar(value):
    """Single-register reads come back either bare or wrapped, depending on the register."""
    return value[0] if isinstance(value, (list, tuple)) else value


def decode_bits(value: int) -> str:
    names = [name for bit, name in sorted(ERROR_BITS.items()) if value & (1 << bit)]
    return ", ".join(names) if names else "none"


def read_register(bus, ids: list[int], register: str) -> dict[int, int | None]:
    """One register across the fleet. Sync first, per-motor when that fails.

    A sync read is one transaction instead of nineteen, but the driver discards the whole
    batch at the first reply that does not parse -- so one flaky motor would blank the
    row. Falling back keeps the other eighteen readable and pins the failure on its motor.
    """
    try:
        values = getattr(bus, f"sync_read_{register}")(ids)
        return {i: int(scalar(v)) for i, v in zip(ids, values)}
    except Exception:
        pass
    out: dict[int, int | None] = {}
    for motor_id in ids:
        try:
            out[motor_id] = int(scalar(getattr(bus, f"read_{register}")(motor_id)))
        except Exception:
            out[motor_id] = None
    return out


def ping_all(bus, ids: list[int]) -> list[int]:
    found = []
    for motor_id in ids:
        try:
            if bus.ping(motor_id):
                found.append(motor_id)
        except Exception:
            pass
    return found


def report_presence(bus, ids: list[int], baud: int) -> tuple[list[int], list[int]]:
    """Who answers, and how they are doing. Returns (answered, motors flagging an error)."""
    print(f"=== Presence: {len(ids)} motor(s) on {SERIAL_PORT} @ {baud} baud ===")
    found = ping_all(bus, ids)

    voltage = read_register(bus, found, "present_input_voltage")
    temperature = read_register(bus, found, "present_temperature")
    firmware = read_register(bus, found, "firmware_version")
    hw_error = read_register(bus, found, "hardware_error_status")

    for motor_id in ids:
        name = ID_TO_MOTOR.get(motor_id, "?")
        if motor_id not in found:
            print(f"  {motor_id:2d} {name:<22} NO ANSWER")
            continue
        volts = voltage.get(motor_id)
        celsius = temperature.get(motor_id)
        version = firmware.get(motor_id)
        errors = hw_error.get(motor_id)
        volts_s = f"{volts * PRESENT_VOLTAGE_UNIT_V:5.2f} V" if volts is not None else "   ? V"
        celsius_s = f"{celsius:3d} C" if celsius is not None else "  ? C"
        version_s = f"fw {version}" if version is not None else "fw ?"
        real = (errors or 0) & ~EXPECTED_ERRORS
        flag = f"   <-- HW ERROR: {decode_bits(real)}" if real else ""
        print(f"  {motor_id:2d} {name:<22} ok   {volts_s}  {celsius_s}  {version_s}{flag}")

    print(f"\n  {len(found)}/{len(ids)} answered.")

    versions = {v for v in firmware.values() if v is not None}
    if len(versions) > 1:
        print(f"  Mixed firmware on the bus: {sorted(versions)}. Behaviour can differ between motors.")

    expected = [i for i in found if (hw_error.get(i) or 0) & EXPECTED_ERRORS]
    if expected:
        print(f"  {len(expected)}/{len(found)} latch the input-voltage bit. Expected: the pack runs near "
              f"{BAM_VIN} V,")
        print("  above the XL330's 7.0 V limit, and Shutdown 52 unmasks it so they keep running.")

    faulted = [i for i in found if (hw_error.get(i) or 0) & ~EXPECTED_ERRORS]
    if faulted:
        print("  A latched hardware error survives until the motor is power-cycled or rebooted.")
    return found, faulted


def hunt_missing(missing: list[int], current_baud: int) -> None:
    """A motor that is silent here may just be at another rate -- worth knowing which."""
    print("\n=== Hunting the silent motors at other baud rates ===")
    for baud in SCAN_BAUDS:
        if baud == current_baud:
            continue
        try:
            bus = Xl330PyController(serial_port=SERIAL_PORT, baudrate=baud, timeout=0.05)
        except Exception as exc:
            print(f"  {baud:>9} : cannot open the port ({exc})")
            continue
        answered = [i for i in missing if i in ping_all(bus, missing)]
        print(f"  {baud:>9} : {len(answered)}/{len(missing)}" + (f" -> {answered}" if answered else ""))
        if answered:
            names = ", ".join(ID_TO_MOTOR.get(i, "?") for i in answered)
            print(f"              {names} live at {baud}, not {current_baud}.")
            print(f"              `make set-baud BAUD={current_baud}` brings the bus back together.")
    print("  Nothing at any rate = power, wiring, or a connector -- not the baud rate.")


def full_scan(current_baud: int, known: list[int]) -> None:
    """Sweep the whole id space for motors we do not expect: a fresh one is still id 1."""
    print("\n=== Full id sweep (0-252) ===")
    bus = Xl330PyController(serial_port=SERIAL_PORT, baudrate=current_baud, timeout=0.02)
    strays = []
    for motor_id in range(253):
        if motor_id in known:
            continue
        try:
            if bus.ping(motor_id):
                strays.append(motor_id)
        except Exception:
            pass
    if strays:
        print(f"  Unexpected ids answering: {strays}")
        print("  A motor still at its factory id (1) has never been through the setup checklist.")
    else:
        print("  No motor outside the expected id list.")


def report_settings(bus, found: list[int], baud: int, verbose: bool) -> int:
    """Compare EEPROM against what the robot assumes. Returns the MUST-deviation count."""
    print("\n=== Settings ===")
    checks = MUST + [("baud_rate", BAUD_CODES[baud], f"code for {baud} baud")]
    values = {reg: read_register(bus, found, reg) for reg, _, _ in checks}
    values.update({reg: read_register(bus, found, reg) for reg, _, _ in ADVISORY})
    values.update({reg: read_register(bus, found, reg) for reg in RUNTIME})

    def deviations(table):
        out = {}
        for motor_id in found:
            bad = [(reg, values[reg].get(motor_id), exp, why)
                   for reg, exp, why in table
                   if values[reg].get(motor_id) is not None and values[reg][motor_id] != exp]
            if bad:
                out[motor_id] = bad
        return out

    unreadable = [(i, reg) for reg, table in values.items() for i in found if table.get(i) is None]

    must_bad = deviations(checks)
    advisory_bad = deviations(ADVISORY)

    for motor_id in found:
        name = ID_TO_MOTOR.get(motor_id, "?")
        n_must, n_advisory = len(must_bad.get(motor_id, [])), len(advisory_bad.get(motor_id, []))
        if n_must:
            verdict = f"{n_must} SETTING(S) WRONG"
        elif n_advisory:
            verdict = f"ok ({n_advisory} non-default)"
        else:
            verdict = "ok"
        print(f"  {motor_id:2d} {name:<22} {verdict}")

    for label, table in (("must be fixed", must_bad), ("non-default, check it was on purpose", advisory_bad)):
        if not table:
            continue
        print(f"\n  --- {label} ---")
        for motor_id, bad in table.items():
            print(f"  {motor_id:2d} {ID_TO_MOTOR.get(motor_id, '?')}")
            for reg, got, exp, why in bad:
                extra = f"  [{decode_bits(got)}]" if reg == "shutdown" else ""
                print(f"       {reg:<24} is {got}, expected {exp}{extra}")
                print(f"       {'':<24} {why}")

    if unreadable:
        print("\n  --- could not be read ---")
        for motor_id, reg in unreadable:
            print(f"  {motor_id:2d} {ID_TO_MOTOR.get(motor_id, '?'):<22} {reg}")

    if verbose:
        registers = [reg for reg, _, _ in checks] + [reg for reg, _, _ in ADVISORY] + RUNTIME
        print("\n  --- every register read ---")
        print(f"  {'register':<26} " + " ".join(f"{i:>5d}" for i in found))
        for reg in registers:
            # An unreadable register is stored as None, so .get's default never fires.
            row = " ".join(f"{'?' if values[reg].get(i) is None else values[reg][i]:>5}" for i in found)
            print(f"  {reg:<26} {row}")

    print("\n  Runtime registers (torque_enable, status_return_level, position_p_gain) are RAM:")
    print("  the control loop writes them at startup, so they say what the last session left.")
    return sum(len(v) for v in must_bad.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Check motor presence and settings (read-only).")
    parser.add_argument("id", nargs="?", type=int, help="restrict to a single motor ID")
    parser.add_argument("--baud", type=int, default=int(os.environ.get("MICROBAN_BAUD", MOTOR_BAUDRATE)))
    parser.add_argument("--verbose", action="store_true", help="print every register for every motor")
    parser.add_argument("--full-scan", action="store_true", help="sweep ids 0-252 for unexpected motors")
    parser.add_argument("--no-hunt", action="store_true", help="do not re-scan other baud rates for silent motors")
    args = parser.parse_args()

    ids = [args.id] if args.id else list(MOTOR_TO_ID.values())
    if args.baud not in BAUD_CODES:
        raise SystemExit(f"Unknown baud {args.baud}. Known: {sorted(BAUD_CODES)}")

    bus = Xl330PyController(serial_port=SERIAL_PORT, baudrate=args.baud, timeout=0.05)
    found, faulted = report_presence(bus, ids, args.baud)

    missing = [i for i in ids if i not in found]
    if missing and not args.no_hunt:
        hunt_missing(missing, args.baud)
    if args.full_scan:
        full_scan(args.baud, list(MOTOR_TO_ID.values()))

    wrong = report_settings(bus, found, args.baud, args.verbose) if found else 0

    print("\n=== Summary ===")
    print(f"  {len(found)}/{len(ids)} motors answered, {wrong} setting(s) wrong, "
          f"{len(faulted)} reporting a hardware error.")
    if missing:
        print("  Silent: " + ", ".join(f"{i} ({ID_TO_MOTOR.get(i, '?')})" for i in missing))
    if faulted:
        print("  Hardware error: " + ", ".join(f"{i} ({ID_TO_MOTOR.get(i, '?')})" for i in faulted))
    if wrong:
        print("  Fix the wrong settings with Dynamixel Wizard over a U2D2 (docs/assembly.md, step 1).")
    if not missing and not wrong and not faulted:
        print("  Bus is healthy and every motor matches the setup checklist.")
    sys.exit(1 if missing or wrong or faulted else 0)


if __name__ == "__main__":
    main()
