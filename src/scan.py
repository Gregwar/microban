# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Ping the motors, report what answers, and blink their LEDs to identify them.

    make scan                # ping every known ID, then chase the LEDs one by one
    make scan ID=12          # only that motor (left_hip_roll)
    make scan ARGS=--no-led  # ping only, touch nothing

Read-only apart from the LED register: no torque is enabled and no goal position is
written, so this is safe to run with the robot on the bench.

The chase blinks one motor at a time, in ID order, printing the name as it goes: watch
the robot and you see which physical joint carries which ID. Use it when a motor is
silent (is it dead, or is it answering under an ID you did not expect?) or after
re-cabling a limb.
"""

import argparse
import os
import sys
import time

from rustypot import Xl330PyController

from constants import MOTOR_TO_ID, ID_TO_MOTOR, MOTOR_BAUDRATE, PRESENT_VOLTAGE_UNIT_V

SERIAL_PORT = "/dev/ttyAMA0"
LED_ON, LED_OFF = 1, 0


def _scalar(value) -> float:
    """Single-register reads come back either bare or wrapped, depending on the register."""
    return float(value[0] if isinstance(value, (list, tuple)) else value)


def ping_all(bus: Xl330PyController, motor_ids: list[int]) -> list[int]:
    """IDs that answer, printed with the details worth having when one does not."""
    found = []
    for motor_id in motor_ids:
        name = ID_TO_MOTOR.get(motor_id, "?")
        try:
            bus.ping(motor_id)
        except Exception as exc:
            print(f"  {motor_id:2d} {name:<22} NO ANSWER ({type(exc).__name__})")
            continue
        found.append(motor_id)
        try:
            voltage = _scalar(bus.read_present_input_voltage(motor_id)) * PRESENT_VOLTAGE_UNIT_V
            temperature = int(_scalar(bus.read_present_temperature(motor_id)))
            firmware = int(_scalar(bus.read_firmware_version(motor_id)))
            errors = int(_scalar(bus.read_hardware_error_status(motor_id)))
            flag = "  <-- HW ERROR" if errors else ""
            print(
                f"  {motor_id:2d} {name:<22} ok   {voltage:5.2f} V  {temperature:3d} C"
                f"  fw {firmware}  err 0x{errors:02x}{flag}"
            )
        except Exception as exc:
            print(f"  {motor_id:2d} {name:<22} ok   (ping only, reads failed: {exc})")
    return found


def blink_together(bus: Xl330PyController, motor_ids: list[int], times: int = 3) -> None:
    """All found motors flashing at once: a quick 'everyone I can see is this set'."""
    for _ in range(times):
        bus.sync_write_led(motor_ids, [LED_ON] * len(motor_ids))
        time.sleep(0.25)
        bus.sync_write_led(motor_ids, [LED_OFF] * len(motor_ids))
        time.sleep(0.25)


def chase(bus: Xl330PyController, motor_ids: list[int], hold: float = 0.6) -> None:
    """One motor lit at a time, so the printed name maps onto a joint you can see."""
    for motor_id in motor_ids:
        print(f"  {motor_id:2d} {ID_TO_MOTOR.get(motor_id, '?')}")
        bus.write_led(motor_id, LED_ON)
        time.sleep(hold)
        bus.write_led(motor_id, LED_OFF)
        time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("id", nargs="?", type=int, help="restrict to a single motor ID")
    parser.add_argument("--no-led", action="store_true", help="ping only, never write the LED register")
    parser.add_argument("--baud", type=int, default=int(os.environ.get("MICROBAN_BAUD", MOTOR_BAUDRATE)))
    args = parser.parse_args()

    motor_ids = [args.id] if args.id else list(MOTOR_TO_ID.values())

    bus = Xl330PyController(serial_port=SERIAL_PORT, baudrate=args.baud, timeout=0.05)

    print(f"Pinging {len(motor_ids)} motor(s) on {SERIAL_PORT} @ {args.baud} baud:")
    found = ping_all(bus, motor_ids)
    print(f"\n{len(found)}/{len(motor_ids)} answered.")

    missing = [i for i in motor_ids if i not in found]
    if missing:
        print(f"Silent: {missing} ({', '.join(ID_TO_MOTOR.get(i, '?') for i in missing)})")
        print("If the whole bus is silent, the motors may be at another baud rate: `make set-baud` surveys them.")

    if not found:
        sys.exit(1)

    if args.no_led:
        return

    # Level 1 = reply to READ only, matching what the control loop expects.
    bus.sync_write_status_return_level(found, [1] * len(found))

    print("\nBlinking all:")
    blink_together(bus, found)
    print("Chasing, one at a time:")
    chase(bus, found)
    bus.sync_write_led(found, [LED_OFF] * len(found))


if __name__ == "__main__":
    main()
