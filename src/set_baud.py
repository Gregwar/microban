# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Change the motor bus baud rate.

    make set-baud BAUD=2000000     # speed the bus up
    make set-baud BAUD=1000000     # put it back

Baud Rate lives in the motors' EEPROM: they switch the moment it is written and stay that
way across power cycles. If the new rate does not work on this Pi's UART or wiring, a motor
that has already switched can no longer be reached — recovering it then needs a U2D2 and a
PC. That is why this script:

  * refuses to run unless it can confirm the baud code table against the motors themselves;
  * moves ONE motor first (the head — the only joint no policy drives, so stranding it
    costs a nod, not a walk) and proves the new rate works before touching the other 18;
  * scans every known rate, so a bus left half-converted by an interrupted run is
    diagnosed and repaired rather than left a mystery.

Read the verdict at the end: nothing else in the repo talks to the motors at the new rate
until constants.MOTOR_BAUDRATE is updated to match.
"""

import sys
import time
import argparse

from rustypot import Xl330PyController

from constants import MOTOR_TO_ID, MOTOR_BAUDRATE

# XL330 Baud Rate register (EEPROM, addr 8). Verified against the motors before any write.
BAUD_CODES = {
    9_600: 0,
    57_600: 1,
    115_200: 2,
    1_000_000: 3,
    2_000_000: 4,
    3_000_000: 5,
    4_000_000: 6,
    4_500_000: 7,
}

# Rates worth scanning when hunting for motors: the ones we might have left them at.
SCAN_BAUDS = (1_000_000, 2_000_000, 3_000_000, 57_600, 115_200)

# The PL011 on this Pi derives baud from a 48 MHz clock, so only rates dividing it by
# 16*n are exact. 4 Mbps would need a divisor below 1 and is not reachable.
MAX_SUPPORTED_BAUD = 3_000_000

SERIAL_PORT = "/dev/ttyAMA0"
CANARY_NAME = "head"


def open_bus(baud: int) -> Xl330PyController:
    return Xl330PyController(serial_port=SERIAL_PORT, baudrate=baud, timeout=0.05)


def scan(baud: int) -> list[int]:
    """IDs answering at this rate."""
    try:
        bus = open_bus(baud)
    except Exception as exc:
        print(f"  {baud:>9} : cannot open the port ({exc})")
        return []
    found = []
    for motor_id in MOTOR_TO_ID.values():
        try:
            if bus.ping(motor_id):
                found.append(motor_id)
        except Exception:
            pass
    print(f"  {baud:>9} : {len(found):2d}/{len(MOTOR_TO_ID)} motors" + (f" -> {found}" if found and len(found) < len(MOTOR_TO_ID) else ""))
    return found


def survey() -> dict[int, list[int]]:
    """Where every motor currently is. A healthy bus has them all at one rate."""
    print("Scanning the bus:")
    return {baud: ids for baud in SCAN_BAUDS if (ids := scan(baud))}


def confirm_code_table(bus: Xl330PyController, ids: list[int], baud: int) -> bool:
    """Check our code table against reality before writing to EEPROM.

    The motors are answering at `baud`, so their Baud Rate register must read back as the
    code we believe means `baud`. If it does not, the table is wrong for this firmware and
    writing from it could set a rate nobody can reach.
    """
    expected = BAUD_CODES[baud]
    try:
        codes = bus.sync_read_baud_rate(ids)
    except Exception as exc:
        print(f"  cannot read the baud register: {exc}")
        return False
    codes = [c[0] if isinstance(c, (list, tuple)) else c for c in codes]
    wrong = [(i, c) for i, c in zip(ids, codes) if c != expected]
    if wrong:
        print(f"  motors answer at {baud} but their baud register reads {sorted({c for _, c in wrong})}, not {expected}.")
        print("  The code table in this script does not match this firmware — refusing to write.")
        return False
    print(f"  baud register reads {expected} on all {len(ids)} motors, matching {baud} — code table confirmed.")
    return True


def set_baud(bus: Xl330PyController, ids: list[int], target: int) -> None:
    """Write the new rate. Torque must be off: Baud Rate is an EEPROM register."""
    bus.sync_write_torque_enable(ids, [False] * len(ids))
    time.sleep(0.05)
    for motor_id in ids:
        try:
            bus.write_baud_rate(motor_id, BAUD_CODES[target])
        except Exception as exc:
            # The motor may have switched and answered its status packet at the new rate.
            print(f"    id {motor_id}: write returned {exc!r} (it may still have switched)")
    time.sleep(0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Set the motor bus baud rate (EEPROM).")
    parser.add_argument("baud", type=int, help=f"target rate, one of {sorted(BAUD_CODES)}")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args()

    target = args.baud
    if target not in BAUD_CODES:
        raise SystemExit(f"Unknown baud {target}. Known: {sorted(BAUD_CODES)}")
    if target > MAX_SUPPORTED_BAUD:
        raise SystemExit(
            f"{target} is above what this Pi's PL011 can generate ({MAX_SUPPORTED_BAUD} max "
            f"from its 48 MHz clock). The motors would switch and become unreachable."
        )

    print(f"Target: {target} baud (code {BAUD_CODES[target]})\n")
    found = survey()
    if not found:
        raise SystemExit("\nNo motors answered at any known rate. Check power and wiring.")

    everywhere = {b: ids for b, ids in found.items() if b != target}
    already = found.get(target, [])
    print()
    if len(already) == len(MOTOR_TO_ID):
        print(f"All {len(already)} motors are already at {target}. Nothing to do.")
        print(f"Make sure constants.MOTOR_BAUDRATE is {target} (currently {MOTOR_BAUDRATE}).")
        return
    if already:
        print(f"Note: {len(already)} motor(s) are already at {target}; {sum(len(v) for v in everywhere.values())} still to move.")

    if len(found) > 1:
        print("The bus is split across rates — probably an interrupted run. This will bring it back together.")

    total_to_move = sum(len(v) for v in everywhere.values())
    canary_id = MOTOR_TO_ID[CANARY_NAME]
    canary_baud = next((b for b, ids in everywhere.items() if canary_id in ids), None)
    source_rates = ", ".join(str(b) for b in sorted(everywhere))

    # --- Stage 1: the head alone, unless a previous run already got it to the target.
    if canary_id in already:
        # The head answered the survey at `target`, so the rate is already proven on this
        # Pi and the head is reachable — exactly what the trial checks. Resume with the rest.
        print(f"\n'{CANARY_NAME}' is already at {target} (an earlier run got this far), so the rate is")
        print("already proven on this Pi. Skipping the trial and resuming with the remaining motors.")
    elif canary_baud is not None:
        print(f"\n=== Step 1 of 2: try '{CANARY_NAME}' (id {canary_id}) alone ===")
        print(f"'{CANARY_NAME}' is the only joint no policy drives. It moves from {canary_baud} to {target}")
        print(f"first; if {target} does not work on this Pi, only the head is stranded, not the other")
        print(f"{len(MOTOR_TO_ID) - 1} motors. The remaining motors are a separate confirmation, after this works.")
        if not args.yes:
            print("\nThis writes the head's EEPROM (not undoable without a working bus).")
            if input(f"Type 'yes' to switch ONLY '{CANARY_NAME}' to {target}: ").strip().lower() != "yes":
                raise SystemExit("Aborted. Nothing was written.")

        bus = open_bus(canary_baud)
        if not confirm_code_table(bus, [canary_id], canary_baud):
            raise SystemExit("Aborted before writing anything.")
        set_baud(bus, [canary_id], target)
        del bus

        try:
            probe = open_bus(target)
        except Exception as exc:
            print(f"  Cannot open the port at {target}: {exc}")
            raise SystemExit(f"  '{CANARY_NAME}' is now at {target} and this Pi cannot reach it. Recover with a U2D2.")

        if not probe.ping(canary_id):
            print(f"\n  '{CANARY_NAME}' does not answer at {target}.")
            print(f"  The other {len(MOTOR_TO_ID) - 1} motors are untouched and still work at {canary_baud}.")
            print(f"  {target} is not usable on this Pi/wiring — recover the head with a U2D2, or try 1000000.")
            raise SystemExit(1)
        print(f"  '{CANARY_NAME}' answers at {target}. The rate works on this Pi.")
        del probe
    else:
        # The head is neither at the target nor at any source rate — it did not answer at
        # all. Without the trial motor, refuse to mass-convert the bus blind.
        raise SystemExit(
            f"\n'{CANARY_NAME}' did not answer at any known rate, so it cannot be the trial motor. "
            f"Check it is connected/powered (or recover it with a U2D2), then re-run. Aborting so the "
            f"other motors are not converted without first proving {target} on this Pi."
        )

    remaining = total_to_move if canary_id in already else total_to_move - 1
    if remaining <= 0:
        print(f"\nOnly the head needed moving. Set MOTOR_BAUDRATE = {target} in src/constants.py (it is {MOTOR_BAUDRATE}).")
        return

    # --- Stage 2: the rest, now that the rate is proven. Confirmed separately.
    print(f"\n=== Step 2 of 2: switch the remaining {remaining} motor(s) to {target} ===")
    if not args.yes:
        print(f"The rate is proven on this Pi ('{CANARY_NAME}' at {target}). This now writes EEPROM on the rest.")
        if input(f"Type 'yes' to switch the other {remaining} motor(s): ").strip().lower() != "yes":
            print(f"\nStopped. The bus is split: '{CANARY_NAME}' at {target}, {remaining} motor(s) at {source_rates}.")
            print(f"Re-run `make set-baud BAUD={target}` to finish, or `BAUD={source_rates}` to go back.")
            raise SystemExit("Aborted before the remaining motors.")

    # --- The rest, grouped by wherever they currently are.
    for baud, ids in everywhere.items():
        rest = [i for i in ids if i != canary_id]
        if not rest:
            continue
        print(f"\n=== Moving {len(rest)} motor(s) from {baud} to {target} ===")
        bus = open_bus(baud)
        if not confirm_code_table(bus, rest, baud):
            raise SystemExit("Aborted; the bus is now split. Re-run to repair.")
        set_baud(bus, rest, target)
        del bus

    # --- Verify.
    print(f"\n=== Verifying at {target} ===")
    final = survey()
    at_target = final.get(target, [])
    if len(at_target) == len(MOTOR_TO_ID):
        bus = open_bus(target)
        positions = bus.sync_read_present_position(list(MOTOR_TO_ID.values()))
        print(f"\n  All {len(at_target)} motors answer at {target} and return positions.")
        print(f"  Sample: {[round(float(p), 3) for p in positions[:4]]} ...")
        print(f"\n  Now set MOTOR_BAUDRATE = {target} in src/constants.py (it is {MOTOR_BAUDRATE}),")
        print("  then `make run`. Check the read time with [p]; `make check-read` times it precisely.")
    else:
        missing = [n for n, i in MOTOR_TO_ID.items() if i not in at_target]
        print(f"\n  Only {len(at_target)}/{len(MOTOR_TO_ID)} answer at {target}. Missing: {missing}")
        print("  The bus is split. Re-run this command to repair, or go back with BAUD=1000000.")
        sys.exit(1)


if __name__ == "__main__":
    main()
