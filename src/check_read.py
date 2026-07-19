# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Verify and benchmark the fused motor state read, on the real robot.

The fused read reaches into the XL330 control table by raw address, so a wrong offset or
endianness would hand a running policy plausible-looking nonsense. This proves it against
the driver's own per-register reads before you trust it:

    make check-read

Read-only: it never enables torque or writes a goal position, so it is safe to run with
the robot simply sitting on the bench. Leave it still — positions are compared between two
reads taken back to back, and a moving joint would show up as a false mismatch.
"""

import time
import statistics

from constants import MOTOR_TO_ID
from robot_controller import RobotController
from xl330_state import STATE_ADDR, STATE_LEN, decode_state_block, ticks_to_radians

SAMPLES = 50


def bench(label: str, fn, samples: int = SAMPLES) -> float:
    fn()  # warm up
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    mean = statistics.mean(times)
    print(f"  {label:<34} mean={mean:6.2f} ms   median={times[len(times) // 2]:6.2f}   max={times[-1]:6.2f}")
    return mean


def main() -> None:
    names = list(MOTOR_TO_ID.keys())
    ids = list(MOTOR_TO_ID.values())

    # "off" so the constructor does not run its own check before we have reported ours.
    robot = RobotController(fused_read="off")
    driver = robot._controller

    print(f"Fused read: addr {STATE_ADDR}, {STATE_LEN} bytes, {len(ids)} motors\n")

    print("=== 1. Raw block decode vs the driver's own registers ===")
    blocks = driver.sync_read_raw_data(ids, STATE_ADDR, STATE_LEN)
    ref_ticks = driver.sync_read_raw_present_position(ids)
    ref_rad = driver.sync_read_present_position(ids)
    ref_vel = driver.sync_read_present_velocity(ids)
    ref_cur = driver.sync_read_present_current(ids)

    print(f"  driver returned blocks of type {type(blocks).__name__}, first = {list(blocks)[0]!r}\n")
    print(f"  {'motor':<22} {'ticks':>7} {'ref':>7}  {'rad':>8} {'ref':>8}  {'vel_raw':>8} {'ref':>8}  {'cur_raw':>7} {'ref':>7}")

    tick_errors, rad_errors = [], []
    for name, motor_id, block, rt, rr, rv, rc in zip(names, ids, blocks, ref_ticks, ref_rad, ref_vel, ref_cur):
        cur_raw, vel_raw, ticks = decode_state_block(block)
        rad = ticks_to_radians(ticks)
        rc_raw = rc[0] if isinstance(rc, (list, tuple)) else rc
        # The driver applies no sign here — compare in raw hardware terms.
        tick_errors.append(abs(ticks - rt))
        rad_errors.append(abs(rad - rr))
        flag = "" if abs(ticks - rt) <= 2 else "   <-- MISMATCH"
        print(
            f"  {name:<22} {ticks:>7} {rt:>7}  {rad:>8.4f} {rr:>8.4f}  "
            f"{vel_raw:>8} {rv:>8}  {cur_raw:>7} {rc_raw:>7}{flag}"
        )

    print()
    print(f"  position ticks   max deviation: {max(tick_errors)} (expect 0-2 from bus jitter)")
    print(f"  position radians max deviation: {max(rad_errors):.5f} rad  -> conversion {'OK' if max(rad_errors) < 0.01 else 'WRONG'}")
    if max(tick_errors) > 2:
        print("\n  Addressing or endianness is wrong. Do not enable the fused read.")
        return

    print("\n=== 2. Repeatability (robot must be still) ===")
    spreads = []
    for _ in range(10):
        pos, _, _, _ = robot.sync_read_state(ids)
        spreads.append(pos)
    per_motor = [max(s[i] for s in spreads) - min(s[i] for s in spreads) for i in range(len(ids))]
    print(f"  position spread over 10 fused reads: max {max(per_motor):.5f} rad (noise, not drift)")

    print("\n=== 3. Timing: what the tick actually pays ===")
    separate = bench(
        "separate (position + velocity)",
        lambda: (driver.sync_read_present_position(ids), driver.sync_read_present_velocity(ids)),
    )
    separate_cur = bench(
        "separate + current (3 reads)",
        lambda: (
            driver.sync_read_present_position(ids),
            driver.sync_read_present_velocity(ids),
            driver.sync_read_present_current(ids),
        ),
    )
    fused = bench("fused (position+velocity+current)", lambda: robot.sync_read_state(ids))
    separate_volt = bench(
        "separate + voltage (4 reads)",
        lambda: (
            driver.sync_read_present_position(ids),
            driver.sync_read_present_velocity(ids),
            driver.sync_read_present_current(ids),
            driver.sync_read_present_input_voltage(ids),
        ),
    )
    fused_volt = bench(
        "fused + voltage (widened block)",
        lambda: robot.sync_read_state(ids, include_voltage=True),
    )

    print()
    print(f"  fused vs 2 separate reads: {separate - fused:+.2f} ms  ({100 * (separate - fused) / separate:+.0f}%)")
    print(
        f"  widened vs separate voltage: {separate_volt - fused_volt:+.2f} ms  "
        f"({100 * (separate_volt - fused_volt) / separate_volt:+.0f}%)"
    )
    print(f"  cost of widening the block: {fused_volt - fused:+.2f} ms")
    print(f"  fused vs 3 separate reads: {separate_cur - fused:+.2f} ms  ({100 * (separate_cur - fused) / separate_cur:+.0f}%)")
    print("  budget at 50 Hz is 20.00 ms per tick")

    print("\n=== 4. Return delay time (2 us/LSB, factory default 250 = 500 us) ===")
    delays = driver.sync_read_return_delay_time(ids)
    delays = [d[0] if isinstance(d, (list, tuple)) else d for d in delays]
    print(f"  values: {sorted(set(delays))}")
    if max(delays) > 0:
        cost_ms = sum(delays) * 2e-3
        print(f"  -> ~{cost_ms:.2f} ms per sync read across all motors; sync_write_return_delay_time(ids, [0]*{len(ids)}) would reclaim it")
    else:
        print("  -> already 0, nothing to reclaim here")

    ok, detail = robot.check_fused_read()
    print(f"\n=== Verdict ===\n  {'PASS' if ok else 'FAIL'} — {detail}")
    if ok:
        print("  The fused read is used automatically (MICROBAN_FUSED_READ=off to disable).")

    robot.shutdown()


if __name__ == "__main__":
    main()
