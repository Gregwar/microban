# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Decoding for the fused XL330 state read.

Present Current, Present Velocity and Present Position sit in one contiguous block of the
XL330 RAM, so a single sync read fetches all three where the driver's per-register calls
would cost one bus round trip each — and at 1 Mbps over 19 motors, those round trips are
most of a 20 ms control tick.

Kept free of any hardware import so the decoding can be tested without a robot. The
addresses below are from the XL330 control table and are checked against the driver's own
readings at startup (see RobotController) rather than trusted blindly.
"""

import math
import struct

# Contiguous RAM block: Present Current (126, 2 B, i16), Present Velocity (128, 4 B, i32),
# Present Position (132, 4 B, i32).
STATE_ADDR = 126
STATE_LEN = 10
_STATE_STRUCT = struct.Struct("<hii")

# Present Position is an absolute encoder count over one turn, centred at half scale.
POSITION_TICKS_PER_TURN = 4096
POSITION_CENTER_TICKS = 2048

# Present Velocity unit is 0.229 rpm/LSB.
VELOCITY_UNIT_RAD_S = 0.229 * math.pi / 30.0


def unstuff(data: bytes) -> bytes:
    """Undo Protocol 2.0 byte stuffing.

    To keep payload data from being mistaken for a packet header, the sender inserts an
    extra 0xFD after any 0xFF 0xFF 0xFD in the payload. This removes it. Verified against
    real motor frames captured while joints were back-driven (see the capture in the
    fused-read work): the negative velocity/current values seen in motion routinely produce
    the trigger, e.g. `00 00 ff ff ff ff fd FD 08 00 00` -> `00 00 ff ff ff ff fd 08 00 00`.

    Safe to run on every frame: a genuine STATE_LEN frame cannot contain 0xFF 0xFF 0xFD (if
    the data did, it would have been stuffed and arrived longer), so on clean frames this is
    a no-op.
    """
    if b"\xff\xff\xfd\xfd" not in data:
        return data  # fast path: no stuffing present (the overwhelming majority of frames)
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        out.append(data[i])
        if (out[-1] == 0xFD and len(out) >= 3 and out[-2] == 0xFF and out[-3] == 0xFF
                and i + 1 < n and data[i + 1] == 0xFD):
            i += 1  # skip the inserted stuffing byte
        i += 1
    return bytes(out)


def decode_state_block(block) -> tuple[int, int, int]:
    """Split one motor's block into raw (current, velocity, position ticks).

    Byte stuffing is undone first, so a stuffed frame decodes on the fast path rather than
    forcing a fallback. Anything that is still not STATE_LEN afterwards (a short read, a
    doubly-corrupted frame) raises, and the caller falls back to the driver's typed reads.

    The driver hands the block back as a sequence of byte values; accept anything
    bytes-like so a list, bytearray or bytes all work.
    """
    data = unstuff(bytes(bytearray(block)))
    if len(data) != STATE_LEN:
        raise ValueError(f"Expected a {STATE_LEN}-byte state block, got {len(data)} bytes")
    return _STATE_STRUCT.unpack(data)


def ticks_to_radians(ticks: int) -> float:
    """Present Position ticks to radians, matching the driver's own conversion."""
    return (ticks - POSITION_CENTER_TICKS) * 2.0 * math.pi / POSITION_TICKS_PER_TURN


def radians_to_ticks(radians: float) -> int:
    """Inverse of ticks_to_radians (used by the tests)."""
    return round(radians * POSITION_TICKS_PER_TURN / (2.0 * math.pi)) + POSITION_CENTER_TICKS


def encode_state_block(current_raw: int, velocity_raw: int, position_ticks: int) -> bytes:
    """Build a block as a motor would return it. Only used to test the decoder."""
    return _STATE_STRUCT.pack(current_raw, velocity_raw, position_ticks)
