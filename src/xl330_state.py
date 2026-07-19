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

# Present Input Voltage (144, 2 B, u16) sits ten bytes past the end of that block, with
# Velocity Trajectory (136, 4 B) and Position Trajectory (140, 4 B) in between. Reading
# through them costs 10 unwanted bytes per motor — 190 bytes, about 1 ms at 2 Mbps —
# against a whole extra round trip with its own turnaround on all 19 motors. So when
# voltage is wanted, it is cheaper to widen this read than to issue a second one.
STATE_EXT_LEN = 20
_STATE_EXT_STRUCT = struct.Struct("<hiiiiH")

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


def _unstuffed(block, length: int) -> bytes:
    """Un-stuff one motor's block and check it came back the expected width.

    Byte stuffing is undone first, so a stuffed frame decodes on the fast path rather than
    forcing a fallback. Anything that is still the wrong length afterwards (a short read, a
    doubly-corrupted frame) raises, and the caller falls back to the driver's typed reads.

    The driver hands the block back as a sequence of byte values; accept anything
    bytes-like so a list, bytearray or bytes all work.
    """
    data = unstuff(bytes(bytearray(block)))
    if len(data) != length:
        raise ValueError(f"Expected a {length}-byte state block, got {len(data)} bytes")
    return data


def decode_state_block(block) -> tuple[int, int, int]:
    """Split one motor's block into raw (current, velocity, position ticks)."""
    return _STATE_STRUCT.unpack(_unstuffed(block, STATE_LEN))


def decode_state_block_ext(block) -> tuple[int, int, int, int]:
    """Same for the widened block: raw (current, velocity, position ticks, voltage).

    The two trajectory registers read through on the way to the voltage are unpacked and
    dropped — they are the price of the single round trip, not something we want.
    """
    current, velocity, position, _vel_traj, _pos_traj, voltage = _STATE_EXT_STRUCT.unpack(
        _unstuffed(block, STATE_EXT_LEN)
    )
    return current, velocity, position, voltage


def ticks_to_radians(ticks: int) -> float:
    """Present Position ticks to radians, matching the driver's own conversion."""
    return (ticks - POSITION_CENTER_TICKS) * 2.0 * math.pi / POSITION_TICKS_PER_TURN


def radians_to_ticks(radians: float) -> int:
    """Inverse of ticks_to_radians (used by the tests)."""
    return round(radians * POSITION_TICKS_PER_TURN / (2.0 * math.pi)) + POSITION_CENTER_TICKS


def encode_state_block(current_raw: int, velocity_raw: int, position_ticks: int) -> bytes:
    """Build a block as a motor would return it. Only used to test the decoder."""
    return _STATE_STRUCT.pack(current_raw, velocity_raw, position_ticks)


def encode_state_block_ext(
    current_raw: int,
    velocity_raw: int,
    position_ticks: int,
    voltage_raw: int,
    vel_traj: int = 0,
    pos_traj: int = 0,
) -> bytes:
    """Build a widened block as a motor would return it. Only used to test the decoder."""
    return _STATE_EXT_STRUCT.pack(
        current_raw, velocity_raw, position_ticks, vel_traj, pos_traj, voltage_raw
    )
