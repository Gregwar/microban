# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Per-motor accounting of motor-bus faults, reported by [p].

The aggregate counters ("3 packets missing") say a fault happened but not where, which is
the one thing needed to act on it: a single motor faulting points at its wire or its place
in the daisy chain, faults spread evenly over all 19 point at the baud rate, the supply or
electrical noise. So every failure is classified and, where the protocol allows, pinned to
a motor.

HOW A MOTOR CAN BE NAMED. rustypot's sync_read sends one broadcast instruction and then
reads one status packet per requested id, *in the order they were requested*, aborting the
whole transaction on the first one that does not parse. Which motor was at fault is
therefore recoverable from the error it raises:

    Incorrect id (A instead of B)   the motor whose turn it was stayed silent, so the read
                                    landed on the *next* motor's status packet. Both ids
                                    are named, and the one requested earlier is the silent
                                    one -- see attribute_incorrect_id(), which reads that
                                    off the request order rather than off the argument
                                    order in rustypot's message (whose two operands are
                                    swapped with respect to its own wording).
    Checksum error                  a status packet arrived but was corrupted on the wire.
    Parsing error                   a status packet arrived with an impossible structure.
    Timeout error                   nothing arrived at all -- the last motor of the request
                                    went silent, several went silent at once, or the port
                                    could not be cleared before sending.

The last three carry no id, because rustypot raises before it knows whose packet it was
reading. They are counted as unattributed, and MICROBAN_BUS_PROBE=1 pings every motor after
such a failure to name it (off by default: 19 extra round trips inside a blown 20 ms tick).

Malformed blocks are the exception: those are caught by our own decoder in
RobotController.sync_read_state, which is iterating over (id, block) pairs and so always
knows the motor.

Kept free of any hardware import so it can be exercised without a robot.
"""

import re
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

from constants import ID_TO_MOTOR

# rustypot surfaces CommunicationErrorKind as a PyRuntimeError carrying its Display text.
_INCORRECT_ID_RE = re.compile(r"[Ii]ncorrect id \((\d+) instead of (\d+)\)")

# Fault kinds, in the order the report lists them.
KIND_SILENT = "silent"          # a motor did not answer; the next motor's frame arrived
KIND_TIMEOUT = "timeout"        # nothing came back before the port timeout
KIND_CHECKSUM = "checksum"      # a frame arrived corrupted (CRC mismatch)
KIND_PARSING = "parsing"        # a frame arrived with an impossible structure
KIND_MALFORMED = "malformed"    # a block that our own decoder could not un-stuff/unpack
KIND_OTHER = "other"

KINDS = (KIND_SILENT, KIND_TIMEOUT, KIND_CHECKSUM, KIND_PARSING, KIND_MALFORMED, KIND_OTHER)

# How many individual faults the report can quote. Only the tail is kept: a session that
# faults thousands of times is diagnosed from the per-motor tallies, not from a transcript.
HISTORY = 24


def motor_name(motor_id: int | None) -> str:
    """"left_knee (14)" for a known id, "id 14" for one that is not in the map."""
    if motor_id is None:
        return "?"
    name = ID_TO_MOTOR.get(motor_id)
    return f"{name} ({motor_id})" if name else f"id {motor_id}"


def attribute_incorrect_id(first: int, second: int, ids: list[int]) -> tuple[int | None, int | None]:
    """Split an "Incorrect id (A instead of B)" into (silent motor, motor that answered).

    Decided from the request order, not from the operand order: rustypot builds the error
    as IncorrectId(received, expected) but its Display prints "{expected} instead of
    {received}", so the wording cannot be trusted to say which is which. What is certain is
    that sync_read walks `ids` in order, so the motor it was waiting for comes *before* the
    one whose packet it actually read.

    When only one of the two is in `ids`, that one is the motor we were waiting for and the
    other is whatever the wire produced (garbage, or a stale frame). When neither is,
    nothing can be attributed.
    """
    pos_first = ids.index(first) if first in ids else None
    pos_second = ids.index(second) if second in ids else None

    if pos_first is not None and pos_second is not None:
        return (first, second) if pos_first < pos_second else (second, first)
    if pos_first is not None:
        return first, second
    if pos_second is not None:
        return second, first
    return None, None


def classify(message: str, ids: list[int]) -> tuple[str, int | None, int | None]:
    """Map one driver exception to (kind, motor at fault, motor that answered instead)."""
    match = _INCORRECT_ID_RE.search(message)
    if match:
        silent, responder = attribute_incorrect_id(int(match.group(1)), int(match.group(2)), ids)
        return KIND_SILENT, silent, responder

    lowered = message.lower()
    if "checksum" in lowered:
        return KIND_CHECKSUM, None, None
    if "parsing" in lowered:
        return KIND_PARSING, None, None
    if "timeout" in lowered or "timed out" in lowered:
        return KIND_TIMEOUT, None, None
    return KIND_OTHER, None, None


@dataclass
class BusFault:
    """One failed transaction, as the report quotes it."""

    t: float                    # seconds since the monitor started
    kind: str
    op: str                     # the read/write that failed, e.g. "sync_read_state"
    motor: int | None           # the motor at fault, when the protocol allowed naming it
    responder: int | None       # the motor that answered in its place (KIND_SILENT only)
    detail: str                 # the driver's own message

    def describe(self) -> str:
        who = motor_name(self.motor) if self.motor is not None else "unattributed"
        line = f"t={self.t:7.1f}s  {self.op:<22} {self.kind:<9} {who}"
        if self.responder is not None:
            line += f"  <- {motor_name(self.responder)} answered instead"
        elif self.motor is None:
            line += f"  ({self.detail})"
        return line


class BusMonitor:
    """Traffic counters plus the per-motor fault tallies behind the [p] report.

    Counts packets, not calls: one sync transaction is a single broadcast instruction
    packet, answered by one status packet per motor. Writes are not counted as expecting an
    answer -- main.py sets status_return_level to 1, so the motors only reply to reads.
    """

    def __init__(self, history: int = HISTORY) -> None:
        self.sent = 0           # instruction packets put on the wire
        self.received = 0       # status packets that came back and were handed to us
        self.expected = 0       # status packets that should have come back
        self.errors = 0         # transactions that raised
        self.malformed = 0      # status payloads that would not decode
        self.fallbacks = 0      # fused reads that had to retry as separate typed reads

        self.by_kind: Counter = Counter()
        # motor id -> Counter of kinds. Only the faults the protocol let us attribute.
        self.by_motor: dict[int, Counter] = defaultdict(Counter)
        self.last_seen: dict[int, float] = {}
        # Motors that answered in a silent motor's slot. Not a fault of theirs, but a
        # neighbour that shows up here repeatedly confirms the daisy-chain ordering.
        self.answered_instead: Counter = Counter()
        self.unattributed: Counter = Counter()
        self.recent: deque[BusFault] = deque(maxlen=history)
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Recording

    def record_sent(self, expected: int = 0) -> None:
        self.sent += 1
        self.expected += expected

    def record_received(self, count: int) -> None:
        self.received += count

    def record_exception(
        self, op: str, ids: list[int], exc: BaseException, attribute_to: int | None = None
    ) -> BusFault:
        """Classify a failed transaction and pin it to a motor where the protocol allows.

        `attribute_to` names the motor when the caller already knows it and the frame does
        not — a single-motor read, where whatever went wrong can only have been that one's.
        """
        self.errors += 1
        kind, motor, responder = classify(str(exc), ids)
        if motor is None:
            motor = attribute_to
        return self._add(BusFault(self._now(), kind, op, motor, responder, str(exc)))

    def record_malformed(self, op: str, motor_id: int, detail: str) -> BusFault:
        """A block that came back the wrong width. The caller knows exactly whose it was."""
        self.malformed += 1
        return self._add(BusFault(self._now(), KIND_MALFORMED, op, motor_id, None, detail))

    def record_probe(self, silent_ids: list[int], op: str) -> None:
        """Attribute an otherwise-anonymous fault to the motors a follow-up ping missed.

        Rewrites the fault just recorded rather than adding new ones: the transaction failed
        once, and the ping only answers *whose* fault it was.
        """
        if not silent_ids or not self.recent:
            return
        fault = self.recent[-1]
        # It is attributed now, so drop it from the unattributed tally — and drop the key
        # outright at zero, or the report would list a kind with a count of none.
        if self.unattributed.get(fault.kind):
            self.unattributed[fault.kind] -= 1
            if not self.unattributed[fault.kind]:
                del self.unattributed[fault.kind]
        fault.motor = silent_ids[0]
        fault.detail += f"; ping found {len(silent_ids)} silent"
        for motor_id in silent_ids:
            self.by_motor[motor_id][fault.kind] += 1
            self.last_seen[motor_id] = fault.t
        _ = op

    def record_fallback(self) -> None:
        self.fallbacks += 1

    def _add(self, fault: BusFault) -> BusFault:
        self.by_kind[fault.kind] += 1
        if fault.motor is not None:
            self.by_motor[fault.motor][fault.kind] += 1
            self.last_seen[fault.motor] = fault.t
        else:
            self.unattributed[fault.kind] += 1
        if fault.responder is not None:
            self.answered_instead[fault.responder] += 1
        self.recent.append(fault)
        return fault

    def _now(self) -> float:
        return time.perf_counter() - self._t0

    # ------------------------------------------------------------------
    # Reporting

    @property
    def missing(self) -> int:
        """Status packets whose data never reached us.

        A failed sync read loses *all* of its status packets, not just the faulting one:
        rustypot aborts the transaction and discards whatever it had already read. So one
        silent motor costs a whole tick's worth of state, which is what this counts.
        """
        return max(0, self.expected - self.received)

    @property
    def error_total(self) -> int:
        return self.errors + self.malformed

    def snapshot(self) -> dict:
        """Everything [p] prints, as plain data (the simulation has no monitor at all)."""
        return {
            "sent": self.sent,
            "received": self.received,
            "expected": self.expected,
            "missing": self.missing,
            "errors": self.errors,
            "malformed": self.malformed,
            "fallbacks": self.fallbacks,
            "error_total": self.error_total,
            "by_kind": dict(self.by_kind),
            "by_motor": {motor_id: dict(kinds) for motor_id, kinds in self.by_motor.items()},
            "last_seen": dict(self.last_seen),
            "answered_instead": dict(self.answered_instead),
            "unattributed": dict(self.unattributed),
            "recent": list(self.recent),
        }

    def report_lines(self, max_recent: int = 5) -> list[str]:
        """The per-motor breakdown, ready to print. Empty when the bus has been clean."""
        if not self.by_kind:
            return []

        lines: list[str] = []
        ranked = sorted(
            self.by_motor.items(),
            key=lambda item: (-sum(item[1].values()), item[0]),
        )
        if ranked:
            lines.append("  faults by motor")
            for motor_id, kinds in ranked:
                breakdown = "  ".join(
                    f"{kind}x{count}" for kind, count in sorted(kinds.items(), key=lambda k: -k[1])
                )
                seen = self.last_seen.get(motor_id)
                when = f"last t={seen:.1f}s" if seen is not None else ""
                lines.append(f"    {motor_name(motor_id):<24} {breakdown:<34} {when}")

        if self.answered_instead:
            neighbours = "  ".join(
                f"{motor_name(motor_id)}x{count}"
                for motor_id, count in self.answered_instead.most_common()
            )
            lines.append(f"    answered in a silent slot: {neighbours}")

        if self.unattributed:
            detail = "  ".join(f"{kind}x{count}" for kind, count in self.unattributed.most_common())
            lines.append(f"  unattributed  {detail}  (no id in the frame that failed)")
            lines.append("                MICROBAN_BUS_PROBE=1 pings after a failure to name the motor")

        if max_recent and self.recent:
            shown = min(max_recent, len(self.recent))
            lines.append("  last fault" if shown == 1 else f"  last {shown} faults")
            for fault in list(self.recent)[-max_recent:]:
                lines.append(f"    {fault.describe()}")

        return lines
