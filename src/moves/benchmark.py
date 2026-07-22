# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import math

from observer import Observation
from moves.move import MotorCommand, Move, MoveState

# Square wave played on the head. Tune here.
#
# The head is the only joint that carries no load and is not part of any policy, so it can
# be stepped hard without disturbing the robot's balance. Everything else stays neutral.
AMPLITUDE_RAD = math.radians(20.0)  # half peak-to-peak: the head alternates between -A and +A
PERIOD_S = 2.0                      # one full low->high->low cycle; keep it long enough that
                                    # the joint fully settles before the next edge

# Held at 0 before the first edge, so the log opens with a flat baseline to measure against.
INITIAL_HOLD_S = 1.0
# Transition in/out of the move, so entering and leaving it is not itself a step.
LERP_DURATION_S = 0.5


class BenchmarkMove(Move):
    """Step the head with a square wave, to measure the delays in the control chain.

    Purely diagnostic: it commands a signal whose edges are instantaneous by construction,
    so any lag between the commanded angle and the read-back angle in the log is delay
    contributed by the robot (bus round trip, motor firmware, mechanical response) rather
    than by the shape of the trajectory.

    Record it with [l] and read it back with src/debug/plot_log.py: tick the `head` joint
    and compare the dashed target against the blue measured position. The horizontal gap
    at an edge is the total loop delay; the rise time after it is the joint's own response.
    """

    # Not a policy, but stamped like one: comparing two benchmark runs is only meaningful
    # once they are lined up on the move start rather than on the moment [l] was hit.
    is_policy = True

    def __init__(self) -> None:
        super().__init__()
        self._lerp_start_time_s: float | None = None
        self._lerp_start_angle: float = 0.0
        self._active_start_time_s: float = 0.0

    def _lerp_to_zero(self, obs: Observation, command: MotorCommand) -> bool:
        """Ease the head to 0. Returns True once it is there."""
        if self._lerp_start_time_s is None:
            self._lerp_start_time_s = obs.robot_state.time_s
            self._lerp_start_angle = obs.robot_state.motor_positions.get("head", 0.0)

        t = min((obs.robot_state.time_s - self._lerp_start_time_s) / LERP_DURATION_S, 1.0)
        command.target_angles["head"] = self._lerp_start_angle * (1.0 - t)

        if t >= 1.0:
            self._lerp_start_time_s = None
            return True
        return False

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        if self._lerp_to_zero(obs, command):
            self._active_start_time_s = obs.robot_state.time_s
            self.state = MoveState.ACTIVE

    def square(self, elapsed_s: float) -> float:
        """Commanded head angle [rad] at *elapsed_s* into the move."""
        if elapsed_s < INITIAL_HOLD_S:
            return 0.0
        phase = (elapsed_s - INITIAL_HOLD_S) % PERIOD_S
        return AMPLITUDE_RAD if phase < PERIOD_S / 2.0 else -AMPLITUDE_RAD

    def step(self, obs: Observation, command: MotorCommand) -> None:
        command.target_angles["head"] = self.square(
            obs.robot_state.time_s - self._active_start_time_s
        )

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        if self._lerp_to_zero(obs, command):
            self.state = MoveState.INACTIVE
