# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import time

from constants import MOTOR_TO_ID, NEUTRAL_POSE
from controller import ControllerProtocol
from observer import Observation
from moves.move import MotorCommand, Move, MoveState

# How long the robot takes to come back to the neutral pose when the move is switched off.
# Releasing usually leaves it slumped or lying down, and the scheduler rebuilds a full
# NEUTRAL_POSE command every tick: re-enabling torque without a ramp would snap every joint
# from wherever gravity put it straight to neutral. Same duration as main.ramp_to_neutral(),
# which solves the same problem at startup.
RECOVER_DURATION_S = 2.0


class ReleasedMove(Move):
    """Cut torque on every motor and leave the robot limp, until the move is switched off.

    While active, the move keeps overwriting the command with the *measured* joint
    positions rather than clearing it. Two reasons, both about not fighting the rest of
    the loop:

    - the scheduler only skips the bus write when ``target_angles`` is empty, and in
      simulation that write is what steps the physics — an empty command would freeze the
      viewer instead of dropping the robot;
    - goal position stays where the robot actually is, so nothing has moved under the
      servos by the time torque comes back.

    Register it *last* in the scheduler's move dict: moves are dispatched in registration
    order, so being last is what lets it override a policy that is still active alongside
    it (releasing while walking should go limp, not walk).

    In simulation ``sync_write_torque_enable`` is a no-op, so the robot does not actually
    go limp there; it follows its own measured position, which sags under gravity — close
    enough to inspect the move, not a substitute for trying it on the robot.
    """

    def __init__(self, controller: ControllerProtocol | None = None) -> None:
        super().__init__()
        self._controller = controller
        # Set on the first STOPPING tick: when the ramp back to neutral started, and the
        # pose it started from. None while the ramp is not running.
        self._recover_start_s: float | None = None
        self._recover_from: dict[str, float] = {}

    def describe(self) -> str:
        return "torque disabled on all motors"

    def _motor_ids(self) -> list[int]:
        return list(MOTOR_TO_ID.values())

    def _measured_pose(self, obs: Observation) -> dict[str, float]:
        """Current joint positions, falling back to neutral for a motor that did not read."""
        positions = obs.robot_state.motor_positions
        return {name: positions.get(name, NEUTRAL_POSE[name]) for name in MOTOR_TO_ID}

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        if self._controller is not None:
            ids = self._motor_ids()
            self._controller.sync_write_torque_enable(ids, [False] * len(ids))
        self._recover_start_s = None
        command.target_angles.update(self._measured_pose(obs))
        self.state = MoveState.ACTIVE

    def step(self, obs: Observation, command: MotorCommand) -> None:
        command.target_angles.update(self._measured_pose(obs))

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        """Re-enable torque and ramp back to neutral over RECOVER_DURATION_S.

        Stays in STOPPING (so the scheduler keeps calling this) until the ramp is done.
        Torque is switched on only once the command already holds the measured pose, so
        the servos come back holding the position the robot is in.
        """
        if self._recover_start_s is None:
            self._recover_from = self._measured_pose(obs)
            self._recover_start_s = time.perf_counter()
            command.target_angles.update(self._recover_from)
            if self._controller is not None:
                ids = self._motor_ids()
                self._controller.sync_write_torque_enable(ids, [True] * len(ids))
            return

        progress = min((time.perf_counter() - self._recover_start_s) / RECOVER_DURATION_S, 1.0)
        if progress >= 1.0:
            # Set outright rather than interpolated, so the last command is exactly the
            # neutral pose the scheduler will keep sending once the move is inactive.
            command.target_angles.update(NEUTRAL_POSE)
            self._recover_start_s = None
            self.state = MoveState.INACTIVE
            return

        for name, start in self._recover_from.items():
            command.target_angles[name] = start + progress * (NEUTRAL_POSE[name] - start)
