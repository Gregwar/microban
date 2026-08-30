# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

from observer import Observation
from constants import NEUTRAL_POSE


def onnx_run_name(session, agent_name: str) -> str:
    """`agent_name` plus the training run it was exported from, for the [?] report.

    The training side stamps the run into the ONNX metadata as `run_path`, e.g.
    2026-08-01_23-35-56_vel_m1_4trajdelay_15k. Policy files get renamed and copied
    between machines (src/agents holds a dozen `squat_*.onnx`), so that stamp — not the
    filename — is what says which training run is actually driving the robot.
    """
    try:
        run = session.get_modelmeta().custom_metadata_map.get("run_path")
    except Exception as exc:  # a policy that runs but whose metadata is odd must still report
        run = f"<metadata unreadable: {exc}>"
    return f"{agent_name:<18} {run or '<no run_path in metadata>'}"


class MoveState(Enum):
    INACTIVE = auto()
    STARTING = auto()
    ACTIVE = auto()
    STOPPING = auto()


@dataclass
class MotorCommand:
    """
    Final motor command built by the scheduler pipeline.
    
    Initialized with the neutral pose.
    """

    target_angles: dict[str, float] = field(default_factory=lambda: dict(NEUTRAL_POSE))


class Move(ABC):
    """Base class for all motion behaviors."""

    # True for moves driven by an RL policy. The scheduler stamps policy_t0 in the log
    # when one of these goes active, so runs of the same manoeuvre can be time-aligned.
    is_policy: bool = False

    # Trunk height [m] this move is commanding on the current tick, or None when it commands
    # none. A height-tracking policy writes its target here each tick; the
    # scheduler records it as the log's `command.height` channel, so the commanded height can
    # be read against the trunk height the run actually produced.
    height_target_command: float | None = None

    def __init__(self) -> None:
        self.state: MoveState = MoveState.INACTIVE

    def preload(self) -> None:
        """Called before the control loop starts. Override to load heavy resources."""

    def describe(self) -> str | None:
        """One line saying what this move actually runs, printed by [?].

        Policy moves return their ONNX and the training run behind it (onnx_run_name);
        None means there is nothing to report and the move is listed as unknown.
        """
        return None

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        """Called each tick while state is STARTING.
        Must set self.state = MoveState.ACTIVE when the transition is done."""
        self.state = MoveState.ACTIVE

    @abstractmethod
    def step(self, obs: Observation, command: MotorCommand) -> None:
        """Called each tick while state is ACTIVE."""

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        """Called each tick while state is STOPPING.
        Must set self.state = MoveState.INACTIVE when the transition is done."""
        self.state = MoveState.INACTIVE