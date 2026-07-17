# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Interaction logic shared by the robot (main.py) and the simulation (sim/sim_main.py).

Input sources differ only in how raw events reach them: escape sequences from a raw
terminal, GLFW keycodes from the MuJoCo viewer, or joystick events from /dev/input/js*.
What a key or button *does* is defined once here, so `make run` and `make sim` react
identically to the same key and can be compared face to face.

A transport maps its raw codes onto the normalized key names below and calls
handle_key(); it never decides what a binding means.
"""

import os
import threading
from pathlib import Path

from input.input_source import InputSource, UserInput

STOP_FLAG_PATH = "/tmp/microban_scheduler.stop"

VELOCITY_STEP = 0.1
VELOCITY_MAX = 1.0

# Which moves can be toggled, and what toggles them. Single source of truth: change a
# binding here and both the robot and the simulation follow.
MOVE_KEYS = {"h": "head", "s": "squat", "v": "walk", "z": "squat_rl"}
GAMEPAD_BUTTON_MOVES = {"A": "walk"}

# Normalized key names.
KEY_UP = "up"
KEY_DOWN = "down"
KEY_LEFT = "left"
KEY_RIGHT = "right"
KEY_RESET_VELOCITY = "x"
KEY_IMU = "i"
KEY_STOP = "q"
KEY_LOG = "l"


class InputActions(InputSource):
    """Thread-safe UserInput state plus the actions every transport drives it with.

    Args:
        stop_flag_path: path to the stop flag file polled by the scheduler.
        line_ending: "\\r\\n" for sources printing from a raw-mode terminal, "\\n" otherwise.
    """

    def __init__(self, stop_flag_path: str = STOP_FLAG_PATH, line_ending: str = "\r\n") -> None:
        self._stop_flag_path = Path(stop_flag_path)
        self._line_ending = line_ending
        self._state = UserInput()
        self._lock = threading.Lock()

    def read(self) -> UserInput:
        with self._lock:
            return UserInput(
                active_moves=set(self._state.active_moves),
                velocity=dict(self._state.velocity),
                show_imu=self._state.show_imu,
                logging=self._state.logging,
                log_name=self._state.log_name,
            )

    def notify(self, message: str) -> None:
        print(message, end=self._line_ending, flush=True)

    # ------------------------------------------------------------------
    # Actions — what a binding does, independent of how it was triggered.

    def toggle_move(self, move_name: str) -> None:
        with self._lock:
            if move_name in self._state.active_moves:
                self._state.active_moves.discard(move_name)
                enabled = False
            else:
                self._state.active_moves.add(move_name)
                enabled = True
        self.notify(f"Move '{move_name}' {'enabled' if enabled else 'disabled'}")

    def adjust_velocity(self, axis: str, delta: float) -> None:
        """Nudge an axis by a step (discrete keys)."""
        with self._lock:
            velocity = self._state.velocity
            velocity[axis] = max(-VELOCITY_MAX, min(VELOCITY_MAX, velocity.get(axis, 0.0) + delta))
            value = velocity[axis]
        self.notify(f"{axis}={value:.1f}")

    def set_velocity(self, axis: str, norm: float) -> None:
        """Set a normalized axis command outright (analog sticks); scale_velocity()
        applies the physical limits later, in the scheduler."""
        with self._lock:
            self._state.velocity[axis] = max(-VELOCITY_MAX, min(VELOCITY_MAX, norm))

    def zero_velocity(self) -> None:
        with self._lock:
            self._state.velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}

    def reset_velocity(self) -> None:
        self.zero_velocity()
        self.notify("Velocity reset to zero")

    def toggle_imu(self) -> bool:
        """Toggle the IMU display and return the new state (subclasses may hook this)."""
        with self._lock:
            self._state.show_imu = not self._state.show_imu
            show = self._state.show_imu
        self.notify(f"IMU display {'enabled' if show else 'disabled'}")
        return show

    def request_stop(self) -> None:
        self._stop_flag_path.write_text("stop\n", encoding="ascii")
        self.notify("Stop requested")

    def prompt_log_name(self) -> str | None:
        """Ask for an optional log name suffix. Return "" for none, None to cancel.

        Hook: only a source owning a terminal can actually prompt (the keyboard does).
        The default answers "no name", so a gamepad would log under a plain date stamp.
        """
        return ""

    def toggle_logging(self) -> None:
        """Start a log session (asking for an optional name first) or stop the running one.

        The scheduler owns the file; it watches UserInput.logging and starts/stops
        RobotLogger on the transition.
        """
        with self._lock:
            running = self._state.logging

        if running:
            with self._lock:
                self._state.logging = False
            self.notify("Logging stopped")
            return

        # Prompt outside the lock: it blocks on terminal input for as long as the user types.
        name = self.prompt_log_name()
        if name is None:
            self.notify("Logging cancelled")
            return

        with self._lock:
            self._state.log_name = name
            self._state.logging = True
        self.notify(f"Logging started{f' ({name})' if name else ''}")


def handle_key(actions: InputActions, key: str, move_keys: dict[str, str]) -> bool:
    """Apply the shared binding for a normalized key name.

    Returns False when the key is unbound, so a transport can fall through to keys that
    only make sense for it (the MuJoCo viewer adds [r] and [t]).
    """
    if key in move_keys:
        actions.toggle_move(move_keys[key])
    elif key == KEY_RESET_VELOCITY:
        actions.reset_velocity()
    elif key == KEY_IMU:
        actions.toggle_imu()
    elif key == KEY_LOG:
        actions.toggle_logging()
    elif key == KEY_STOP:
        actions.request_stop()
    elif key == KEY_UP:
        actions.adjust_velocity("vx", +VELOCITY_STEP)
    elif key == KEY_DOWN:
        actions.adjust_velocity("vx", -VELOCITY_STEP)
    # Left is +vtheta (CCW), matching VTHETA_SIGN on the gamepad's right stick.
    elif key == KEY_LEFT:
        actions.adjust_velocity("vtheta", +VELOCITY_STEP)
    elif key == KEY_RIGHT:
        actions.adjust_velocity("vtheta", -VELOCITY_STEP)
    else:
        return False
    return True


def help_lines(move_keys: dict[str, str]) -> list[str]:
    """Help text for the keys handle_key() binds, generated from the same table."""
    return [
        *(f"  [{key}]       toggle move '{name}'" for key, name in move_keys.items()),
        f"  [{KEY_IMU}]       toggle IMU/gyro display",
        f"  [{KEY_LOG}]       start/stop logging (asks for an optional name)",
        "  [arrows]  vx (up/down), vtheta (left/right)",
        f"  [{KEY_RESET_VELOCITY}]       reset velocity to zero",
        f"  [{KEY_STOP}]       stop scheduler",
    ]


def build_input_source() -> InputSource:
    """Use the gamepad when one is connected, otherwise fall back to the keyboard.

    Override with MICROBAN_INPUT=keyboard|gamepad. The robot and the simulation both call
    this and nothing else, so they are driven by the very same two input sources.
    """
    requested = os.environ.get("MICROBAN_INPUT", "auto").lower()

    if requested in ("auto", "gamepad"):
        from input.gamepad_input import GamepadInputSource, find_gamepad_path

        if find_gamepad_path() is not None:
            return GamepadInputSource(button_moves=GAMEPAD_BUTTON_MOVES)
        if requested == "gamepad":
            raise RuntimeError("MICROBAN_INPUT=gamepad but no gamepad was found.")
        print("No gamepad detected; using keyboard input.")

    from input.keyboard_input import KeyboardInputSource

    return KeyboardInputSource(move_keys=MOVE_KEYS)
