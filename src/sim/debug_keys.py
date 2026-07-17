# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Sim-only debug keys, read from the MuJoCo viewer window.

Deliberately *not* an InputSource: the simulation drives the robot through the very same
keyboard/gamepad sources as the real robot (input.actions.build_input_source), so the
control interface is identical on both. The two keys here have no robot counterpart —
they poke the simulator itself — and they cannot touch moves or velocity.
"""

import threading

_KEY_RESET_ROBOT = "r"
_KEY_TORQUE = "t"


class SimDebugKeys:
    """[r] resets the robot to its initial pose, [t] toggles the torque sum display.

    Pass ``key_callback`` to ``mujoco.viewer.launch_passive``, and the instance itself to
    ``MuJoCoController(reset_source=...)``, which polls it every tick.
    """

    def __init__(self) -> None:
        self._reset_requested = threading.Event()
        self._show_torque = False
        self._lock = threading.Lock()

    @property
    def show_torque(self) -> bool:
        with self._lock:
            return self._show_torque

    def consume_reset(self) -> bool:
        """Return True (and clear the flag) if a reset was requested since last call."""
        if not self._reset_requested.is_set():
            return False
        self._reset_requested.clear()
        return True

    def key_callback(self, keycode: int) -> None:
        # GLFW reports letters as uppercase ASCII.
        key = chr(keycode).lower() if 32 <= keycode < 127 else ""
        if key == _KEY_RESET_ROBOT:
            self._reset_requested.set()
            self._notify("Robot reset to initial pose")
        elif key == _KEY_TORQUE:
            with self._lock:
                self._show_torque = not self._show_torque
                show = self._show_torque
            self._notify(f"Torque sum display {'enabled' if show else 'disabled'}")

    def help_lines(self) -> list[str]:
        return [
            "Simulation debug keys (press in the MuJoCo viewer window):",
            f"  [{_KEY_TORQUE}]       toggle torque sum display",
            f"  [{_KEY_RESET_ROBOT}]       reset robot to initial pose",
        ]

    @staticmethod
    def _notify(message: str) -> None:
        # The keyboard input source puts the terminal in raw mode, so a bare \n would
        # staircase the output.
        print(message, end="\r\n", flush=True)
