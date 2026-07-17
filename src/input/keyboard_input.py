# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import os
import sys
import select
import termios
import tty
import threading

from input.actions import (
    KEY_DOWN,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_STOP,
    KEY_UP,
    STOP_FLAG_PATH,
    InputActions,
    handle_key,
    help_lines,
)

_ESCAPE = "\x1b"

# Raw terminal input → the normalized key names the shared bindings use.
_SEQUENCE_TO_KEY = {
    "\x1b[A": KEY_UP,
    "\x1b[B": KEY_DOWN,
    "\x1b[C": KEY_RIGHT,
    "\x1b[D": KEY_LEFT,
    "\x03": KEY_STOP,  # Ctrl+C
}


class KeyboardInputSource(InputActions):
    """Read keyboard input from a raw terminal in a background daemon thread.

    Transport only — which key does what lives in input.actions, shared with the
    simulation's viewer keyboard.

    Args:
        move_keys: mapping from key character to move name, e.g. {"h": "head"}.
        stop_flag_path: path to the stop flag file polled by the scheduler.
    """

    def __init__(
        self,
        move_keys: dict[str, str],
        stop_flag_path: str = STOP_FLAG_PATH,
    ) -> None:
        super().__init__(stop_flag_path=stop_flag_path)
        self._move_keys = move_keys
        self._thread: threading.Thread | None = None
        self._running = False
        self._old_settings: list | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        self._print_help()

    def stop(self) -> None:
        self._running = False
        self._restore_terminal()

    # ------------------------------------------------------------------
    # Internal

    def _read_loop(self) -> None:
        fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self._running:
                # Use select with a timeout so the loop can notice _running=False.
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    continue
                key = self._read_key(fd)
                if key:
                    handle_key(self, _SEQUENCE_TO_KEY.get(key, key), self._move_keys)
        finally:
            self._restore_terminal()

    def prompt_log_name(self) -> str | None:
        """Read an optional log name from the raw terminal. "" if blank, None if cancelled.

        Called from the input thread, so no other binding is dispatched while the user
        types — Ctrl+C therefore stays wired to the stop request, keeping the robot
        stoppable mid-prompt, and Esc cancels. The terminal is already in raw mode, so
        the line is echoed and edited by hand here.
        """
        fd = sys.stdin.fileno()
        buffer = ""
        self.notify("Log name (optional) — Enter to confirm, Esc to cancel:")
        print("> ", end="", flush=True)

        while self._running:
            if not select.select([sys.stdin], [], [], 0.1)[0]:
                continue

            key = self._read_key(fd)
            if key in ("\r", "\n"):
                print("", end="\r\n", flush=True)
                return buffer
            if key == _ESCAPE:
                print("", end="\r\n", flush=True)
                return None
            if key == "\x03":  # Ctrl+C — keep the emergency stop reachable while typing
                print("", end="\r\n", flush=True)
                self.request_stop()
                return None
            if key in ("\x7f", "\x08"):
                buffer = buffer[:-1]
            elif len(key) == 1 and key.isprintable():
                buffer += key
            else:
                continue  # arrows and other escape sequences have no meaning here

            print(f"\r\x1b[K> {buffer}", end="", flush=True)

        return None

    def _read_key(self, fd: int) -> str:
        ch = os.read(fd, 1).decode("utf-8", errors="replace")
        if ch == _ESCAPE:
            # Try to read the CSI sequence that follows (e.g., "[A" for arrow up).
            if select.select([sys.stdin], [], [], 0.05)[0]:
                rest = os.read(fd, 2).decode("utf-8", errors="replace")
                return ch + rest
        return ch

    def _restore_terminal(self) -> None:
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass
            self._old_settings = None

    def _print_help(self) -> None:
        for line in ["Keyboard controls:", *help_lines(self._move_keys)]:
            self.notify(line)
