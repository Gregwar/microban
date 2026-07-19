# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import time
import math
from collections import deque
from pathlib import Path
from typing import Optional


from constants import (
    MOTOR_TO_ID,
    BAM_MAX_CURRENT,
    OVERCURRENT_CUTOFF_A,
    OVERCURRENT_DEBOUNCE_TICKS,
    OVERCURRENT_PROXY_DELAY_TICKS,
    PROXY_KT,
    PROXY_R,
    PROXY_VIN,
    PROXY_ERROR_GAIN,
    PROXY_MAX_PWM,
    PROXY_KP,
)
from controller import ControllerProtocol
from imu_reader import imu_quat_to_body
from observer import Observer, Observation
from input.input_source import InputSource, UserInput, scale_velocity
from moves.move import MotorCommand, Move, MoveState
from robot_logger import RobotLogger

# How often the [p] timing report is printed. Averaging over a window beats printing every
# tick: at 50 Hz that would flood the terminal, and the I/O would distort what it measures.
TIMING_PRINT_INTERVAL_S = 0.5
from moves.rotate_head import RotateHeadMove
from moves.squat import SquatMove
from moves.walk import WalkMove


class Scheduler:
    def __init__(
        self,
        frequency_hz: float = 50.0,
        controller: ControllerProtocol = None,
        stop_flag_path: str = "/tmp/microban_scheduler.stop",
        input_source: Optional[InputSource] = None,
        moves: Optional[dict[str, Move]] = None,
    ):
        self.dt = 1.0 / frequency_hz
        self.controller = controller
        self.stop_flag_path = Path(stop_flag_path)
        self._cleanup_done = False
        self.input_source = input_source

        self.observer = Observer(self.controller)
        self.logger = RobotLogger()
        # Name of a policy that went active this tick, consumed by _update_logging (which
        # runs later in the tick, once the logger has had a chance to start).
        self._policy_started: str | None = None

        # All moves are registered here. They only run when activated via user_input.active_moves.
        self.registered_moves: dict[str, Move] = moves if moves is not None else {
            "head": RotateHeadMove(),
            "squat": SquatMove(),
            "walk": WalkMove(),
        }

        self.loop_start_time = time.perf_counter()
        self._serial_errors = 0
        # Cumulative, unlike _serial_errors above, which resets after every good read.
        self._serial_errors_total = 0
        self._last_imu_print_s: float = 0.0
        self._last_imu_stale_warn_s: float = 0.0
        self._timings: dict[str, list[float]] = {}
        self._timing_overruns: int = 0
        # Anchored to now rather than 0.0, so a source that starts with timings already on
        # still reports its first full window instead of a single-tick sample.
        self._last_timing_print_s: float = time.perf_counter()
        self._overcurrent_ticks = 0
        # History of sent target_angles, to align the current proxy with the delayed feedback:
        # the oldest entry is the command issued OVERCURRENT_PROXY_DELAY_TICKS ticks ago.
        self._cmd_history: deque[dict[str, float]] = deque(maxlen=OVERCURRENT_PROXY_DELAY_TICKS + 1)

    def run(self):
        print(f"Starting control loop at {1 / self.dt:.1f} Hz", end="\r\n", flush=True)
        if self.stop_flag_path.exists():
            self.stop_flag_path.unlink()

        if self.input_source:
            self.input_source.start()

        try:
            while True:
                if self.stop_flag_path.exists():
                    print("Stop requested through stop flag", end="\r\n", flush=True)
                    break

                start_time = time.perf_counter()

                # Read robot observations and user input
                try:
                    robot_state = self.observer.read_state(self.dt)
                except RuntimeError as e:
                    self._serial_errors += 1
                    self._serial_errors_total += 1
                    if self._serial_errors >= 3:
                        print(f"Serial communication error: {e}", end="\r\n", flush=True)
                        break
                    print(f"Warning: serial read error (attempt {self._serial_errors}/3): {e}", end="\r\n", flush=True)
                    continue
                self._serial_errors = 0
                read_s = time.perf_counter() - start_time

                robot_state.time_s = start_time - self.loop_start_time
                user_input = self.input_source.read() if self.input_source else UserInput()
                user_input.velocity = scale_velocity(user_input.velocity)
                # Applies from the next tick: the state above was already read with the
                # previous setting. Current comes with the fused block for free, so [u]
                # turns on both telemetry channels rather than only voltage.
                self.observer.observe_voltage = user_input.read_voltage
                self.observer.observe_current = user_input.read_voltage
                obs = Observation(robot_state=robot_state, user_input=user_input)

                imu_status_getter = getattr(self.controller, "get_imu_status", None)
                if callable(imu_status_getter):
                    imu_status = imu_status_getter()
                    age_s = float(imu_status.get("age_s", 0.0))
                    if age_s > self.dt and (start_time - self._last_imu_stale_warn_s) >= 1.0:
                        print(f"Warning: stale IMU data ({age_s * 1000.0:.1f} ms old)", end="\r\n", flush=True)
                        self._last_imu_stale_warn_s = start_time

                # Update move states and dispatch one call per move per tick
                moves_start = time.perf_counter()
                command = MotorCommand()
                for name, move in self.registered_moves.items():
                    in_active = name in obs.user_input.active_moves

                    if in_active and move.state == MoveState.INACTIVE:
                        move.state = MoveState.STARTING
                        if move.is_policy:
                            self._policy_started = name
                    elif not in_active and move.state in (MoveState.STARTING, MoveState.ACTIVE):
                        move.state = MoveState.STOPPING

                    if move.state == MoveState.STARTING:
                        move.on_start(obs, command)
                    elif move.state == MoveState.ACTIVE:
                        move.step(obs, command)
                    elif move.state == MoveState.STOPPING:
                        move.on_stop(obs, command)
                moves_s = time.perf_counter() - moves_start

                # Overcurrent safety: estimate the current from the command that was actually
                # active when the (delayed) position/velocity feedback was sampled — the oldest
                # buffered command (== DELAY_TICKS old once full). This reconstructs the real
                # (delayed) current instead of pairing a fresh target with stale feedback, which
                # would inflate the error term and false-trigger at gait start.
                aligned_targets = self._cmd_history[0] if self._cmd_history else command.target_angles
                if self._check_overcurrent(robot_state, aligned_targets):
                    break

                # Send command to motors
                self._cmd_history.append(dict(command.target_angles))
                send_start = time.perf_counter()
                self._send_to_motors(command)
                send_s = time.perf_counter() - send_start

                self._update_logging(obs, command)

                # Sampled before the displays below, so the report measures the control
                # work rather than the cost of reporting it.
                self._update_timings(
                    obs.user_input.show_timings,
                    start_time,
                    {
                        "read": read_s,
                        "moves": moves_s,
                        "send": send_s,
                        "tick": time.perf_counter() - start_time,
                    },
                )

                # IMU / gyro terminal display
                if obs.user_input.show_imu and (start_time - self._last_imu_print_s) >= 0.5:
                    acc = obs.robot_state.acc
                    gyro = obs.robot_state.gyro
                    quat = obs.robot_state.quat
                    print("--------------------------------------------", end="\r\n", flush=True)
                    if gyro:
                        gx, gy, gz = gyro
                        print(f"Gyro: gx={gx:+.3f}  gy={gy:+.3f}  gz={gz:+.3f} rad/s", end="\r\n", flush=True)
                    if acc:
                        ax, ay, az = acc
                        print(f"Acc:  ax={ax:+.3f}  ay={ay:+.3f}  az={az:+.3f} g", end="\r\n", flush=True)
                    if quat:
                        w, x, y, z = quat
                        roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
                        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
                        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                        print(f"IMU:  roll={roll:+.1f}°  pitch={pitch:+.1f}°  yaw={yaw:+.1f}°", end="\r\n", flush=True)
                        bw, bx, by, bz = imu_quat_to_body((w, x, y, z))
                        b_roll = math.degrees(math.atan2(2 * (bw * bx + by * bz), 1 - 2 * (bx * bx + by * by)))
                        b_pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (bw * by - bz * bx)))))
                        b_yaw = math.degrees(math.atan2(2 * (bw * bz + bx * by), 1 - 2 * (by * by + bz * bz)))
                        print(f"Body: roll={b_roll:+.1f}°  pitch={b_pitch:+.1f}°  yaw={b_yaw:+.1f}°", end="\r\n", flush=True)
                    self._last_imu_print_s = start_time

                # Sleep to keep a fixed control frequency
                elapsed_time = time.perf_counter() - start_time
                sleep_time = self.dt - elapsed_time

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    ms_late = -sleep_time * 1000
                    print(f"Warning: control loop overrun by {ms_late:.2f} ms", end="\r\n", flush=True)

        except KeyboardInterrupt:
            print("Control loop interrupted by user", end="\r\n", flush=True)
        finally:
            self._cleanup()

    def _update_timings(self, show: bool, now: float, phases: dict[str, float]) -> None:
        """Accumulate per-tick phase durations and report them twice a second.

        `read` is the observer (serial + IMU on the robot), `moves` the policy/IK work,
        `send` the bus write (in simulation, the physics step and viewer sync), and `tick`
        the three plus the loop's own overhead — everything before the sleep that keeps
        the loop at its frequency.
        """
        if not show:
            self._timings.clear()
            self._timing_overruns = 0
            # Keep the window anchored to now, so enabling [p] starts a fresh interval.
            self._last_timing_print_s = now
            return

        for name, value in phases.items():
            self._timings.setdefault(name, []).append(value)
        if phases["tick"] > self.dt:
            self._timing_overruns += 1

        if (now - self._last_timing_print_s) < TIMING_PRINT_INTERVAL_S:
            return

        ticks = len(self._timings["tick"])
        print("--------------------------------------------", end="\r\n", flush=True)
        print(
            f"Timings over {ticks} tick{'s' if ticks != 1 else ''} — budget {self.dt * 1000:.1f} ms/tick",
            end="\r\n",
            flush=True,
        )
        for name in ("read", "moves", "send", "tick"):
            values = self._timings.get(name)
            if not values:
                continue
            avg_ms = sum(values) / len(values) * 1000.0
            max_ms = max(values) * 1000.0
            over = f"  ({self._timing_overruns} over budget)" if name == "tick" else ""
            print(f"  {name:<5} avg={avg_ms:6.2f}  max={max_ms:6.2f} ms{over}", end="\r\n", flush=True)

        self._print_bus_stats()

        self._timings.clear()
        self._timing_overruns = 0
        self._last_timing_print_s = now

    def _print_bus_stats(self) -> None:
        """Report bus traffic and communication faults, session-cumulative.

        Counters run from startup rather than over the timing window, so a fault that
        happened once during a run stays visible instead of scrolling away. Silent on
        controllers that do not track them (the simulation).
        """
        getter = getattr(self.controller, "get_bus_stats", None)
        if not callable(getter):
            return

        s = getter()
        loss = (s["missing"] / s["expected"] * 100.0) if s["expected"] else 0.0
        errors = s["error_total"] + self._serial_errors_total

        print(
            f"  packets  sent={s['sent']}  received={s['received']}/{s['expected']}"
            f"  ({loss:.2f}% missing)",
            end="\r\n",
            flush=True,
        )
        detail = (
            f"missing={s['missing']} timeouts={s['errors']} malformed={s['malformed']} "
            f"retries={s['fallbacks']} loop={self._serial_errors_total}"
        )
        print(f"  errors   {errors} total — {detail}", end="\r\n", flush=True)

    def _update_logging(self, obs: Observation, command: MotorCommand) -> None:
        """Start, feed or stop the log session to follow the [l] toggle."""
        if obs.user_input.logging:
            if not self.logger.active:
                path = self.logger.start(obs.user_input.log_name)
                print(f"Logging to {path}", end="\r\n", flush=True)
            self.logger.record(obs.robot_state, command.target_angles, obs.user_input.velocity)
            if self._policy_started is not None:
                self.logger.mark_policy_start(self._policy_started)
        elif self.logger.active:
            self._stop_logging()

        self._policy_started = None

    def _stop_logging(self) -> None:
        """Close the log session. Never raises: losing a log must not take down the
        control loop, nor block the torque-off in _cleanup()."""
        try:
            path = self.logger.stop()
        except Exception as exc:
            print(f"Warning: could not write log: {exc}", end="\r\n", flush=True)
            return
        if path is not None:
            print(f"Log written to {path}", end="\r\n", flush=True)

    def _cleanup(self) -> None:
        """Disable torque, stop input source, and clear stop artifacts."""
        if self._cleanup_done:
            return

        self._cleanup_done = True

        if self.input_source:
            self.input_source.stop()

        shutdown = getattr(self.controller, "shutdown", None)
        if callable(shutdown):
            shutdown()

        motor_ids = list(MOTOR_TO_ID.values())
        self.controller.sync_write_torque_enable(motor_ids, [False] * len(motor_ids))
        print("Torque disabled on all motors", end="\r\n", flush=True)

        # Flush an in-progress session last, so quitting mid-log still yields a file but a
        # logging failure can never keep torque enabled on the way out.
        if self.logger.active:
            self._stop_logging()

        if self.stop_flag_path.exists():
            self.stop_flag_path.unlink()

    def _estimate_total_current(self, robot_state, target_angles: dict[str, float]) -> float:
        """Estimate the total pack current [A] from whichever signal is available.

        - If present_current was read (Observer.observe_current = True), use the measured
          sum of |current| over all motors.
        - Otherwise, fall back to the bam XL330 m6 current proxy (no extra bus read), from
          the (delay-aligned) command target, present_position and present_velocity:
            duty = clip(PROXY_KP * PROXY_ERROR_GAIN * (target - q), ±PROXY_MAX_PWM)
            I    = (PROXY_VIN * duty - PROXY_KT * dq) / PROXY_R, |I| capped at BAM_MAX_CURRENT
          The back-EMF term (PROXY_KT * dq) keeps the estimate low when the motor is moving
          toward an overshot RL target, while still flagging stalled motors (low dq, high error).
        """
        currents = robot_state.motor_currents
        if currents:
            return sum(abs(c) for c in currents.values())

        positions = robot_state.motor_positions
        velocities = robot_state.motor_velocities
        total = 0.0
        for name, target in target_angles.items():
            pos = positions.get(name)
            if pos is None:
                continue
            dq = velocities.get(name, 0.0)
            duty = PROXY_KP * PROXY_ERROR_GAIN * (target - pos)
            duty = max(-PROXY_MAX_PWM, min(PROXY_MAX_PWM, duty))
            current = (PROXY_VIN * duty - PROXY_KT * dq) / PROXY_R
            total += min(abs(current), BAM_MAX_CURRENT)
        return total

    def _check_overcurrent(self, robot_state, target_angles: dict[str, float]) -> bool:
        """Report when the estimated total pack current stays above OVERCURRENT_CUTOFF_A
        for OVERCURRENT_DEBOUNCE_TICKS consecutive ticks.

        When True, the run loop breaks and _cleanup() disables torque on every motor,
        leaving the robot compliant so the BMS does not trip on the current spike.
        """
        total_current = self._estimate_total_current(robot_state, target_angles)
        if total_current >= OVERCURRENT_CUTOFF_A:
            self._overcurrent_ticks += 1
        else:
            self._overcurrent_ticks = 0

        if self._overcurrent_ticks >= OVERCURRENT_DEBOUNCE_TICKS:
            print(
                f"Overcurrent safety triggered: {total_current:.2f} A (threshold "
                f"{OVERCURRENT_CUTOFF_A:.2f} A) — disabling torque",
                end="\r\n",
                flush=True,
            )
            return True
        return False

    def _send_to_motors(self, command: MotorCommand):
        """Send one batched goal position command from the composed command dict."""
        if not command.target_angles:
            return

        motor_ids = [MOTOR_TO_ID[name] for name in command.target_angles]
        target_positions = list(command.target_angles.values())

        self.controller.sync_write_goal_position(motor_ids, target_positions)