# Using Microban

This guide covers day-to-day operation once the robot has been set up (see the
[Deployment Guide](deployment.md)). You drive the robot from your computer with the
`Makefile`, which talks to the Pi over SSH. You can use the keyboard or a Bluetooth 
gamepad to control it, and optionally run it

> [!IMPORTANT]
> Always run `make shutdown` before cutting power to the robot. This is **not**
> automatic — powering off the Pi without a clean shutdown can corrupt the SD card.
> Wait 10-15 s after the command before flipping the power switch off, to give the
> Pi time to actually halt.

## Makefile commands

Run these from the repository root on your computer. They target the host `microban`
by default; add `HOST=microban-ext` to operate over the secondary network (see the
[Deployment Guide](deployment.md)).

| Command | What it does |
| :--- | :--- |
| `make run` | Sync the code and start the control loop on the robot (50 Hz). Stays attached to your terminal for live control. |
| `make stop` | Stop the control loop and disable torque on all motors. |
| `make shutdown` | Power off the Pi cleanly. |
| `make setup` | Sync the code and (re)install dependencies on the robot (`uv sync --frozen`). Run after changing dependencies. |
| `make sync` | Sync your local copy to the robot without touching dependencies. |
| `make imu` | Stream the robot's IMU/gyro readings to your terminal. |
| `make voltage` | Read the voltage of all motors. |
| `make voltage ID=<id>` | Read the voltage of motor `<id>`. |
| `make sim` | Run the MuJoCo simulation locally (no robot needed). |
| `make viewer` | Open the MuJoCo viewer locally (no robot needed). |
| `make get-logs` | Copy the JSON logs recorded with `l` from the robot into `./logs/`. |

## Running the robot

1. Place the robot on a stable surface, or hold it securely — on start it enables
   torque and ramps to its neutral pose.
2. `make run` — the control loop starts at 50 Hz and stays attached to your terminal. Some latency is expected due to the SSH connection.
3. Toggle moves and drive the robot (see below).
4. `make stop` (or press `q`) to stop; `make shutdown` to power off.

## Controlling with the keyboard

| Key | Action |
| :--- | :--- |
| `v` | toggle the **walk** move |
| `h` | toggle the **head** move |
| `s` | toggle the **squat** move |
| arrows | `vx` (up/down), `vtheta` (left/right) |
| `x` | reset velocity to zero |
| `i` | toggle the IMU/gyro display |
| `l` | start/stop logging (see below) |
| `p` | toggle the scheduler timing display (see below) |
| `q` | stop the control loop |

## Timing the control loop

Press `p` to report where each 20 ms tick goes, twice a second:

```
--------------------------------------------
Timings over 25 ticks — budget 20.0 ms/tick
  read  avg=  8.21  max= 12.05 ms
  moves avg=  1.34  max=  2.02 ms
  send  avg=  2.11  max=  3.44 ms
  tick  avg= 11.66  max= 17.52 ms  (0 over budget)
```

`read` is the observer (the serial and IMU reads — usually the bulk of it on the robot),
`moves` the policy or IK work, `send` the bus write (in simulation, the physics step and
viewer sync), and `tick` the whole of it. Anything above the budget cannot keep 50 Hz, and
is what the `control loop overrun` warnings report.

Figures are averaged over the window rather than printed every tick: at 50 Hz a per-tick
print would flood the terminal, and the writing would itself distort what is being
measured. For the same reason the numbers exclude the timing and IMU displays themselves.

## Logging a session

Press `l` to start recording, and `l` again to stop. On start you are prompted for an
optional name — press Enter to skip it, or `Esc` to cancel. Each session is written as one
JSON file under `logs/`, named after its start date plus that name:
`logs/2026-07-17_14-32-05_walk-test.json`.

Every tick records the target position, the read position and velocity of each motor, the
IMU gyro and quaternion, and the `vx`/`vy`/`vtheta` command fed to the walk policy. Channels
are keyed by motor name and are all the same length as `time`, so a dropped IMU read shows
up as `null` rather than shifting the series.

Quitting with `q` while a session is running still writes the file. Logs stay on the robot
until you pull them over with `make get-logs`, which copies them into your local `logs/`.

If a policy (`walk`, `squat_rl`) goes active during the session, the moment it started is
recorded as `policy_t0` in the log's metadata. A policy already running when you press `l`
is not stamped, since its real start time falls outside the log.

To inspect a log, `uv run --group debug src/debug/plot_log.py` plots the newest one (pass a
path for a specific one). Tick a joint and it gets its own pair of plots — position (goal
dashed, read solid) and velocity; untick to take it away. Nothing is plotted until you tick
something. The `debug` group carries matplotlib for the scripts in `src/debug/`; it is not
installed on the robot.

Below the joints, `roll`, `pitch` and `gyro x/y/z` tick the same way, each taking a
full-width row since they have no goal to compare against. Roll and pitch are the trunk's,
in degrees, derived from the logged `body_quat`; the gyro is plotted raw, in the IMU sensor
frame — the same signal the policies observe. Logs recorded before `body_quat` existed
still offer the gyro, but not roll/pitch.

Pass several logs to compare the same joint across runs, overlaid with one colour per log:

```
uv run --group debug src/debug/plot_log.py logs/a.json logs/b.json
```

When every log has a `policy_t0`, time is shifted so `t = 0` is the policy start in each
run, lining the traces up however late you happened to press `l`. If any log lacks it, the
comparison falls back to raw log time and says so.

> While the name prompt is open, keys are captured by the prompt — `Ctrl+C` still stops
> the control loop if you need it.

## Controlling with a gamepad

A Bluetooth Xbox controller can be used instead of the keyboard. The detailed explanation of the gamepad usage is in [Gamepad Guide](gamepad.md). 

Using a gamepad allows to drive the robot through two different modes: with a terminal (SSH) or fully headless (no SSH, no terminal). The second mode is particularly useful for demonstration purposes, due to the fact that it allows to drive the robot without any computer connected to it.

## Moves

Moves are toggled independently and run on top of the neutral pose:

- **Walk** (`v` / gamepad **A**) — a reinforcement-learning policy. Once active, the
  velocity command drives it: `vx` (forward/back), `vy` (lateral), `vtheta` (turn),
  set from the arrow keys or the gamepad sticks.
- **Head** (`h`) — oscillates the head.
- **Squat** (`s`) — squat motion computed with inverse kinematics.

### Velocity command

Every input source emits a **normalized** command in `[-1, 1]` per axis; the scheduler
maps it to physical limits with `scale_velocity()`, so the behavior is identical for
keyboard, gamepad and sim. Defaults (in [constants.py](../src/constants.py)):

| Axis | Max |
| :--- | :--- |
| `vx` (forward) | +0.7 |
| `vx` (backward) | -0.5 |
| `vy` (lateral) | ±0.3 |
| `vtheta` (turning in place, `vx = vy = 0`) | ±3.0 |
| `vtheta` (while translating) | ±1.5 |

## Developing: adding your own moves

Each behavior is a subclass of `Move` ([src/moves/move.py](../src/moves/move.py)) with
a simple lifecycle driven by the scheduler:

- `preload()` — optional, called once before the loop starts (load heavy resources).
- `on_start(obs, command)` — called each tick while *starting*; set
  `self.state = MoveState.ACTIVE` when ready (e.g. after ramping in).
- `step(obs, command)` — called each tick while *active*; write your target joint
  angles into `command.target_angles`.
- `on_stop(obs, command)` — called each tick while *stopping*; set
  `self.state = MoveState.INACTIVE` when done (e.g. after ramping back to neutral).

To add a move:

1. Create a new file in [src/moves/](../src/moves/) with a class subclassing `Move`.
   Use [rotate_head.py](../src/moves/rotate_head.py) (a simple oscillation) or
   [squat.py](../src/moves/squat.py) (inverse kinematics with placo) as a template.
2. Register it in [src/main.py](../src/main.py): add it to the `moves` dict passed to
   the `Scheduler`, and add a trigger — a key in `MOVE_KEYS` (keyboard) and/or a button
   in `GAMEPAD_BUTTON_MOVES` (gamepad).
3. In `step()`, read the robot state from `obs.robot_state` (motor positions and
   velocities, IMU gyro, projected gravity) and write your targets into
   `command.target_angles`.

### Training your own walk (or other RL) policies

The walk move runs an ONNX policy trained in simulation. 
You can train your own walking — or other learned skills — and drop the resulting `.onnx` file into [src/agents/](../src/agents/) to use it on the robot. Check the repository [MarcDcls/mjlab_microban](https://github.com/MarcDcls/mjlab_microban) for the training pipeline. 

If you achieve some interesting results, don't hesitate to make a pull request to the repository as it is also a community-driven project!