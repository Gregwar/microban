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
| `g` | toggle the **getup** move |
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

With `u` also on, the report gains a battery line — the servo input voltage averaged over
every motor, and over the window like the timings above it:

```
  battery  11.68 V  (mean over 14 motors)
```

It only appears while `u` is on, since the voltage register is not read otherwise.

## Tracing packet loss

The same report carries the motor bus traffic, counted from startup rather than over the
window so that a fault which happened once during the run stays on screen:

```
  packets  sent=2007  received=38000/38133  (0.35% missing)
  errors   9 total — silent=6 timeout=1 malformed=1 retries=1 loop=1
```

`sent` counts instruction packets; `received`/`expected` counts the status packets the
motors owe in reply. One silent motor costs a whole transaction, not one packet: the driver
aborts a sync read at the first reply that does not parse and discards the ones it had
already collected, so nineteen motors' worth of state is lost each time. That is why
`missing` climbs in steps of nineteen while `errors` climbs by one.

When anything has gone wrong, the per-motor breakdown follows — the part worth acting on:

```
  faults by motor
    right_knee (24)          silentx5                           last t=41.3s
    left_elbow (33)          malformedx1                        last t=52.8s
    answered in a silent slot: right_ankle_pitch (25)x5
  unattributed  timeoutx1  (no id in the frame that failed)
                MICROBAN_BUS_PROBE=1 pings after a failure to name the motor
  last 5 faults
    t=   41.3s  sync_read_raw_data     silent    right_knee (24)  <- right_ankle_pitch (25) answered instead
    t=   52.8s  sync_read_state        malformed left_elbow (33)
```

The fault kinds, and what each one points at:

| kind | what happened | usual cause |
| --- | --- | --- |
| `silent` | the motor did not answer its slot, so the *next* motor's reply was read in its place | that motor's wiring, connector, or power |
| `timeout` | nothing came back at all | the last motor of the request went quiet, several at once, or the bus is down |
| `checksum` | a reply arrived corrupted | electrical noise, or a baud rate the wiring cannot carry |
| `parsing` | a reply arrived structurally impossible | same, further along |
| `malformed` | the reply decoded to the wrong width | a stuffed frame our decoder could not undo |
| `retries` | a fused read fell back to three separate typed reads | follows a `malformed`, costs latency, not data |
| `loop` | the scheduler's own read retries | one per tick that lost its state entirely |

Read the shape, not just the count. **One motor accounting for nearly every fault** is
physical — its connector, its wire, or its position in the daisy chain; swap it with a
neighbour and see whether the fault follows the motor or stays at the position. **Faults
spread evenly over all nineteen** are systemic — try `MICROBAN_BAUD=1000000` to see whether
the loss disappears at half the rate, and check the supply under load with `u`.

`silent` names a motor because the protocol makes it recoverable: replies come back in the
order they were requested, so a reply bearing the wrong id says both who stayed quiet and
who answered in their place. `timeout`, `checksum` and `parsing` fail before any id is
known, and are counted as unattributed. Running with `MICROBAN_BUS_PROBE=1` pings every
motor after such a failure to name the silent one — off by default, because it costs
nineteen extra round trips inside a tick that has already overrun, so use it on the bench
rather than under a walking policy.

## Logging a session

Press `l` to start recording, and `l` again to stop. On start you are prompted for an
optional name — press Enter to skip it, or `Esc` to cancel. Each session is written as one
JSON file under `logs/`, named after its start date plus that name:
`logs/2026-07-17_14-32-05_walk-test.json`.

Every tick records the target position, the read position and velocity of each motor, the
IMU gyro and quaternion, and the command fed to the policies: `vx`/`vy`/`vtheta` for the walk
policy, and `height` — the trunk height target in metres — for the squat policy, `null` on
every tick where no move commanded one. Channels are keyed by motor name and are all the
same length as `time`, so a dropped IMU read shows up as `null` rather than shifting the
series.

Quitting with `q` while a session is running still writes the file. Logs stay on the robot
until you pull them over with `make get-logs`, which copies them into your local `logs/`.

If a policy (`walk`, `squat_rl`, `getup`) goes active during the session, the moment it started is
recorded as `policy_t0` in the log's metadata. A policy already running when you press `l`
is not stamped, since its real start time falls outside the log.

To inspect a log, `uv run --group debug src/debug/plot_log.py` plots the newest one (pass a
path for a specific one). Tick a joint and it gets its own pair of plots — position (goal
dashed, read solid) and velocity; untick to take it away. Nothing is plotted until you tick
something. The `debug` group carries matplotlib for the scripts in `src/debug/`; it is not
installed on the robot.

Below the joints, `roll`, `pitch`, `gyro x/y/z` and — on a log where the squat policy ran —
`height target` tick the same way, each taking a full-width row since they have no goal to
compare against. `height target` is the trunk height the squat policy was told to track, so
a squat run should read back as the commanded sine, with gaps wherever the policy was not
running. Roll and pitch are the trunk's,
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

### Odometry

`--odometry` replays the log through the kinematic estimator in `src/odometry.py` and adds
where the robot went: `odom x/y/z` (the trunk in the world, in metres), `odom yaw` (its
heading in degrees, unwrapped so a turn reads continuously), `body vx/vy` (the trunk's
linear velocity in its own frame) and `body vyaw` (its yaw rate). They tick like any other
trace.

`body vyaw` is the gyro with the IMU mount rotation taken out, so it is the trunk yaw rate
itself — the raw `gyro y` channel is the *negative* of it, which is easy to misread.

Those three velocity rows each carry two overlays. A dashed EMA (0.75 s, a stride or two)
in the log's own colour: the raw curve swings by ±0.5 m/s within every stride, so the EMA is
what you read a sustained walking speed or a steady turn rate off. And a solid black line
for the velocity *command* — the `vx`/`vy`/`vtheta` the scheduler handed the policy, logged
in physical units, so tracking error reads straight off the gap between black and dashed.
The command is black in comparison figures too: it is the shared target, not a per-log
measurement. Logs predating the `command` channel just get no black line.

`odom z` carries the same black overlay for the squat policy: the trunk height it was asked
for against the height the kinematics say it reached.

Worth knowing what that gap looks like. On `old_logs/2026-07-23_08-37-17_walk_m4_cur.json`,
steady walking commanded 0.479 m/s forward and the odometry estimates 0.313 — the robot
delivers about two thirds of what it is asked for. On the same stretch `vtheta` is commanded
0 while the estimate sits at −0.085 rad/s, i.e. ~5 deg/s of unasked-for yaw.

```
uv run --group debug src/debug/plot_log.py logs/a.json --odometry
```

The estimate is a Python transcription of what the rhoban humanoid runs online. Each tick it
puts the read joint angles into a placo model, takes the floating base angular velocity
straight from the gyro, hands the support anchor to whichever sole corner is now lowest —
the `*_foot_front_left` and friends sites in `src/model/mjcf/robot.xml` — solves for the base
linear velocity that keeps that anchor immobile, and re-hangs the robot from it at the
attitude `body_quat` reports. Position therefore accumulates step by step from the
kinematics: it is dead reckoning, it drifts, and nothing ever corrects it. Heading is the
IMU's own gyro-integrated yaw. The velocities are per-tick estimates driven by the servos'
own velocity reads, so they are noisy at stride frequency even when the position curve is
clean.

The log is linearly interpolated up to a uniform 5 ms grid (`odometry.ODOMETRY_DT`) before
any of that runs, so the odometry traces are on a finer time axis than the rest of the plot.
The anchor is only ever re-planted *between* ticks, wherever forward kinematics says the new
corner is at the moment the switch is noticed, so the grid sets how late every transfer
lands. It matters more than it looks: on a 24 s walk log, refining 20 ms → 10 → 5 → 2.5 →
1.25 ms moves the final position by 0.27, 0.13, 0.05 and 0.03 m respectively — halving each
time, converging. 20 ms is the outlier; 5 ms lands within ~0.1 m of the limit for
essentially no extra cost (the placo model load dominates the runtime either way).

`--view` replays that estimate in the MuJoCo viewer instead of plotting, looping until you
close the window, paced by the log's own timestamps:

```
uv run --group debug --group sim src/debug/plot_log.py logs/a.json --view
```

The joints come from the recorded readback and the floating base from the odometry, both on
that same interpolated 5 ms grid, so the playback is smoother than the 50 Hz the log was
recorded at. Only forward kinematics runs, nothing is simulated. So this shows what the estimator believes
happened — a foot sinking through the floor or hovering above it is the drift made visible.
For a *physical* replay, where the log's goal positions drive simulated motors, use
`src/debug/sim_log.py` instead.

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
- **Getup** (`g`) — a reinforcement-learning policy that rights the robot from a fallen
  pose, face down, face up or on its side, and keeps balancing once it is up. It takes no
  command: toggle it on where the robot lies, and off once it is standing. Unlike the walk
  and squat policies it has no fall check — being fallen is what it is for — so it will
  keep driving the joints whatever the orientation until you untoggle it.

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