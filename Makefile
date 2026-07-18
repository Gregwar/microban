.PHONY: sync setup run stop shutdown voltage imu sim viewer get-logs check-read set-baud gamepad-headless-enable gamepad-headless-disable

HOST ?= microban
ID ?=
BAUD ?=

sync:
	rsync -avz \
		--exclude='.git' \
		--exclude='.venv' \
		--exclude='__pycache__' \
		--exclude='cad' \
		--exclude='docs' \
		--exclude='logs' \
		--exclude='sd_card' \
		--exclude='src/debug' \
		--exclude='src/sim' \
		--exclude='src/model/mjcf' \
		./ $(HOST):microban

setup: sync
	ssh $(HOST) "bash -l -c 'cd microban && uv sync --frozen'"

sim:
	PYTHONPATH=src uv run --group sim src/sim/sim_main.py --hz 50

viewer:
	PYTHONPATH=src uv run src/sim/viewer_main.py --hz 25

run: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/main.py'"

stop:
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/stop.py'"

# Fetch the JSON logs recorded with [l] during a session. Kept out of `sync` (which
# excludes logs/), so deploying never touches what is on the robot.
get-logs:
	rsync -avz $(HOST):microban/logs/ ./logs/

voltage: sync
	ssh $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/voltage.py $(ID)'"

imu: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/imu.py'"

# Verify the fused motor state read against the driver's own registers, and time it.
# Read-only (no torque, no goal writes) — leave the robot still on the bench.
check-read: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/check_read.py'"

# Change the motor bus rate: `make set-baud BAUD=2000000` (or 1000000 to go back).
# Writes EEPROM — read src/set_baud.py before running. Update MOTOR_BAUDRATE afterwards.
set-baud: sync
	@test -n "$(BAUD)" || { echo "Usage: make set-baud BAUD=2000000"; exit 1; }
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/set_baud.py $(BAUD)'"

shutdown:
	ssh -tt $(HOST) "sudo shutdown -h now"

# Opt-in headless mode: a service launches the control loop when START is held 2s on
# the gamepad (no SSH needed); B stops it. See docs/usage.md.
gamepad-headless-enable: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && sudo bash systemd/install-gamepad-daemon.sh'"

gamepad-headless-disable:
	ssh -tt $(HOST) "bash -l -c 'cd microban && sudo bash systemd/install-gamepad-daemon.sh --uninstall'"
