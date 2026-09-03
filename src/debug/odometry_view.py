# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""MuJoCo window showing what the odometry (src/odometry.py) believes the robot is doing.

Purely kinematic: the joints are written straight from the readback and the free base from
the estimated trunk pose, then only forward kinematics is run. Nothing is simulated, so a
foot sinking into or floating above the floor is the estimate being wrong, made visible.

The anchor — the sole corner the estimator currently holds immobile — is drawn as a dot on
the floor, blue under the left foot and red under the right. It jumps to a new corner at
every transfer, and each jump is where the odometry actually advances: the trunk moves to
wherever the leg says it must be for that corner to have landed there.

Shared by the offline replay (`src/debug/plot_log.py --view`) and the live view of a running
robot (`src/debug/live_viewer.py`), which both hand it `OdometrySample`s.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import MOTOR_TO_ID  # noqa: E402
from odometry import OdometrySample, matrix_to_quat  # noqa: E402

# Scene rather than the bare robot: a floor and lights to judge the estimate against. The
# odometry itself loads the robot alone (odometry.DEFAULT_MODEL_PATH).
SCENE_PATH = "src/model/mjcf/scene.xml"

ANCHOR_RADIUS = 0.006
ANCHOR_RGBA = {
    "left": (0.2, 0.4, 1.0, 1.0),
    "right": (1.0, 0.25, 0.2, 1.0),
}


class OdometryView:
    """A passive MuJoCo viewer posed from odometry samples. Use as a context manager."""

    def __init__(self, scene_path: str = SCENE_PATH, key_callback=None) -> None:
        import mujoco
        import mujoco.viewer

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        self._qpos_adr: dict[str, int] = {}
        for name in MOTOR_TO_ID:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                self._qpos_adr[name] = self.model.jnt_qposadr[joint_id]

        # The left panel is MuJoCo's simulation options, none of which apply to a
        # kinematic replay, and it covers the robot on a small window (Tab brings it back).
        self._viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=key_callback, show_left_ui=False
        )

        # One user geom for the anchor dot, created once and moved every frame.
        scn = self._viewer.user_scn
        mujoco.mjv_initGeom(
            scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([ANCHOR_RADIUS, 0.0, 0.0]),
            np.zeros(3),
            np.eye(3).flatten(),
            np.array(ANCHOR_RGBA["left"], dtype=np.float32),
        )
        scn.ngeom = 1

    def __enter__(self) -> "OdometryView":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def is_running(self) -> bool:
        return self._viewer.is_running()

    def close(self) -> None:
        self._viewer.close()

    def show(self, sample: OdometrySample) -> None:
        """Pose the robot as `sample` says and redraw."""
        T = sample.T_world_trunk
        with self._viewer.lock():
            self.data.qpos[0:3] = T[:3, 3]
            self.data.qpos[3:7] = matrix_to_quat(T[:3, :3])
            for name, adr in self._qpos_adr.items():
                self.data.qpos[adr] = sample.joints.get(name, 0.0)
            self._mujoco.mj_forward(self.model, self.data)

            anchor = self._viewer.user_scn.geoms[0]
            if sample.support_position is not None:
                anchor.pos[:] = sample.support_position
                anchor.rgba[:] = ANCHOR_RGBA.get(sample.support_side, ANCHOR_RGBA["left"])
                self._viewer.user_scn.ngeom = 1
            else:
                self._viewer.user_scn.ngeom = 0
        self._viewer.sync()
