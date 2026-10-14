# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Show the collision and visual meshes of a URDF in the Newton viewer, straight from the URDF.

No Isaac Lab, no USD conversion and no physics: the URDF is parsed by Newton and drawn at its zero joint pose (or at
``--joint-pos``, or at the robot config's default joint positions with ``--home``). Toggle "Show Collision" and
"Show Visual" under "Visualization Markers" in the side panel (H toggles the panel).

Usage:

.. code-block:: bash

    uv run python scripts/tools/franka_orca/view_urdf_collision.py
    uv run python scripts/tools/franka_orca/view_urdf_collision.py --home
    uv run python scripts/tools/franka_orca/view_urdf_collision.py --fingertips no_taxels
"""

import argparse
import os
import re
from pathlib import Path

# on Wayland, pyGLFW loads a GLFW without the X11 symbols the viewer's imgui needs, and the side panel is missing
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import newton  # noqa: E402
import newton.viewer  # noqa: E402

from isaaclab_assets.robots.franka_orca import FINGERTIP_METHODS, FRANKA_ORCA_CFG, franka_orca_usd  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fingertips", choices=FINGERTIP_METHODS, default="mesh_taxels", help="Asset to show.")
    parser.add_argument("--urdf", type=Path, help="URDF to show instead of the --fingertips asset's.")
    parser.add_argument("--joint-pos", type=float, default=0.0, help="Position of every joint [rad].")
    parser.add_argument(
        "--home", action="store_true", help="Pose the joints at FRANKA_ORCA_CFG's default joint positions."
    )
    parser.add_argument("--hide-visual", action="store_true", help="Start with only the collision meshes shown.")
    args = parser.parse_args()

    builder = newton.ModelBuilder()
    urdf = args.urdf or franka_orca_usd(args.fingertips).with_name("franka_orca.urdf")
    builder.add_urdf(str(urdf), floating=False, force_show_colliders=True)
    # the URDF importer keeps the meshes as they are; the simulation collides with their capped convex hulls
    builder.approximate_meshes("convex_hull")
    builder.joint_q[:] = [args.joint_pos] * len(builder.joint_q)
    if args.home:
        for joint, label in enumerate(builder.joint_label):
            name = label.rsplit("/", 1)[-1]
            for pattern, value in FRANKA_ORCA_CFG.init_state.joint_pos.items():
                if re.fullmatch(pattern, name):
                    builder.joint_q[builder.joint_q_start[joint]] = value
                    break
    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    viewer.show_collision = True
    viewer.show_visual = not args.hide_visual
    print(f"{urdf}: {model.body_count} bodies, {model.shape_count} shapes")
    time = 0.0
    while viewer.is_running():
        viewer.begin_frame(time)
        viewer.log_state(state)
        viewer.end_frame()
        time += 1.0 / 60.0
    viewer.close()


if __name__ == "__main__":
    main()
