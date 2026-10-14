# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Show the hand's collision hulls as the environments import them, colored by kind, in the Newton viewer.

The robot USD is imported the way Isaac Lab's Newton backend imports it (``add_usd`` with the Newton and PhysX schema
resolvers), so each mesh collider becomes what the simulation uses: a convex hull (Newton caps it at 64 vertices), or
the triangle mesh itself where the asset sets ``physics:approximation = "none"`` (the taxels). The arm
is hidden and nothing is simulated; the hand is drawn at its zero joint pose (or at ``--joint-pos``, or at the robot
config's default joint positions with ``--home``).

Colors:

* taxel colliders (``<finger>_taxel_NNN``): orange, alternating shades so neighboring taxels stand apart
* fingertip unsensed hulls (``<finger>_unsensed_NN``): blue, alternating shades
* pad proxies (``<finger>_pad``): magenta, only with ``--pads`` because they overlap the taxels
* hulls of the other hand links (``<link>_hull``): gray

Toggle "Show Visual" under "Visualization Markers" in the side panel (H toggles the panel) to compare with the meshes.

Usage:

.. code-block:: bash

    uv run python scripts/tools/franka_orca/view_hand_collision.py
    uv run python scripts/tools/franka_orca/view_hand_collision.py --home --pads
    uv run python scripts/tools/franka_orca/view_hand_collision.py --usd assets/franka_orca/no_taxels/franka_orca.usda
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

# on Wayland, pyGLFW loads a GLFW without the X11 symbols the viewer's imgui needs, and the side panel is missing
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import newton  # noqa: E402
import newton.usd  # noqa: E402
import newton.viewer  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

from isaaclab_assets.robots.franka_orca import FRANKA_ORCA_CFG, FRANKA_ORCA_USD  # noqa: E402

KIND_COLORS = {
    "taxel": [(1.0, 0.55, 0.1), (0.85, 0.35, 0.0)],
    "unsensed": [(0.2, 0.45, 0.9), (0.1, 0.3, 0.7)],
    "pad": [(0.85, 0.2, 0.75)],
    "hull": [(0.6, 0.6, 0.6)],
}
"""Display colors per collider kind; a kind with several colors cycles through them."""


def collider_kind(name: str) -> str:
    """Kind of a hand collider from the name of its collision prim (see ``build_asset.py``)."""
    if "_taxel_" in name:
        return "taxel"
    if "_unsensed" in name:
        return "unsensed"
    if name.endswith("_pad"):
        return "pad"
    return "hull"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--usd", type=Path, default=FRANKA_ORCA_USD)
    parser.add_argument("--joint-pos", type=float, default=0.0, help="Position of every joint [rad].")
    parser.add_argument(
        "--home", action="store_true", help="Pose the joints at FRANKA_ORCA_CFG's default joint positions."
    )
    parser.add_argument("--pads", action="store_true", help="Also show the pad proxies.")
    args = parser.parse_args()

    builder = newton.ModelBuilder()
    builder.add_usd(
        str(args.usd),
        floating=False,
        hide_collision_shapes=True,
        schema_resolvers=[newton.usd.SchemaResolverNewton(), newton.usd.SchemaResolverPhysx()],
    )

    builder.joint_q[:] = [args.joint_pos] * len(builder.joint_q)
    if args.home:
        for joint, label in enumerate(builder.joint_label):
            name = label.rsplit("/", 1)[-1]
            for pattern, value in FRANKA_ORCA_CFG.init_state.joint_pos.items():
                if re.fullmatch(pattern, name):
                    builder.joint_q[builder.joint_q_start[joint]] = value
                    break

    # color the hand colliders by kind and hide everything of the arm; nothing is simulated, so the flags only matter
    # to the viewer
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    hull_vertices = defaultdict(list)
    for shape, label in enumerate(builder.shape_label):
        body = builder.shape_body[shape]
        on_arm = body < 0 or builder.body_label[body].rsplit("/", 1)[-1].startswith("fr3_")
        if on_arm:
            builder.shape_flags[shape] = 0
            continue
        if not builder.shape_flags[shape] & collide:
            continue
        # the converter puts each collision in a prim named after it, holding a mesh that may be shared and renamed
        kind = collider_kind(label.split("/")[-2])
        hull_vertices[kind].append(len(builder.shape_source[shape].vertices))
        colors = KIND_COLORS[kind]
        builder.shape_color[shape] = colors[len(hull_vertices[kind]) % len(colors)]
        if kind == "pad" and not args.pads:
            builder.shape_flags[shape] &= ~collide

    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    print(f"{args.usd}: hand colliders as imported")
    for kind, counts in hull_vertices.items():
        print(f"  {kind:>9}: {len(counts):4d} hulls, {min(counts)}-{max(counts)} vertices, {sum(counts)} in total")

    # look at the hand from the side, slightly above
    hand_bodies = [b for b, label in enumerate(model.body_label) if not label.rsplit("/", 1)[-1].startswith("fr3_")]
    center = state.body_q.numpy()[hand_bodies, :3].mean(axis=0)
    eye = center + np.array([0.35, 0.35, 0.15])
    direction = center - eye
    yaw = np.degrees(np.arctan2(direction[1], direction[0]))
    pitch = np.degrees(np.arcsin(direction[2] / np.linalg.norm(direction)))

    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    viewer.show_collision = True
    viewer.show_visual = False
    viewer.set_camera(wp.vec3(*eye), pitch, yaw)
    time = 0.0
    while viewer.is_running():
        viewer.begin_frame(time)
        viewer.log_state(state)
        viewer.end_frame()
        time += 1.0 / 60.0
    viewer.close()


if __name__ == "__main__":
    main()
