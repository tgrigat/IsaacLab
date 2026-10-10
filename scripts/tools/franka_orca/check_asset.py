# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sanity-check the Franka + ORCA asset written by ``build_asset.py``, in plain Newton (no Kit).

Three checks, each printing one line and failing loudly:

* :func:`check_taxels`: every taxel in a finger's sidecar is exactly one collision shape on that
  finger's fingertip body.
* :func:`check_against_reference`: at the same joint positions, every body sits where it sits in a
  reference USD, such as a previous build, so mount and link frames are unchanged.
* :func:`check_self_collision`: no two robot bodies touch at the home pose, and no pair touches in every
  random pose, so the collision filter pairs cover every permanent overlap.

Usage:

.. code-block:: bash

    uv run python scripts/tools/franka_orca/check_asset.py --reference /path/to/previous/franka_orca.usda
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import newton
import numpy as np
import warp as wp

from isaaclab_assets.robots.franka_orca import FRANKA_ORCA_CFG, FRANKA_ORCA_TACTILE_DIR, FRANKA_ORCA_USD

MAX_POSITION_ERROR = 1e-4  # m
"""Largest body position difference to the reference asset."""

MAX_ANGLE_ERROR = 1e-3  # rad
"""Largest body orientation difference to the reference asset."""


def load_model(usd: Path) -> newton.Model:
    """Load a robot USD on its own into a Newton model."""
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.0  # report touching shapes only, not the default 10 cm of speculative contacts
    builder.add_usd(str(usd))
    return builder.finalize()


def leaf(label: str) -> str:
    """Last path component of a Newton label, i.e. the prim name."""
    return label.rsplit("/", 1)[-1]


def check_taxels(model: newton.Model, finger: str) -> None:
    """Check that the sidecar's taxels are collision shapes on the sidecar's fingertip body."""
    sidecar = np.load(FRANKA_ORCA_TACTILE_DIR / f"{finger}.npz")
    body_names = [leaf(label) for label in model.body_label]
    fingertip = body_names.index(str(sidecar["body"]))
    shape_body = model.shape_body.numpy()
    # a URDF collision becomes an Xform named after it with one collider mesh below
    shape_parent = [model.shape_label[i].rsplit("/", 2)[-2] for i in range(model.shape_count)]
    for name in sidecar["patch_names"]:
        shapes = [i for i, parent in enumerate(shape_parent) if parent == name]
        if len(shapes) != 1 or shape_body[shapes[0]] != fingertip:
            raise RuntimeError(f"{finger}: taxel {name} is {len(shapes)} shapes, expected one on {sidecar['body']}")
    print(f"[ok] {finger}: {len(sidecar['patch_names'])} taxel shapes on {sidecar['body']}")


def joint_positions(model: newton.Model, values: dict[str, float]) -> np.ndarray:
    """Model joint coordinates with the named 1-DoF joints set and everything else at zero."""
    q = np.zeros(model.joint_coord_count, dtype=np.float32)
    starts = model.joint_q_start.numpy()
    for joint, label in enumerate(model.joint_label):
        if leaf(label) in values:
            q[starts[joint]] = values[leaf(label)]
    return q


def body_poses(model: newton.Model, q: np.ndarray) -> dict[str, np.ndarray]:
    """Body poses ``[x, y, z, qx, qy, qz, qw]`` by name after forward kinematics at ``q``."""
    state = model.state()
    newton.eval_fk(model, wp.array(q, dtype=wp.float32, device=model.device), model.joint_qd, state)
    poses = state.body_q.numpy()
    return {leaf(label): poses[i] for i, label in enumerate(model.body_label)}


def check_against_reference(model: newton.Model, reference_usd: Path) -> None:
    """Check that bodies sit where they sit in the reference asset, position and orientation, at the home pose."""
    reference = load_model(reference_usd)
    home = FRANKA_ORCA_CFG.init_state.joint_pos
    ours = body_poses(model, joint_positions(model, home))
    theirs = body_poses(reference, joint_positions(reference, home))
    shared = [name for name in ours if name in theirs]
    position_error = {name: np.linalg.norm(ours[name][:3] - theirs[name][:3]) for name in shared}
    # angle between unit quaternions, sign-agnostic
    angle_error = {name: 2 * np.arccos(min(1.0, abs(ours[name][3:] @ theirs[name][3:]))) for name in shared}
    worst = max(position_error, key=position_error.get)
    if position_error[worst] > MAX_POSITION_ERROR:
        raise RuntimeError(f"{worst} is {position_error[worst] * 1e3:.2f} mm off the reference asset")
    worst_angle = max(angle_error, key=angle_error.get)
    if angle_error[worst_angle] > MAX_ANGLE_ERROR:
        raise RuntimeError(f"{worst_angle} is rotated {np.degrees(angle_error[worst_angle]):.1f} deg off the reference")
    print(
        f"[ok] {len(shared)} bodies match the reference (worst {position_error[worst] * 1e3:.4f} mm, "
        f"{np.degrees(angle_error[worst_angle]):.4f} deg)"
    )


def check_self_collision(model: newton.Model, num_poses: int, seed: int = 0) -> None:
    """Check that no two bodies touch at the home pose, and that no pair touches in all random poses."""
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    shape_body = model.shape_body.numpy()
    body_names = [leaf(label) for label in model.body_label]
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()

    rng = np.random.default_rng(seed)
    home = joint_positions(model, FRANKA_ORCA_CFG.init_state.joint_pos)
    touching = Counter()
    for pose in range(num_poses + 1):
        q = home if pose == 0 else rng.uniform(lower, upper).astype(np.float32)  # all joints are 1-DoF
        state = model.state()
        newton.eval_fk(model, wp.array(q, dtype=wp.float32, device=model.device), model.joint_qd, state)
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        pairs = {tuple(sorted((body_names[shape_body[a]], body_names[shape_body[b]]))) for a, b in zip(shape0, shape1)}
        if pose == 0 and pairs:
            raise RuntimeError(f"bodies touch at the home pose: {sorted(pairs)}")
        touching.update(pairs)
    for (a, b), n in touching.most_common():
        print(f"     {a} - {b}: touching in {n}/{num_poses} random poses")
    # touching in every pose means the pair overlaps permanently and needs a filter pair
    permanent = [pair for pair, n in touching.items() if n == num_poses]
    if permanent:
        raise RuntimeError(f"permanently overlapping, unfiltered bodies: {permanent}")
    print(f"[ok] no contacts at the home pose and no permanent overlaps over {num_poses} random poses")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fingers", nargs="+", default=["thumb"])
    parser.add_argument("--reference", type=Path, help="USD to compare body poses against.")
    parser.add_argument("--num-poses", type=int, default=40)
    args = parser.parse_args()

    model = load_model(FRANKA_ORCA_USD)
    for finger in args.fingers:
        check_taxels(model, finger)
    if args.reference:
        check_against_reference(model, args.reference)
    check_self_collision(model, args.num_poses)


if __name__ == "__main__":
    main()
