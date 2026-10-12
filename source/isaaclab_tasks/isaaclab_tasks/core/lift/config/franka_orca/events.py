# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms of the Franka + ORCA-hand lift environments."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def aim_camera_at_env(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    env_index: int,
    eye: tuple[float, float, float],
    lookat: tuple[float, float, float],
):
    """Point the visualizer camera (and so the recorded videos) at one environment.

    The visualizer camera is placed in world coordinates, while the environments are spread over a grid centered
    on the world origin, so a fixed world camera sees none of them up close. Run at startup.

    Args:
        env: The environment.
        env_ids: Unused; the camera is shared by all environments.
        env_index: Environment to look at.
        eye: Camera position relative to that environment's origin [m].
        lookat: Camera target relative to that environment's origin [m].
    """
    origin = env.scene.env_origins[env_index].tolist()
    env.sim.set_camera_view(
        eye=tuple(o + e for o, e in zip(origin, eye)), target=tuple(o + t for o, t in zip(origin, lookat))
    )


def reset_joints_to_random_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    poses: list[dict[str, float]],
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset some joints to one of several poses, drawn uniformly per environment, plus a uniform offset.

    The velocities of the joints are reset to zero.

    Args:
        env: The environment.
        env_ids: Environments to reset.
        poses: Joint positions of each pose [rad], by joint name; every pose names the same joints.
        position_range: Range of the offset added to every joint [rad].
        asset_cfg: The articulation; its joint names are ignored, the poses name the joints.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, joint_names = asset.find_joints(list(poses[0]), preserve_order=True)
    targets = torch.tensor([[pose[name] for name in joint_names] for pose in poses], device=env.device)
    joint_pos = targets[torch.randint(len(poses), (len(env_ids),), device=env.device)]
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    limits = asset.data.soft_joint_pos_limits.torch[env_ids[:, None], joint_ids]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), joint_ids=joint_ids, env_ids=env_ids)


def object_in_bounds(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    in_bound_range: dict[str, tuple[float, float]],
    margin: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Validity criterion for :class:`~isaaclab_tasks.core.lift.mdp.events.conditional_reset`: object inside the
    workspace, at least :paramref:`margin` from every face of it.

    Rejects reset states that the out-of-bounds termination would end at once, or soon after.

    Args:
        env: The environment.
        env_ids: Environments to check.
        in_bound_range: Workspace ranges of the object position in the environment frame [m], keys ``x``/``y``/``z``.
        margin: Required distance of the object position from the workspace faces [m].
        asset_cfg: The object.

    Returns:
        Whether each environment's object is inside, shape [len(env_ids)].
    """
    asset = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w.torch[env_ids] - env.scene.env_origins[env_ids]
    ok = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
    for axis, key in enumerate(("x", "y", "z")):
        low, high = in_bound_range[key]
        ok &= (pos[:, axis] >= low + margin) & (pos[:, axis] <= high - margin)
    return ok


def reset_object_on_table_near_body(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    target_cfg: SceneEntityCfg,
    offset: tuple[float, float, float],
    radius_range: tuple[float, float],
    height: float,
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset an object onto a table, at a random spot around the point below a body, at a random yaw.

    The spot is the body's position plus :paramref:`offset` in the body frame, projected onto the table, plus a
    horizontal offset of random direction and a length drawn from :paramref:`radius_range`. The object is placed upright
    at :paramref:`height` in the environment frame (e.g. resting on the table top), at rest. The body pose is read from
    the live simulation state, so the term must run after the reset terms that pose the body.

    Args:
        env: The environment.
        env_ids: Environments to reset.
        target_cfg: Body around which the object is placed (e.g. the palm).
        offset: Offset in the body frame of the point to place the object around [m].
        radius_range: Range of the horizontal distance from that point [m].
        height: Height of the object position in the environment frame [m].
        yaw_range: Range of the yaw about the vertical [rad].
        asset_cfg: The object.
    """
    env_ids = env.scene._ALL_INDICES[env_ids]
    asset = env.scene[asset_cfg.name]
    target = env.scene[target_cfg.name]
    body_pos = target.data.body_pos_w.torch[env_ids, target_cfg.body_ids[0]]
    body_quat = target.data.body_quat_w.torch[env_ids, target_cfg.body_ids[0]]
    num = len(env_ids)
    offset_w = math_utils.quat_apply(body_quat, torch.tensor(offset, device=env.device).expand(num, 3))
    radius = math_utils.sample_uniform(*radius_range, (num,), env.device)
    direction = math_utils.sample_uniform(-math.pi, math.pi, (num,), env.device)
    pos = body_pos + offset_w
    pos[:, 0] += radius * torch.cos(direction)
    pos[:, 1] += radius * torch.sin(direction)
    pos[:, 2] = env.scene.env_origins[env_ids, 2] + height
    yaw = math_utils.sample_uniform(*yaw_range, (num,), env.device)
    quat = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(num, 6, device=env.device), env_ids=env_ids)
