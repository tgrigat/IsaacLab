# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms of the plate-balancing environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


def _gaussian(std: float, shape: tuple[int, ...], max_sigma: float, device: str) -> torch.Tensor:
    """Zero-mean Gaussian samples with standard deviation ``std``, clipped at ``max_sigma`` standard deviations."""
    return (torch.randn(shape, device=device) * std).clamp(-max_sigma * std, max_sigma * std)


def reset_tilted_mount(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    mount_std: tuple[float, float, float],
    plate_yaw_std: float,
    ball_pos_range: dict[str, tuple[float, float]],
    max_sigma: float = 2.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
):
    """Reset the hand, the plate and the ball as one rigid setup tilted by a mounting error, with the plate turned on
    the fingertips.

    The mounting error is a rotation with Gaussian roll, pitch and yaw [rad] of standard deviations ``mount_std``,
    about the plate's default centre, so the plate stays at its default position and only the hand's orientation to
    gravity changes. The plate is turned on the fingertips about its normal by a Gaussian yaw [rad] of standard
    deviation ``plate_yaw_std``. The ball is placed at its default offset from the plate, shifted uniformly within
    ``ball_pos_range`` (keys ``x`` and ``y``) [m] in the plate frame. Samples are clipped at ``max_sigma`` standard
    deviations. All three start at rest.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.scene.num_envs, device=robot.device)
    n, device = len(env_ids), robot.device
    origins = env.scene.env_origins[env_ids]

    roll, pitch, yaw = (_gaussian(std, (n,), max_sigma, device) for std in mount_std)
    mount = quat_from_euler_xyz(roll, pitch, yaw)
    zeros = torch.zeros(n, device=device)
    plate_turn = quat_from_euler_xyz(zeros, zeros, _gaussian(plate_yaw_std, (n,), max_sigma, device))

    # the plate's default centre is the pivot of the mounting error
    plate_default = plate.data.default_root_pose.torch[env_ids]
    pivot = plate_default[:, :3]
    plate_quat = quat_mul(mount, quat_mul(plate_default[:, 3:7], plate_turn))

    robot_default = robot.data.default_root_pose.torch[env_ids]
    robot_pos = pivot + quat_apply(mount, robot_default[:, :3] - pivot)
    robot_quat = quat_mul(mount, robot_default[:, 3:7])

    # the ball's offset from the plate, shifted in the plate frame, before the plate's turn and the mounting error
    ball_offset = ball.data.default_root_pose.torch[env_ids, :3] - pivot
    for axis, key in enumerate(("x", "y")):
        low, high = ball_pos_range.get(key, (0.0, 0.0))
        ball_offset[:, axis] += torch.empty(n, device=device).uniform_(low, high)
    ball_pos = pivot + quat_apply(plate_quat, ball_offset)

    robot.write_root_pose_to_sim_index(root_pose=torch.cat([robot_pos + origins, robot_quat], dim=-1), env_ids=env_ids)
    plate.write_root_pose_to_sim_index(root_pose=torch.cat([pivot + origins, plate_quat], dim=-1), env_ids=env_ids)
    ball.write_root_pose_to_sim_index(
        root_pose=torch.cat([ball_pos + origins, ball.data.default_root_pose.torch[env_ids, 3:7]], dim=-1),
        env_ids=env_ids,
    )
    for asset in (plate, ball):
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=device), env_ids=env_ids)
