# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms of the Franka + ORCA-hand lift environments."""

from __future__ import annotations

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
