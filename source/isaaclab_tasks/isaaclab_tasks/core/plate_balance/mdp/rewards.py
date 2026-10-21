# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms of the plate-balancing environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import ball_pos_plate, plate_contact_forces_w
from .terminations import plate_tilt

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_target_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    """``exp(-d^2 / std^2)`` of the distance ``d`` [m] on the plate between the ball and its target."""
    target = env.command_manager.get_command(command_name)
    d2 = (ball_pos_plate(env, ball_cfg, plate_cfg)[:, :2] - target).square().sum(dim=-1)
    return torch.exp(-d2 / std**2)


def fingers_share_plate(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    force_scale: float,
    pad_regions: list[str] | None = None,
    pad_tolerance: float = 1.0e-3,
    fingertips_cfg: SceneEntityCfg | None = None,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    """Mean over the listed fingertips of each one's contact force [N] on the plate over ``force_scale`` [N], each at
    most 1; with ``pad_regions``, only the force on each fingertip's pad (see :func:`plate_contact_forces_w`).

    Each fingertip earns its share only by carrying the plate, so holding it on a tripod, as policies otherwise learn
    to with the thumb, index and pinky, forfeits two fifths. The sensors must be filtered to the plate.
    """
    forces = torch.linalg.norm(
        plate_contact_forces_w(env, contact_sensor_names, pad_regions, pad_tolerance, fingertips_cfg, plate_cfg), dim=-1
    )
    return (forces / force_scale).clamp(max=1.0).mean(dim=-1)


def plate_tilt_l2(env: ManagerBasedRLEnv, plate_cfg: SceneEntityCfg = SceneEntityCfg("plate")) -> torch.Tensor:
    """Squared tilt [rad^2] of the plate from level."""
    return plate_tilt(env, plate_cfg).square()
