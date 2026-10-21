# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms of the plate-balancing environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .observations import ball_pos_plate, plate_contact_forces_w

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg


def plate_tilt(env: ManagerBasedRLEnv, plate_cfg: SceneEntityCfg = SceneEntityCfg("plate")) -> torch.Tensor:
    """Angle [rad] between the plate's normal and the world's up axis."""
    plate: RigidObject = env.scene[plate_cfg.name]
    # z component of the plate's z axis in the world frame
    x, y = plate.data.root_quat_w.torch[:, 0], plate.data.root_quat_w.torch[:, 1]
    return torch.acos((1.0 - 2.0 * (x.square() + y.square())).clamp(-1.0, 1.0))


def ball_off_plate(
    env: ManagerBasedRLEnv,
    half_extents: tuple[float, float],
    min_height: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    """Whether the ball's centre left the plate: past its edge, at ``half_extents`` [m] in the plate frame, or lower
    than ``min_height`` [m] above the plate's centre."""
    pos = ball_pos_plate(env, ball_cfg, plate_cfg)
    outside = (pos[:, 0].abs() > half_extents[0]) | (pos[:, 1].abs() > half_extents[1])
    return outside | (pos[:, 2] < min_height)


def plate_dropped(
    env: ManagerBasedRLEnv,
    max_drop: float,
    max_tilt: float,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    """Whether the plate fell off the fingers: more than ``max_drop`` [m] below its initial height, or tilted more
    than ``max_tilt`` [rad]."""
    plate: RigidObject = env.scene[plate_cfg.name]
    drop = plate.data.default_root_pose.torch[:, 2] + env.scene.env_origins[:, 2] - plate.data.root_pos_w.torch[:, 2]
    return (drop > max_drop) | (plate_tilt(env, plate_cfg) > max_tilt)


class fingertip_off_plate(ManagerTermBase):
    """Whether any of the listed fingertips has been off the plate for more than ``max_air_time`` [s]: its contact
    force on the plate at most ``force_threshold`` [N]; with ``pad_regions``, its force on its pad (see
    :func:`plate_contact_forces_w`).

    The time off the plate is counted here rather than taken from the sensors' air time, which counts any contact,
    so that with pad regions a fingertip touching the plate off its pad is off it too.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._air_time = torch.zeros(env.num_envs, len(cfg.params["contact_sensor_names"]), device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        self._air_time[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        contact_sensor_names: list[str],
        max_air_time: float,
        force_threshold: float,
        pad_regions: list[str] | None = None,
        pad_tolerance: float = 1.0e-3,
        fingertips_cfg: SceneEntityCfg | None = None,
        plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    ) -> torch.Tensor:
        forces = plate_contact_forces_w(
            env, contact_sensor_names, pad_regions, pad_tolerance, fingertips_cfg, plate_cfg
        )
        in_contact = torch.linalg.norm(forces, dim=-1) > force_threshold
        self._air_time = torch.where(in_contact, 0.0, self._air_time + env.step_dt)
        return (self._air_time > max_air_time).any(dim=-1)
