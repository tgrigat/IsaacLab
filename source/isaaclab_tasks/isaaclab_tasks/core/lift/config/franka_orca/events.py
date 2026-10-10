# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms of the Franka + ORCA-hand lift environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
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
