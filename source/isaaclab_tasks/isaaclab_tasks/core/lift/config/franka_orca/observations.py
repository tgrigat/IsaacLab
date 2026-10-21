# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka + ORCA observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab_newton.sensors import TactileSensor

    from isaaclab.envs import ManagerBasedRLEnv


def taxel_forces_s(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Per-taxel reaction forces [N] of one fingertip in its sensor frame, shape (num_envs, 3 * num_taxels)."""
    sensor: TactileSensor = env.scene.sensors[sensor_name]
    return sensor.data.forces_s.torch.flatten(start_dim=1)


def fingers_in_contact(env: ManagerBasedRLEnv, contact_sensor_names: list[str], threshold: float) -> torch.Tensor:
    """Binary contact of each listed fingertip with anything, shape (num_envs, num_sensors).

    1 where the net normal contact force of the sensor's body exceeds ``threshold`` [N], else 0. The net force sums
    every contact of the body (object, table, arm and other fingers), not only the filtered ones.
    """
    forces = torch.stack(
        [env.scene.sensors[name].data.net_normal_forces_w.torch.view(env.num_envs, 3) for name in contact_sensor_names],
        dim=1,
    )
    return (torch.linalg.norm(forces, dim=-1) > threshold).float()
