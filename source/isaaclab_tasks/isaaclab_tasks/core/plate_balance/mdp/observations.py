# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms of the plate-balancing environments."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab_newton.sensors import TactileSensor

    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def ball_pos_plate(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    """Position [m] of the ball's centre in the plate frame, shape (num_envs, 3)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]
    pos, _ = subtract_frame_transforms(
        plate.data.root_pos_w.torch, plate.data.root_quat_w.torch, ball.data.root_pos_w.torch
    )
    return pos


def ball_vel_plate(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    """Velocity [m/s] of the ball's centre relative to the plate's centre, in the plate frame, shape (num_envs, 3).

    The plate's own rotation is not removed: the plate turns slowly, and the difference is small next to the ball's
    rolling speed.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]
    vel_w = ball.data.root_lin_vel_w.torch - plate.data.root_lin_vel_w.torch
    return quat_apply_inverse(plate.data.root_quat_w.torch, vel_w)


def plate_pose_root(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position [m] and orientation (x, y, z, w) of the plate in the robot root frame, shape (num_envs, 7)."""
    robot: Articulation = env.scene[asset_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]
    pos, quat = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        plate.data.root_pos_w.torch,
        plate.data.root_quat_w.torch,
    )
    return torch.cat([pos, quat], dim=-1)


PAD_POINT_SPACING = 0.5e-3
"""Distance [m] between neighbouring points of a pad region in :func:`pad_surface`."""

PAD_FACING_COS = 0.5
"""Cosine of the largest angle between a pad point's normal and the plate's normal for the point to face the plate,
in :func:`pad_footprint_distance`."""


@functools.cache
def pad_surface(path: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Points [m] covering a pad region mesh about :data:`PAD_POINT_SPACING` apart, and its outward unit normals
    there, each of shape (num_points, 3). Loaded once per mesh and device.

    The mesh's triangles are subdivided to edges at most :data:`PAD_POINT_SPACING` long, and its vertices thinned to
    one per cube of that size.
    """
    mesh = trimesh.load(path, force="mesh")
    mesh.merge_vertices()
    vertices, faces = trimesh.remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge=PAD_POINT_SPACING, max_iter=20
    )
    mesh = trimesh.Trimesh(vertices, faces)
    _, keep = np.unique(np.floor(mesh.vertices / PAD_POINT_SPACING), axis=0, return_index=True)
    return (
        torch.tensor(np.asarray(mesh.vertices[keep]), dtype=torch.float32, device=device),
        torch.tensor(np.asarray(mesh.vertex_normals[keep]), dtype=torch.float32, device=device),
    )


def pad_footprint_distance(point_b: torch.Tensor, normal_b: torch.Tensor, pad_region: str) -> torch.Tensor:
    """Distance [m] of contact points from a pad as a plate sees it, shape (num_points,): from each point, across
    its plate's normal, to the nearest pad point facing the plate (normal within ``acos(PAD_FACING_COS)`` of the
    plate's). Zero for a point over the pad, at whatever depth; NaN points give NaN.

    Args:
        point_b: Contact points in the pad region's frame [m], shape (num_points, 3).
        normal_b: The plate's unit normal, pointing away from the pad, in the same frame, shape (num_points, 3).
        pad_region: Path of the pad region mesh, see :func:`pad_surface`.
    """
    points, normals = pad_surface(pad_region, str(point_b.device))
    distance_sq = torch.cdist(point_b, points).square()
    along_normal = normal_b @ points.T - (normal_b * point_b).sum(dim=-1, keepdim=True)
    across_sq = (distance_sq - along_normal.square()).clamp(min=0.0)
    across_sq = across_sq.masked_fill(normal_b @ normals.T < PAD_FACING_COS, torch.inf)
    return across_sq.amin(dim=-1).sqrt()


def pad_contacts(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    pad_regions: list[str],
    pad_tolerance: float,
    fingertips_cfg: SceneEntityCfg,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contact of each listed fingertip with the plate, and whether it is on the fingertip's pad.

    A fingertip touches the plate on its pad, as a tactile sensor would feel it, while its contact point, the
    force-weighted mean of its contacts, is over the pad as the plate sees it, at most ``pad_tolerance`` [m] past its
    edge (:func:`pad_footprint_distance`). How deep the point is does not count: the contact points the simulator
    reports sit several millimetres inside a fingertip that collides as its convex hull, whose flat faces span the
    curved pad. A contact across the pad's edge counts whole or not at all, by its mean point.

    The sensors must be filtered to the plate, so their force matrix holds only the fingertip-plate contact, and track
    their contact points; ``pad_regions`` holds the pad region mesh of each sensor's fingertip in its body frame (see
    :func:`pad_surface`), and ``fingertips_cfg`` names the sensors' bodies in the same order.

    Returns:
        The contact forces [N] in the world frame, shape (num_envs, num_sensors, 3); the contact points [m] in the
        fingertip frames, NaN without a contact, of the same shape; whether each is on its pad, shape
        (num_envs, num_sensors); and the fingertips' orientations (x, y, z, w) in the world frame, shape
        (num_envs, num_sensors, 4).
    """
    robot: Articulation = env.scene[fingertips_cfg.name]
    fingertip_pos_w = robot.data.body_pos_w.torch[:, fingertips_cfg.body_ids]
    fingertip_quat_w = robot.data.body_quat_w.torch[:, fingertips_cfg.body_ids]
    plate: RigidObject = env.scene[plate_cfg.name]
    plate_normal_w = quat_apply(plate.data.root_quat_w.torch, torch.tensor([0.0, 0.0, 1.0], device=env.device))
    forces, points, on_pad = [], [], []
    for i, name in enumerate(contact_sensor_names):
        data = env.scene.sensors[name].data
        point_b = quat_apply_inverse(
            fingertip_quat_w[:, i], data.contact_pos_w.torch.view(env.num_envs, 3) - fingertip_pos_w[:, i]
        )
        normal_b = quat_apply_inverse(fingertip_quat_w[:, i], plate_normal_w)
        # without a contact, the point is NaN and so is its distance, which is not on the pad
        on_pad.append(pad_footprint_distance(point_b, normal_b, pad_regions[i]) <= pad_tolerance)
        forces.append(data.force_matrix_w.torch.view(env.num_envs, 3))
        points.append(point_b)
    return torch.stack(forces, dim=1), torch.stack(points, dim=1), torch.stack(on_pad, dim=1), fingertip_quat_w


def plate_contact_forces_w(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    pad_regions: list[str] | None = None,
    pad_tolerance: float = 1.0e-3,
    fingertips_cfg: SceneEntityCfg | None = None,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    """Contact forces [N] between each listed fingertip and the plate, in the world frame, shape
    (num_envs, num_sensors, 3).

    The sensors must be filtered to the plate, so their force matrix holds only the fingertip-plate contact. With
    ``pad_regions``, a fingertip's force counts only while it touches the plate on its pad, and is zero otherwise
    (see :func:`pad_contacts`).
    """
    if pad_regions is None:
        return torch.stack(
            [env.scene.sensors[name].data.force_matrix_w.torch.view(env.num_envs, 3) for name in contact_sensor_names],
            dim=1,
        )
    force_w, _, on_pad, _ = pad_contacts(
        env, contact_sensor_names, pad_regions, pad_tolerance, fingertips_cfg, plate_cfg
    )
    return force_w * on_pad.unsqueeze(-1)


def plate_contact_forces_root(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    pad_regions: list[str] | None = None,
    pad_tolerance: float = 1.0e-3,
    fingertips_cfg: SceneEntityCfg | None = None,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Contact forces [N] between each listed fingertip and the plate, in the robot root frame, shape
    (num_envs, 3 * num_sensors); with ``pad_regions``, only those on the fingertips' pads (see
    :func:`plate_contact_forces_w`)."""
    force_w = plate_contact_forces_w(env, contact_sensor_names, pad_regions, pad_tolerance, fingertips_cfg, plate_cfg)
    robot: Articulation = env.scene[asset_cfg.name]
    root_quat_w = robot.data.root_quat_w.torch.unsqueeze(1).expand(-1, force_w.shape[1], -1)
    return quat_apply_inverse(root_quat_w, force_w).view(env.num_envs, -1)


def taxel_total_force_s(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Sum of one fingertip's per-taxel reaction forces [N], in its sensor frame, shape (num_envs, 3): the force the
    whole sensing pad feels, without where on the pad."""
    sensor: TactileSensor = env.scene.sensors[sensor_name]
    return sensor.data.forces_s.torch.sum(dim=1)


def taxel_forces_s(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Per-taxel reaction forces [N] of one fingertip in its sensor frame, shape (num_envs, 3 * num_taxels)."""
    sensor: TactileSensor = env.scene.sensors[sensor_name]
    return sensor.data.forces_s.torch.flatten(start_dim=1)


def pad_contact_forces_b(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    pad_regions: list[str],
    pad_tolerance: float,
    fingertips_cfg: SceneEntityCfg,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    """Tactile force: each listed fingertip's contact force [N] on its pad, in its own frame, zero off the pad (see
    :func:`pad_contacts`), shape (num_envs, 3 * num_sensors)."""
    force_w, _, on_pad, quat_w = pad_contacts(
        env, contact_sensor_names, pad_regions, pad_tolerance, fingertips_cfg, plate_cfg
    )
    return quat_apply_inverse(quat_w, force_w * on_pad.unsqueeze(-1)).view(env.num_envs, -1)


def pad_contact_directions_b(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    pad_regions: list[str],
    pad_tolerance: float,
    fingertips_cfg: SceneEntityCfg,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    """Tactile force direction: the unit direction of each listed fingertip's contact force on its pad, in its own
    frame, zero without a contact on the pad (see :func:`pad_contacts`), shape (num_envs, 3 * num_sensors)."""
    force_w, _, on_pad, quat_w = pad_contacts(
        env, contact_sensor_names, pad_regions, pad_tolerance, fingertips_cfg, plate_cfg
    )
    force_b = quat_apply_inverse(quat_w, force_w * on_pad.unsqueeze(-1))
    return torch.nn.functional.normalize(force_b, dim=-1).view(env.num_envs, -1)


def pad_contact_points_b(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    pad_regions: list[str],
    pad_tolerance: float,
    fingertips_cfg: SceneEntityCfg,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    """Tactile contact position: each listed fingertip's contact point [m] on its pad, in its own frame, zero (the
    fingertip's origin, off the pad) without a contact on the pad (see :func:`pad_contacts`), shape
    (num_envs, 3 * num_sensors)."""
    _, point_b, on_pad, _ = pad_contacts(
        env, contact_sensor_names, pad_regions, pad_tolerance, fingertips_cfg, plate_cfg
    )
    return torch.where(on_pad.unsqueeze(-1), point_b, 0.0).view(env.num_envs, -1)


def pad_contact_binary(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    pad_regions: list[str],
    pad_tolerance: float,
    fingertips_cfg: SceneEntityCfg,
    force_threshold: float,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    """Tactile contact: 1 for each listed fingertip whose contact force on its pad is above ``force_threshold`` [N],
    else 0 (see :func:`pad_contacts`), shape (num_envs, num_sensors)."""
    force_w, _, on_pad, _ = pad_contacts(
        env, contact_sensor_names, pad_regions, pad_tolerance, fingertips_cfg, plate_cfg
    )
    return (on_pad & (torch.linalg.norm(force_w, dim=-1) > force_threshold)).float()
