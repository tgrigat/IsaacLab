# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the target point of the ball on the plate."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from .commands import PlateTargetCommand


@configclass
class PlateTargetCommandCfg(CommandTermCfg):
    """Configuration for :class:`PlateTargetCommand`."""

    class_type: type[PlateTargetCommand] = PlateTargetCommand

    plate_name: str = "plate"
    """Name of the plate in the scene."""

    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate")
    """The plate, whose frame the target is in."""

    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")
    """The ball, whose distance to the target is the ``position_error`` metric."""

    @configclass
    class Ranges:
        """Uniform ranges of the target in the plate frame."""

        pos_x: tuple[float, float] = MISSING
        """Range of the target's x [m]."""

        pos_y: tuple[float, float] = MISSING
        """Range of the target's y [m]."""

    ranges: Ranges = MISSING
    """Ranges of the target."""

    marker_height: float = 0.0
    """Height [m] of the target marker above the plate's centre, in the plate frame; the plate's half thickness puts
    it on the top face."""

    target_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/plate_target",
        markers={
            "target": sim_utils.CylinderCfg(
                radius=0.006,
                height=0.001,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
            ),
        },
    )
    """Marker of the target, a flat disc."""
