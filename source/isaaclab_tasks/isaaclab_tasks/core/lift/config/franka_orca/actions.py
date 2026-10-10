# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms of the Franka + ORCA-hand lift environments."""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass, replace


class VisualizedDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Differential IK action term that draws its target and its controlled frame.

    With ``debug_vis`` on, pose IK draws both as coordinate frames, the target one larger, and position IK as
    spheres, the target in the target color and the controlled point (the body with its offset) in the current
    color.
    """

    cfg: VisualizedDifferentialInverseKinematicsActionCfg

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                name = self.cfg.marker_name or self._body_name
                self.target_visualizer = VisualizationMarkers(self._marker_cfg(f"/Visuals/IK/{name}_target", True))
                self.current_visualizer = VisualizationMarkers(self._marker_cfg(f"/Visuals/IK/{name}_current", False))
            self.target_visualizer.set_visibility(True)
            self.current_visualizer.set_visibility(True)
        elif hasattr(self, "target_visualizer"):
            self.target_visualizer.set_visibility(False)
            self.current_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # the asset can be de-initialized, e.g. while the simulation stops
        if not self._asset.is_initialized:
            return
        root_pos_w = self._asset.data.root_pos_w.torch
        root_quat_w = self._asset.data.root_quat_w.torch
        target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
            root_pos_w, root_quat_w, self._ik_controller.ee_pos_des, self._ik_controller.ee_quat_des
        )
        current_pos_b, current_quat_b = self._compute_frame_pose()
        current_pos_w, current_quat_w = math_utils.combine_frame_transforms(
            root_pos_w, root_quat_w, current_pos_b, current_quat_b
        )
        # position IK leaves the target orientation unset; spheres do not show it anyway
        if self.cfg.controller.command_type == "position":
            target_quat_w = current_quat_w
        self.target_visualizer.visualize(target_pos_w, target_quat_w)
        self.current_visualizer.visualize(current_pos_w, current_quat_w)

    def _marker_cfg(self, prim_path: str, target: bool) -> VisualizationMarkersCfg:
        """Marker of the target or of the controlled frame: a frame for pose IK, a sphere for position IK."""
        if self.cfg.controller.command_type == "pose":
            marker_cfg = replace(FRAME_MARKER_CFG, prim_path=prim_path)
            scale = self.cfg.frame_scale * (1.5 if target else 1.0)
            marker_cfg.markers["frame"].scale = (scale,) * 3
            return marker_cfg
        color = self.cfg.target_color if target else self.cfg.current_color
        return VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=self.cfg.sphere_radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
                )
            },
        )


@configclass
class VisualizedDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration of :class:`VisualizedDifferentialInverseKinematicsAction`."""

    class_type: type = VisualizedDifferentialInverseKinematicsAction

    marker_name: str | None = None
    """Name of the markers under ``/Visuals/IK``; the body name if None."""

    frame_scale: float = 0.05
    """Scale of the frame marker of the controlled frame of pose IK; the target frame is drawn 1.5 times larger."""

    sphere_radius: float = 0.006
    """Radius of the sphere markers of position IK [m]."""

    target_color: tuple[float, float, float] = (0.0, 1.0, 0.0)
    """Color of the target sphere."""

    current_color: tuple[float, float, float] = (1.0, 0.0, 0.0)
    """Color of the sphere at the controlled point."""
