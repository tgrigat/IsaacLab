# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Target point on the plate for the ball."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_apply

from .observations import ball_pos_plate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import PlateTargetCommandCfg


class PlateTargetCommand(CommandTerm):
    """Target position of the ball on the plate, in the plate frame.

    The command is the target's ``(x, y)`` on the plate's top face, sampled uniformly in the configured ranges. It is
    fixed to the plate, so it moves and tilts with it, and the ball is on target when its position in the plate frame
    matches it, wherever the plate is.
    """

    cfg: PlateTargetCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: PlateTargetCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.plate: RigidObject = env.scene[cfg.plate_name]
        self.target_p = torch.zeros(self.num_envs, 2, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PlateTargetCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Target ``(x, y)`` [m] of the ball in the plate frame. Shape is (num_envs, 2)."""
        return self.target_p

    def _update_metrics(self):
        ball_p = ball_pos_plate(self._env, self.cfg.ball_cfg, self.cfg.plate_cfg)
        self.metrics["position_error"] = torch.linalg.norm(ball_p[:, :2] - self.target_p, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        num_envs = len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        r = torch.empty(num_envs, device=self.device)
        self.target_p[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.target_p[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
            self.target_visualizer.set_visibility(True)
        elif hasattr(self, "target_visualizer"):
            self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.plate.is_initialized:
            return
        # on the plate's top face, where the ball's lowest point is when on target
        target = torch.zeros(self.num_envs, 3, device=self.device)
        target[:, :2] = self.target_p
        target[:, 2] = self.cfg.marker_height
        plate_quat_w = self.plate.data.root_quat_w.torch
        self.target_visualizer.visualize(
            self.plate.data.root_pos_w.torch + quat_apply(plate_quat_w, target), plate_quat_w
        )
