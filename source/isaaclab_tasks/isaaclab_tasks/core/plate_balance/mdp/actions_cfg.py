# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the action terms of the plate-balancing environments."""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.utils import configclass

from .actions import EMARelativeJointPositionAction


@configclass
class EMARelativeJointPositionActionCfg(JointActionCfg):
    """Configuration for :class:`EMARelativeJointPositionAction`; :attr:`scale` is the joint target increment [rad]
    per unit action."""

    class_type: type[EMARelativeJointPositionAction] = EMARelativeJointPositionAction

    alpha: float = 0.5
    """Weight of the new increment in the exponential moving average, in (0, 1]."""

    clip_value: float = 1.0
    """Magnitude the raw action is clipped to before scaling."""
