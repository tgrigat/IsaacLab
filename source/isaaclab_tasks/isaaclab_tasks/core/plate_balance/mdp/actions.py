# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms of the plate-balancing environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import EMARelativeJointPositionActionCfg


class EMARelativeJointPositionAction(JointAction):
    r"""Joint position targets moved by smoothed increments, as in Pan et al. (2026), "Beyond Binary".

    Each step, the action is clipped to ``[-clip_value, clip_value]``, scaled, smoothed by an exponential moving
    average and added to the previous joint position target, which is kept within the soft joint limits:

    .. math::

        \delta_t = \alpha \, s \, \text{clip}(a_t) + (1 - \alpha) \, \delta_{t-1}, \qquad
        q^*_t = \text{clamp}(q^*_{t-1} + \delta_t, q_{\min}, q_{\max})

    The targets start, on reset, at the joints' positions, and the increment at zero. Unlike
    :class:`~isaaclab.envs.mdp.actions.RelativeJointPositionAction`, the increment is added to the target, not to the
    measured position, so a fingertip pressing on the plate keeps its force instead of drifting with the contact.
    """

    cfg: EMARelativeJointPositionActionCfg

    def __init__(self, cfg: EMARelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._increment = torch.zeros_like(self._raw_actions)
        self._targets = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        self._lower, self._upper = limits[..., 0], limits[..., 1]

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        step = actions.clamp(-self.cfg.clip_value, self.cfg.clip_value) * self._scale
        self._increment = self.cfg.alpha * step + (1.0 - self.cfg.alpha) * self._increment
        self._targets = torch.clamp(self._targets + self._increment, self._lower, self._upper)
        self._processed_actions = self._targets

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0.0
        self._increment[env_ids] = 0.0
        # the reset events, which run before the action reset, have written the joint positions
        self._targets[env_ids] = self._asset.data.joint_pos.torch[env_ids][:, self._joint_ids]
