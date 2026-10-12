# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka + ORCA reward terms.

:func:`finger_thumb_opposition` and :func:`fingertip_min_distance` are optional, ported from the Isaac Lab 2.3
dexsuite FR3+ORCA config. Both address grasp failures seen with this hand on the 2.3 port: the policy brought the
closest fingertip to ~7 cm of the object and stalled there, and closed grasps without the thumb. They are off
(``None``) in :class:`~.franka_orca_env_cfg.FrankaOrcaRewardCfg` by default, so the baseline matches Kuka-Allegro.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import RewardTermCfg, SceneEntityCfg

from ...mdp.rewards import _ProgressReward

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def joint_action_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, action_name: str = "action") -> torch.Tensor:
    """Penalize the actions of some joints using L2 squared kernel.

    Assumes the action term commands every joint of the asset in its joint order, as the relative joint position
    action of this robot does, so the joint ids index the action.

    Args:
        env: The environment instance.
        asset_cfg: The robot, with the joints to penalize.
        action_name: Name of the action term.

    Returns:
        Sum of the squared actions, shape (num_envs,).
    """
    actions = env.action_manager.get_term(action_name).raw_actions
    return torch.sum(torch.square(actions[:, asset_cfg.joint_ids]), dim=1)


def finger_thumb_opposition(env: ManagerBasedRLEnv, thumb_name: str, finger_names: list[str]) -> torch.Tensor:
    """Reward fingers pressing on the object against the thumb, i.e. a pinch.

    Each sensor reports the object's reaction force on that fingertip. A finger and the thumb squeezing the
    object from opposite sides see opposite forces, so ``-dot(finger, thumb)`` is positive for a pinch, grows
    with grip force and is zero while either side is out of contact. Summed over the fingers.

    The product is in N^2, far larger than the other ~0-1 terms, so weight it well below 1.

    Args:
        env: The environment instance.
        thumb_name: Name of the thumb's object contact sensor.
        finger_names: Names of the other fingers' object contact sensors.

    Returns:
        Opposition [N^2], shape (num_envs,).
    """
    thumb: ContactSensor = env.scene.sensors[thumb_name]
    thumb_force = thumb.data.normal_force_matrix_w.torch.view(env.num_envs, 3)
    opposition = torch.zeros(env.num_envs, device=env.device)
    for name in finger_names:
        finger: ContactSensor = env.scene.sensors[name]
        finger_force = finger.data.normal_force_matrix_w.torch.view(env.num_envs, 3)
        opposition -= torch.sum(finger_force * thumb_force, dim=-1)
    return opposition


def fingertip_min_distance(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the closest of the given fingertips approaching the object, with a tanh kernel.

    Complements :func:`~isaaclab_tasks.core.lift.mdp.object_ee_distance`, which takes the farthest body and
    is nearly saturated a few centimetres out. A tight ``std`` here keeps a gradient into the last centimetres.

    Args:
        env: The environment instance.
        std: Standard deviation of the tanh kernel [m].
        asset_cfg: The robot, with the fingertip bodies to consider.
        object_cfg: The object.

    Returns:
        Reward in (0, 1], shape (num_envs,).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    fingertip_pos = robot.data.body_pos_w.torch[:, asset_cfg.body_ids]
    object_pos = obj.data.root_pos_w.torch
    min_distance = torch.norm(fingertip_pos - object_pos[:, None], dim=-1).min(dim=-1).values
    return 1 - torch.tanh(min_distance / std)


class hand_object_progress(_ProgressReward):
    """Reward every step that brings the hand closer to the object than ever before in the episode.

    The hand's distance is the mean distance from the hand bodies (e.g. the palm and the fingertips) to the object
    center. See :class:`~isaaclab_tasks.core.lift.mdp.rewards._ProgressReward` for the bookkeeping: 1.0 is paid each
    time the distance beats the episode's best by :paramref:`min_improvement`, so the hand is led to a far object,
    but staying near it earns nothing. Unlike the lift ``fingers_to_object`` term, which pays a tenth until the
    thumb and a finger touch the object, it pays without contact; and unlike a reward for being near the object, it
    does not pay the policy to hover next to a floating object rather than risk pushing it away.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # no command: the bar is seeded once per episode
        self._no_command = torch.zeros(env.num_envs, 1, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        min_improvement: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        """Compute the progress reward.

        Args:
            env: The environment instance.
            asset_cfg: The robot, with the hand bodies.
            min_improvement: Distance [m] the hand must gain on the episode best to be paid again.
            object_cfg: The object.

        Returns:
            1.0 where the hand reached a new closest distance, else 0, shape (num_envs,).
        """
        asset: Articulation = env.scene[asset_cfg.name]
        obj: RigidObject = env.scene[object_cfg.name]
        hand_pos = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
        distance = torch.linalg.norm(hand_pos - obj.data.root_pos_w.torch[:, None], dim=-1).mean(dim=-1)
        # a diverged simulation leaves non-finite poses, which must not become the bar
        distance = torch.nan_to_num(distance, nan=float("inf"))
        gate = torch.isfinite(distance)
        return self._progress(distance, gate, min_improvement, self._no_command)
