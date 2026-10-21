# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lift FrankaOrca lift and in-hand lift environments."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

FINGERTIP_VARIANTS = ["Primitives", "NoTaxels", "ConvexTaxels", "MeshTaxels", "MeshFingertips"]
"""Task suffix and config class infix of each fingertip collision method (``FINGERTIP_METHODS``), from cheapest to
most exact: every link a capsule or box; convex fingertip hulls; convex taxels over a convex fingertip hull;
triangle-mesh taxels over CoACD fingertip hulls; triangle-mesh taxels over a triangle-mesh fingertip. The first two
have no taxels, which keeps MJWarp's contacts from overflowing in dense grasps."""

TASKS = {
    "Isaac-Lift-FrankaOrca": "franka_orca_env_cfg:FrankaOrcaLift",
    "Isaac-Lift-FrankaOrca-InHand": "franka_orca_in_hand_env_cfg:FrankaOrcaInHandLift",
    "Isaac-Lift-FrankaOrca-InHand-Staged": "franka_orca_in_hand_env_cfg:FrankaOrcaInHandStagedLift",
}
"""Task id prefix and config class prefix of each task."""

for task, cfg in TASKS.items():
    for variant in FINGERTIP_VARIANTS:
        gym.register(
            id=f"{task}-{variant}",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.{cfg}{variant}EnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaOrcaPPORunnerCfg",
                "default_agent": "rsl_rl",
            },
        )

# the NoTaxels staged task observing binary fingertip contacts instead of the fingertip contact forces
gym.register(
    id="Isaac-Lift-FrankaOrca-InHand-Staged-NoTaxels-BinaryContacts",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_orca_in_hand_env_cfg:FrankaOrcaInHandStagedLiftNoTaxelsBinaryContactsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaOrcaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# scripted tests of the fingertip IK (scripts/environments/state_machine/lift_cube_franka_orca_sm.py), no agent
gym.register(
    id="Isaac-Lift-Cube-FrankaOrca-FingertipIK-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_orca_ik_env_cfg:FrankaOrcaFingertipIKLiftEnvCfg",
    },
)
