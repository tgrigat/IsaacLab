# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ORCA-hand ball-on-plate balancing environments."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

FINGERTIP_VARIANTS = ["NoTaxels", "ConvexTaxels"]
"""Task suffix and config class infix of each fingertip collision method built for the hand alone: convex fingertip
hulls without taxels, and convex taxels over a convex fingertip hull. Triangle-mesh taxels (``mesh_taxels``) are left
out: the plate keeps rocking on them, and the ring finger and thumb lose contact."""

for variant in FINGERTIP_VARIANTS:
    gym.register(
        id=f"Isaac-PlateBalance-OrcaHand-{variant}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.orca_hand_env_cfg:OrcaHandPlateBalance{variant}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OrcaHandPlateBalancePPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )

STUDENT_VARIANTS = ["Force", "Position", "PositionDirection", "Binary"]
"""Tactile signal of each student the NoTaxels policy is distilled to: each pad's contact force vector, its contact
point, its contact point and force direction, or whether it is in contact."""

for variant in STUDENT_VARIANTS:
    gym.register(
        id=f"Isaac-PlateBalance-OrcaHand-NoTaxels-Student{variant}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.orca_hand_env_cfg:OrcaHandPlateBalanceStudent{variant}EnvCfg",
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_distillation_cfg:OrcaHandPlateBalanceDistillationRunnerCfg"
            ),
            "default_agent": "rsl_rl",
        },
    )
