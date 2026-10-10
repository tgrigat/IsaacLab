# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO runner configuration for the FrankaOrca lift environments."""

from isaaclab.utils import configclass

from ...kuka_allegro.agents.rsl_rl_ppo_cfg import KukaAllegroPPORunnerCfg


@configclass
class FrankaOrcaPPORunnerCfg(KukaAllegroPPORunnerCfg):
    """The Kuka-Allegro state PPO setup, logged under its own experiment."""

    experiment_name = "lift_franka_orca"
