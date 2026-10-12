# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lift FrankaOrca reorient, lift and in-hand lift environments."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Reorient-FrankaOrca",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_orca_env_cfg:FrankaOrcaReorientEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaOrcaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-Lift-FrankaOrca",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_orca_env_cfg:FrankaOrcaLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaOrcaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-Lift-FrankaOrca-InHand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_orca_in_hand_env_cfg:FrankaOrcaInHandLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaOrcaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-Lift-FrankaOrca-InHand-Staged",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_orca_in_hand_env_cfg:FrankaOrcaInHandStagedLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaOrcaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# ablations with plain fingertip hulls and no tactile sensing, which keep MJWarp's contacts from overflowing
gym.register(
    id="Isaac-Lift-FrankaOrca-NoTaxels",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_orca_env_cfg:FrankaOrcaLiftNoTaxelsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaOrcaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-Lift-FrankaOrca-InHand-NoTaxels",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_orca_in_hand_env_cfg:FrankaOrcaInHandLiftNoTaxelsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaOrcaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-Lift-FrankaOrca-InHand-Staged-NoTaxels",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_orca_in_hand_env_cfg:FrankaOrcaInHandStagedLiftNoTaxelsEnvCfg",
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
