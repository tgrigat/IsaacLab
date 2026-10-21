# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distillation runner configuration for the ORCA-hand plate-balancing tactile students."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlRNNModelCfg,
)

from .rsl_rl_ppo_cfg import OrcaHandPlateBalancePPORunnerCfg


@configclass
class OrcaHandPlateBalanceDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distills the NoTaxels PPO policy, loaded with ``--checkpoint``, to a recurrent tactile student.

    The student sees neither the ball nor the plate, so it is recurrent, to infer them, and their motion, from how the
    tactile signal changes.
    """

    num_steps_per_env = 48
    max_iterations = 1000
    save_interval = 100
    experiment_name = "plate_balance_orca_hand_distillation"
    logger = "wandb"
    wandb_project = "plate_balance_orca_hand"
    obs_groups = {"student": ["student"], "teacher": ["policy", "proprio"]}
    student = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    # the PPO actor, to load the teacher's weights into
    teacher = OrcaHandPlateBalancePPORunnerCfg().actor
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=16,
        max_grad_norm=1.0,
    )
