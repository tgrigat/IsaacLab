# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO runner configuration for the ORCA-hand plate-balancing environments."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlRNNModelCfg


@configclass
class OrcaHandPlateBalancePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """State PPO, actor and critic observing the task state and the proprioception."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 200
    experiment_name = "plate_balance_orca_hand"
    logger = "wandb"
    wandb_project = "plate_balance_orca_hand"
    obs_groups = {"actor": ["policy", "proprio"], "critic": ["policy", "proprio"]}
    actor = RslRlMLPModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
        obs_normalization=True,
        hidden_dims=[512, 256, 128],
        activation="elu",
    )
    critic = RslRlMLPModelCfg(obs_normalization=True, hidden_dims=[512, 256, 128], activation="elu")
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class OrcaHandPlateBalanceCoPPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Recurrent asymmetric PPO of Pan et al. (2026), "Beyond Binary": a GRU actor observing what the real hand
    senses, and a GRU critic also observing the ball and the plate."""

    num_steps_per_env = 16
    max_iterations = 3000
    save_interval = 200
    experiment_name = "plate_balance_orca_hand_cop"
    logger = "wandb"
    wandb_project = "plate_balance_orca_hand"
    obs_groups = {"actor": ["actor"], "critic": ["critic"]}
    actor = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    critic = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )
