# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Show the staged in-hand FrankaOrca task at a curriculum difficulty picked with a slider, with or without a policy.

The "Curriculum" section of the Newton viewer's side panel has a difficulty slider; on release every environment is
reset at that difficulty through the task's own curriculum terms, so the reset modes drawn and the gravity are exactly
the ones training uses at that difficulty, and every later episode stays at it. The panel lists the stage each
environment started in.

Without ``--checkpoint`` the actions are zero, so the arm and fingers hold their reset pose while the cube does what
the physics does with it (floats without gravity, falls with it). With ``--checkpoint``, a trained RSL-RL policy acts,
so it can be watched stage by stage. The checkpoint must come from the same task variant: the fingertip method decides
the observations.

Usage:

.. code-block:: bash

    PYGLFW_LIBRARY_VARIANT=x11 uv run python scripts/tools/franka_orca/view_in_hand_curriculum.py --viz newton_gl
    PYGLFW_LIBRARY_VARIANT=x11 uv run python scripts/tools/franka_orca/view_in_hand_curriculum.py --viz newton_gl \
        --checkpoint logs/rsl_rl/lift_franka_orca/<run>/model_<iteration>.pt
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="Isaac-Lift-FrankaOrca-InHand-Staged-NoTaxels", help="Staged in-hand task.")
parser.add_argument("--num_envs", type=int, default=9, help="Number of environments, shown side by side.")
parser.add_argument("--difficulty", type=float, default=0.0, help="Start difficulty fraction, in [0, 1].")
parser.add_argument("--checkpoint", help="RSL-RL checkpoint of a policy for the task; zero actions without it.")
add_launcher_args(parser)
# unrecognized arguments are Hydra-style task config overrides; Hydra reads them from sys.argv
args_cli, hydra_overrides = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_overrides]

import gymnasium as gym
import torch

from isaaclab.utils import to_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config


def load_policy(env: gym.Env, agent_cfg, checkpoint: str):
    """Wrap the environment for RSL-RL and load the checkpoint's inference policy.

    Returns:
        The wrapped environment and the policy, which maps its observations to actions.
    """
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    print(f"[INFO] Policy loaded from {checkpoint}")
    return env, runner.get_inference_policy(device=env.unwrapped.device)


def main():
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point" if args_cli.checkpoint else None)
    env_cfg.scene.num_envs = args_cli.num_envs
    # every environment in view, and a small bank taken as it comes, so the viewer opens quickly
    env_cfg.sim.default_visualizer_cfg.visible_env_indices = None
    env_cfg.events.aim_camera = None
    reset_cfg = env_cfg.events.conditional_reset.params
    reset_cfg["buffer_size_per_group"] = 32
    reset_cfg["diversity_feature"] = None

    with launch_simulation(env_cfg, args_cli):
        gym_env = gym.make(args_cli.task, cfg=env_cfg)
        env = gym_env.unwrapped
        adr = env.curriculum_manager.cfg.adr
        max_difficulty = adr.params["max_difficulty"]
        reset_term = env.event_manager.get_term_cfg("conditional_reset")
        reset_params = reset_term.params
        gravity_params = env.event_manager.get_term_cfg("variable_gravity").params
        if args_cli.checkpoint:
            policy_env, policy = load_policy(gym_env, agent_cfg, args_cli.checkpoint)
        else:
            actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)

        # look at the middle of the environment grid from the front left, above the tables
        origins = env.scene.env_origins
        center = origins.mean(dim=0).tolist()
        extent = float((origins.amax(dim=0) - origins.amin(dim=0)).norm()) + 2.0
        env.sim.set_camera_view(
            eye=(center[0] - 0.8 * extent, center[1] + 0.5 * extent, 0.6 * extent),
            target=(center[0] - 0.5, center[1], 0.4),
        )

        state = {"difficulty": min(max(args_cli.difficulty, 0.0), 1.0), "pending": True}

        def pin_difficulty():
            """Hold every environment at the slider's difficulty; the scheduler moves it by one per episode."""
            level = state["difficulty"] * max_difficulty
            adr.params["min_difficulty"] = level
            adr.func.current_difficulties[:] = level

        def reset_all():
            pin_difficulty()
            env.reset()

        def stage_of_envs() -> list[str]:
            """Reset mode each environment's current episode started in."""
            func = reset_term.func
            if func._monitor is None or (func._playing_row < 0).any():
                return []
            modes = func._mode_names
            mode = (func._playing_row // reset_params["buffer_size_per_group"]) % len(modes)
            return [modes[m] for m in mode.tolist()]

        def ui(imgui):
            if not imgui.collapsing_header("Curriculum"):
                return
            imgui.text("Difficulty (mean ADR level / max)")
            changed, value = imgui.slider_float("##difficulty", state["difficulty"], 0.0, 1.0, "%.2f")
            if changed:
                state["difficulty"] = value
            # reset once the slider is let go, not on every frame of a drag
            if imgui.is_item_deactivated_after_edit():
                state["pending"] = True
            if imgui.button("Reset"):
                state["pending"] = True
            # one row per environment, all at the slider's level
            weights = torch.as_tensor(reset_params["mode_weights"]).reshape(-1, len(reset_term.func._mode_names))
            drawn = (weights > 0).any(dim=0).tolist()
            unlocked = [name for name, d in zip(reset_term.func._mode_names, drawn) if d]
            imgui.text(f"Stages drawn: {', '.join(unlocked) or '-'}")
            gravity = gravity_params["gravity_distribution_params"][0][2]
            imgui.text(f"Gravity: {gravity:.2f} m/s^2")
            imgui.separator()
            for index, stage in enumerate(stage_of_envs()):
                imgui.text(f"env {index}: {stage}")

        viewer = next((viz for viz in env.sim.visualizers if hasattr(viz, "register_ui_callback")), None)
        if viewer is None:
            raise RuntimeError("No Newton viewer to put the slider in; pass --viz newton_gl.")
        viewer.register_ui_callback(ui, position="panel")

        # the viewer reports running only once it has drawn a frame, so the check follows the step
        while True:
            if state["pending"]:
                state["pending"] = False
                reset_all()
                if args_cli.checkpoint:
                    obs = policy_env.get_observations()
                    policy.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
            pin_difficulty()
            if args_cli.checkpoint:
                with torch.inference_mode():
                    obs, _, dones, _ = policy_env.step(policy(obs))
                policy.reset(dones)
            else:
                env.step(actions)
            if not viewer.is_running():
                break
        env.close()


if __name__ == "__main__":
    main()
