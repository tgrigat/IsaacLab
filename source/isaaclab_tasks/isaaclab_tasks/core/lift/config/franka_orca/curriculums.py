# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum modifiers of the Franka + ORCA-hand lift environments, for :class:`~isaaclab.envs.mdp.modify_term_cfg`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _difficulty_frac(env: ManagerBasedRLEnv, difficulty_term_str: str) -> float:
    """Mean difficulty of a :class:`~isaaclab.envs.mdp.DifficultyScheduler` term, normalized to [0, 1]."""
    return getattr(env.curriculum_manager.cfg, difficulty_term_str).func.difficulty_frac


def stage_mode_weights(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: torch.Tensor | Sequence[float] | None,
    stage_fracs: Sequence[tuple[float, float]],
    difficulty_term_str: str,
) -> torch.Tensor:
    """Per-environment weights of the reset modes, uniform over the modes whose difficulty window holds the
    environment's own difficulty.

    Intended for the ``mode_weights`` of :class:`~isaaclab_tasks.core.lift.mdp.events.conditional_reset`. A mode is
    drawn while the environment's difficulty fraction (its level over the maximum) is in the mode's window
    ``[start, end)`` (an end of 1 includes the maximum), so a mode can stay in the mix after the next one unlocks, or
    drop out once a later one replaces it. The level falls with failures, so a mode can also come back.

    Each environment draws by its own level rather than by the mean, so the share of each mode across the
    environments changes gradually. Gated by the mean, a mode that the policy cannot solve yet unlocks for every
    environment at once, its failures pull the mean back under the threshold, it locks again, and the mix flips
    every few iterations.

    Args:
        env: The environment.
        env_ids: Unused; the weights of every environment are updated, as the levels are cheap to read.
        data: Current mode weights.
        stage_fracs: Difficulty window ``(start, end)`` of each mode, in the order of the modes. The windows must
            cover [0, 1] between them, so some mode is always drawn.
        difficulty_term_str: Name of the difficulty scheduler curriculum term.

    Returns:
        The new weights, shape (num_envs, num_modes).
    """
    term_cfg = getattr(env.curriculum_manager.cfg, difficulty_term_str)
    frac = term_cfg.func.current_difficulties / max(term_cfg.params["max_difficulty"], 1)
    start, end = torch.tensor(stage_fracs, dtype=torch.float, device=env.device).unbind(-1)
    frac = frac.unsqueeze(-1)
    return ((start <= frac) & ((frac < end) | (end >= 1.0))).float()


def _interpolate(initial: Any, final: Any, alpha: float) -> Any:
    """Interpolate the leaf scalars of nested sequences."""
    if isinstance(initial, Sequence):
        return type(initial)(_interpolate(i, f, alpha) for i, f in zip(initial, final))
    return initial + alpha * (final - initial)


def delayed_interpolate_fn(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: Any,
    initial_value: Any,
    final_value: Any,
    start_frac: float,
    difficulty_term_str: str,
    end_frac: float = 1.0,
) -> Any:
    """Hold a parameter at its initial value until a difficulty fraction, then interpolate it to the final value.

    Like :func:`~isaaclab.envs.mdp.initial_final_interpolate_fn`, but the ramp spans difficulty fractions
    [:paramref:`start_frac`, :paramref:`end_frac`] instead of [0, 1], so it can sit between curriculum stages.

    Args:
        env: The environment.
        env_ids: Unused; the value is shared by all environments.
        data: Current value of the parameter.
        initial_value: Value up to :paramref:`start_frac`.
        final_value: Value from :paramref:`end_frac` on.
        start_frac: Difficulty fraction at which the ramp starts.
        difficulty_term_str: Name of the difficulty scheduler curriculum term.
        end_frac: Difficulty fraction at which the ramp reaches the final value. Defaults to 1.

    Returns:
        The interpolated value.
    """
    frac = _difficulty_frac(env, difficulty_term_str)
    alpha = min(max((frac - start_frac) / max(end_frac - start_frac, 1e-6), 0.0), 1.0)
    return _interpolate(initial_value, final_value, alpha)


class reward_weight_ramp(ManagerTermBase):
    """Ramp reward weights linearly from zero to their final values over a number of steps spent at a high difficulty.

    The ramp advances only on the steps at which the mean difficulty fraction of a
    :class:`~isaaclab.envs.mdp.DifficultyScheduler` term is at least :paramref:`start_frac`: it waits for the curriculum
    to get there, and pauses while the failures the growing penalties may cause pull the difficulty back under it.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._steps = 0
        """Steps ramped so far."""
        self._last_step = env.common_step_counter

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        final_weights: dict[str, float],
        num_steps: int,
        start_frac: float,
        difficulty_term_str: str,
    ) -> float:
        """Set the weights for the current step.

        Args:
            env: The environment.
            env_ids: Unused; the weights are shared by all environments.
            final_weights: Final weight of each reward term, by term name.
            num_steps: Environment steps at or above :paramref:`start_frac` the ramp takes.
            start_frac: Mean difficulty fraction from which the ramp advances.
            difficulty_term_str: Name of the difficulty scheduler curriculum term.

        Returns:
            The share of the final weights reached, in [0, 1].
        """
        # the manager computes the curriculum on every step with a reset, so steps are counted since the last call
        if _difficulty_frac(env, difficulty_term_str) >= start_frac:
            self._steps += env.common_step_counter - self._last_step
        self._last_step = env.common_step_counter
        alpha = min(self._steps / max(num_steps, 1), 1.0)
        for name, weight in final_weights.items():
            term_cfg = env.reward_manager.get_term_cfg(name)
            term_cfg.weight = alpha * weight
            env.reward_manager.set_term_cfg(name, term_cfg)
        return alpha
