# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum modifiers of the Franka + ORCA-hand lift environments, for :class:`~isaaclab.envs.mdp.modify_term_cfg`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ... import mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _difficulty_frac(env: ManagerBasedRLEnv, difficulty_term_str: str) -> float:
    """Mean difficulty of a :class:`~isaaclab.envs.mdp.DifficultyScheduler` term, normalized to [0, 1]."""
    return getattr(env.curriculum_manager.cfg, difficulty_term_str).func.difficulty_frac


def stage_mode_weights(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: Sequence[float] | None,
    stage_fracs: Sequence[tuple[float, float]],
    difficulty_term_str: str,
) -> list[float] | Any:
    """Weights of the reset modes, uniform over the modes whose difficulty window holds the current difficulty.

    Intended for the ``mode_weights`` of :class:`~isaaclab_tasks.core.lift.mdp.events.conditional_reset`. A mode is
    drawn while the mean difficulty fraction is in its window ``[start, end)`` (an end of 1 includes the maximum), so
    a mode can stay in the mix after the next one unlocks, or drop out once a later one replaces it. The difficulty
    falls with failures, so a mode can also come back.

    Args:
        env: The environment.
        env_ids: Unused; the weights are shared by all environments.
        data: Current mode weights.
        stage_fracs: Difficulty window ``(start, end)`` of each mode, in the order of the modes. The windows must
            cover [0, 1] between them, so some mode is always drawn.
        difficulty_term_str: Name of the difficulty scheduler curriculum term.

    Returns:
        The new weights, or :attr:`~isaaclab.envs.mdp.modify_env_param.NO_CHANGE` if they are unchanged.
    """
    frac = _difficulty_frac(env, difficulty_term_str)
    weights = [float(start <= frac and (frac < end or end >= 1.0)) for start, end in stage_fracs]
    if data is not None and list(data) == weights:
        return mdp.modify_env_param.NO_CHANGE
    return weights


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
