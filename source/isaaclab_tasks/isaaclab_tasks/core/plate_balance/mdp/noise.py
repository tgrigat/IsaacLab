# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation noise of the plate-balancing environments."""

from __future__ import annotations

from collections.abc import Callable

import torch

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg


def cop_noise(data: torch.Tensor, cfg: CoPNoiseCfg) -> torch.Tensor:
    """Corrupt a centre-of-pressure observation, ``(f_n, x, y, z)`` per sensor (see
    :func:`~.observations.pad_contact_cop_b`): independently for each sensor in contact, with probability
    ``cfg.prob``, scale its normal force by ``U(force_scale)`` and shift its contact point by ``U(±pos_range)`` [m] on
    each axis. Sensors without a contact stay zero."""
    cop = data.view(data.shape[0], -1, 4)
    hit = (torch.rand(cop.shape[:2], device=data.device) < cfg.prob) & (cop[..., 0] > 0.0)
    low, high = cfg.force_scale
    scale = torch.empty(cop.shape[:2], device=data.device).uniform_(low, high)
    shift = torch.empty(*cop.shape[:2], 3, device=data.device).uniform_(-cfg.pos_range, cfg.pos_range)
    force = torch.where(hit, cop[..., 0] * scale, cop[..., 0])
    point = torch.where(hit.unsqueeze(-1), cop[..., 1:] + shift, cop[..., 1:])
    return torch.cat([force.unsqueeze(-1), point], dim=-1).view_as(data)


@configclass
class CoPNoiseCfg(NoiseCfg):
    """Configuration for :func:`cop_noise`; the defaults are those of Pan et al. (2026), "Beyond Binary"."""

    func: Callable[[torch.Tensor, CoPNoiseCfg], torch.Tensor] = cop_noise

    prob: float = 0.2
    """Probability, per sensor and step, of corrupting its reading."""

    force_scale: tuple[float, float] = (0.9, 1.1)
    """Range of the normal force's scale."""

    pos_range: float = 1.0e-3
    """Largest shift [m] of the contact point on each axis."""
