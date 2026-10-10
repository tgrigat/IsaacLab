# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from ..contact_sensor import ContactSensorCfg

if TYPE_CHECKING:
    from .tactile_sensor import TactileSensor


@configclass
class TactileSensorCfg(ContactSensorCfg):
    """Configuration for a :class:`TactileSensor` over one fingertip's taxel colliders.

    :attr:`prim_path` is the fingertip body. Below it, written by halltaxel at asset build time (see
    ``scripts/tools/franka_orca/build_asset.py``), each taxel is a prim ``<finger>_taxel_NNN`` holding one collision
    shape, and ``<finger>_pad`` is a pad proxy: the whole sensed pad as one shape, in the same place as the taxels.

    Collisions, set before the model is built:

    * taxels collide with everything except other taxels (:attr:`taxel_collision_group`),
    * a pad proxy collides only with the taxels and the pad proxies in its own group (:attr:`pad_collision_group`).

    Two ways to use the pad proxies:

    * Every finger in its own group: two pads touching give one contact pair per taxel instead of one per pair of
      taxels, and the taxels sense the other finger. Such a touch is resolved twice, finger A's taxels against B's
      pad and B's taxels against A's pad, so each side carries about half the force; :attr:`finger_contact_scale`
      corrects for that. For it to know which forces come from other fingers, set :attr:`filter_shape_prim_expr`
      to the pad proxies, e.g. ``["{ENV_REGEX_NS}/Robot/.*_pad/.*"]``.
    * All fingers in one group, with the asset filtering each pad proxy against the other fingers' taxels: two
      fingertips touch with a single pad-to-pad pair, and the taxels sense only the object, the table and the arm.

    :attr:`sensor_shape_prim_expr` is set by the sensor (the taxels below :attr:`prim_path`).
    """

    class_type: type[TactileSensor] | str = "{DIR}.tactile_sensor:TactileSensor"

    sidecar_path: str = MISSING
    """Path of the finger's ``<finger>.npz`` sidecar.

    It holds ``patch_names`` (taxel order), ``taxel_positions`` (N, 3) and ``sensor_xform`` (7,)
    ``[x, y, z, qx, qy, qz, qw]``, both in the fingertip body frame.
    """

    taxel_collision_group: int | None = -1
    """Newton collision group of the taxel shapes. Defaults to -1.

    A negative group collides with every group except itself, so the taxels of all tactile sensors sharing it skip
    each other. None keeps the groups from the asset.
    """

    pad_collision_group: int | None = None
    """Newton collision group of this finger's pad proxy: a positive group other than the default 1, either distinct
    for every finger (e.g. 2 to 6 for five fingers) or shared by all (see the class docs). Defaults to None, for
    assets without pad proxies.

    A positive group collides only with itself and with negative groups, so the pad proxy touches only the taxels
    and the pad proxies sharing its group.
    """

    finger_contact_scale: float = 2.0
    """Factor on the part of a taxel's force that comes from the counterparts in :attr:`filter_shape_prim_expr`
    (other fingers' pad proxies). Defaults to 2, since each side of a pad-to-pad touch carries about half of it."""
