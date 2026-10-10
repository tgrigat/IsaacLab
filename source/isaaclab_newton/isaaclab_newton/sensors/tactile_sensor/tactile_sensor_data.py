# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp

from isaaclab.utils.warp import ProxyArray

from ..contact_sensor import ContactSensorData


class TactileSensorData(ContactSensorData):
    """Data container for :class:`TactileSensor`.

    Adds the taxel readings to the contact sensor data. The inherited per-sensor quantities follow the contact
    sensor's shape order; the taxel quantities here follow the sidecar's taxel order.
    """

    _forces_s: wp.array | None = None
    _forces_s_ta: ProxyArray | None = None
    _taxel_pos_b: wp.array | None = None
    _taxel_pos_b_ta: ProxyArray | None = None

    @property
    def forces_s(self) -> ProxyArray:
        """Per-taxel reaction forces [N] in the sensor frame.

        `wp.vec3f` array of shape (N, T), with N environments and T taxels; (N, T, 3) as a `torch.Tensor`.
        Same convention as halltaxel's ``TactileReading.forces``: the force each taxel exerts on what touches it
        (the negated contact force), expressed in the sidecar's ``sensor_xform`` frame.
        """
        if self._forces_s_ta is None:
            self._forces_s_ta = ProxyArray(self._forces_s)
        return self._forces_s_ta

    @property
    def taxel_pos_b(self) -> ProxyArray:
        """Taxel positions [m] in the fingertip body frame, shape (T,); (T, 3) as a `torch.Tensor`."""
        if self._taxel_pos_b_ta is None:
            self._taxel_pos_b_ta = ProxyArray(self._taxel_pos_b)
        return self._taxel_pos_b_ta
