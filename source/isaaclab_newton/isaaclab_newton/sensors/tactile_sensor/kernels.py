# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for :class:`~isaaclab_newton.sensors.TactileSensor`."""

import warp as wp


@wp.kernel
def taxel_reaction_forces_kernel(
    # in
    env_mask: wp.array(dtype=wp.bool),
    num_sensors: int,
    num_finger_pads: int,
    finger_contact_scale: float,
    taxel_rows: wp.array(dtype=wp.int32),  # (num_taxels,) contact-sensor row of each taxel
    shape_to_sensor: wp.array(dtype=wp.quatf),  # (num_taxels,) taxel shape frame -> sensor frame
    net_forces_w: wp.array2d(dtype=wp.vec3f),  # (num_envs, num_sensors) from everything
    finger_forces_w: wp.array3d(dtype=wp.vec3f),  # (num_envs, num_sensors, num_finger_pads) or None
    shape_transforms_w: wp.array(dtype=wp.transformf),  # (num_envs * num_sensors,)
    # out
    forces_s: wp.array2d(dtype=wp.vec3f),  # (num_envs, num_taxels)
):
    """Per (env, taxel): the taxel's reaction force in the sensor frame, ``-(q_shape_w * q_shape_to_sensor)^-1 f_w``.

    ``f_w`` is the taxel's net force, with the part from other fingers' pad proxies scaled by
    ``finger_contact_scale``. Launch with ``dim=(num_envs, num_taxels)``.
    """
    env, taxel = wp.tid()
    if env_mask:
        if not env_mask[env]:
            return
    row = taxel_rows[taxel]
    force_w = net_forces_w[env, row]
    for pad in range(num_finger_pads):
        force_w += (finger_contact_scale - 1.0) * finger_forces_w[env, row, pad]
    sensor_quat_w = wp.transform_get_rotation(shape_transforms_w[env * num_sensors + row]) * shape_to_sensor[taxel]
    force_s = -wp.quat_rotate_inv(sensor_quat_w, force_w)
    # a diverged environment keeps non-finite shape poses until its reset state has been simulated; report no force
    # rather than letting the NaNs into the observation history
    if not wp.isfinite(force_s):
        force_s = wp.vec3f(0.0)
    forces_s[env, taxel] = force_s
