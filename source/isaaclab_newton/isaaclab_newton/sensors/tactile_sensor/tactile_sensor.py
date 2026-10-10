# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tactile sensor: halltaxel's contact-patch readout on the Newton backend.

The taxel patches are ordinary collision shapes baked into the robot asset, so this sensor needs no runtime
halltaxel. It is a shape-level contact sensor over those shapes (the same :class:`newton.sensors.SensorContact`
halltaxel's ``ContactPatchBackend`` uses) that additionally reorders the per-shape forces into taxel order and
rotates them into the sensor frame on the GPU.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from isaaclab.physics import PhysicsEvent
from isaaclab.sim import SimulationContext

from isaaclab_newton.physics import NewtonBuilderCfg, NewtonManager

from ..contact_sensor import ContactSensor
from .kernels import taxel_reaction_forces_kernel
from .tactile_sensor_data import TactileSensorData

if TYPE_CHECKING:
    from .tactile_sensor_cfg import TactileSensorCfg

logger = logging.getLogger(__name__)


class TactileSensor(ContactSensor):
    """Per-taxel reaction forces of one fingertip, in its sensor frame."""

    cfg: TactileSensorCfg

    def __init__(self, cfg: TactileSensorCfg):
        # the taxel colliders are the shapes named <finger>_taxel_NNN below the fingertip body
        cfg.sensor_shape_prim_expr = [f"{cfg.prim_path}/.*_taxel_[0-9]+/.*"]
        super().__init__(cfg)
        self._data: TactileSensorData = TactileSensorData()

    @property
    def data(self) -> TactileSensorData:
        self._update_outdated_buffers()
        return self._data

    @property
    def num_taxels(self) -> int:
        """Number of taxels, in the sidecar's order."""
        return len(self._taxel_names)

    @property
    def taxel_names(self) -> list[str]:
        """Taxel names (``<finger>_taxel_NNN``), in the sidecar's order."""
        return self._taxel_names

    def _register_callbacks(self):
        """Also set the collision groups of the taxels and the pad proxy before the model is built."""
        super()._register_callbacks()
        self._model_init_handle = NewtonManager.register_callback(
            self._set_collision_groups, PhysicsEvent.MODEL_INIT, name=f"tactile_collision_groups_{self.cfg.prim_path}"
        )

    def _clear_callbacks(self):
        super()._clear_callbacks()
        if getattr(self, "_model_init_handle", None) is not None:
            self._model_init_handle.deregister()
            self._model_init_handle = None

    def _set_collision_groups(self, _event):
        """Set the taxel and pad-proxy collision groups in the shared model builder (see :class:`TactileSensorCfg`)."""
        sim = SimulationContext.instance()
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
        # shapes are matched like newton.sensors.SensorContact matches them: full regex over the labels
        taxels = re.compile("|".join(self.cfg.sensor_shape_prim_expr))
        pads = re.compile(f"{self.cfg.prim_path}/.*_pad/.*")
        num_taxels = num_pads = 0
        for shape, label in enumerate(builder.shape_label):
            if taxels.fullmatch(label) and self.cfg.taxel_collision_group is not None:
                builder.shape_collision_group[shape] = self.cfg.taxel_collision_group
                num_taxels += 1
            elif pads.fullmatch(label):
                if self.cfg.pad_collision_group is None or self.cfg.pad_collision_group <= 1:
                    raise ValueError(
                        f"Tactile sensor '{self.cfg.prim_path}': the asset has a pad proxy, so 'pad_collision_group'"
                        " must be a positive group other than the default 1, distinct for every finger. Left in"
                        " the default group, the proxy would collide with the object as well."
                    )
                builder.shape_collision_group[shape] = self.cfg.pad_collision_group
                num_pads += 1
        logger.info(
            f"Tactile sensor '{self.cfg.prim_path}': {num_taxels} taxel shapes in collision group"
            f" {self.cfg.taxel_collision_group}, {num_pads} pad-proxy shapes in group {self.cfg.pad_collision_group}"
        )

    def _initialize_impl(self):
        """Register the shape contact sensor, then map its rows to the sidecar's taxel order."""
        super()._initialize_impl()
        with np.load(self.cfg.sidecar_path) as sidecar:
            self._taxel_names = [str(name) for name in sidecar["patch_names"]]
            taxel_positions = sidecar["taxel_positions"]
            sensor_quat_b = wp.quatf(*sidecar["sensor_xform"][3:7])

        # a taxel is a prim named <finger>_taxel_NNN holding one collision shape, so a row's taxel name is the
        # parent of its shape (in the first environment; all environments share the layout)
        model = NewtonManager.backend.model
        shape_indices = self.contact_view.sensing_indices[: self._num_sensors]
        row_names = [model.shape_label[i].rsplit("/", 2)[-2] for i in shape_indices]
        missing = [name for name in self._taxel_names if name not in row_names]
        if missing or len(row_names) != len(self._taxel_names):
            raise RuntimeError(
                f"Tactile sensor '{self.cfg.prim_path}' finds {len(row_names)} taxel shapes, but the sidecar"
                f" '{self.cfg.sidecar_path}' lists {len(self._taxel_names)} (missing: {missing[:3]}); the asset and"
                " the sidecar are from different builds."
            )
        rows = [row_names.index(name) for name in self._taxel_names]
        # per taxel: the rotation from its collider's frame to the sensor frame, both fixed on the fingertip body
        shape_quat_b = model.shape_transform.numpy()[[shape_indices[row] for row in rows], 3:7]
        shape_to_sensor = [wp.quat_inverse(wp.quatf(*q)) * sensor_quat_b for q in shape_quat_b]

        self._taxel_rows = wp.array(rows, dtype=wp.int32, device=self._device)
        self._shape_to_sensor = wp.array(shape_to_sensor, dtype=wp.quatf, device=self._device)
        self._data._forces_s = wp.zeros((self._num_envs, self.num_taxels), dtype=wp.vec3f, device=self._device)
        self._data._taxel_pos_b = wp.array(taxel_positions, dtype=wp.vec3f, device=self._device)

    def _update_buffers_impl(self, env_mask: wp.array):
        """Copy the contact forces, then turn the taxel rows into sensor-frame reaction forces."""
        super()._update_buffers_impl(env_mask)
        wp.launch(
            taxel_reaction_forces_kernel,
            dim=(self._num_envs, self.num_taxels),
            inputs=[
                env_mask,
                self._num_sensors,
                self._num_filter_objects,  # the other fingers' pad proxies, if filter_shape_prim_expr names them
                self.cfg.finger_contact_scale,
                self._taxel_rows,
                self._shape_to_sensor,
                self._data._net_forces_w,
                self._data._force_matrix_w,
                self.contact_view.sensing_transforms,
            ],
            outputs=[self._data._forces_s],
            device=self._device,
        )
