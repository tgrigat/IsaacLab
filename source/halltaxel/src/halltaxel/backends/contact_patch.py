"""Contact-patch tactile backend.

Each taxel patch of a segmented sensing surface becomes its own collision
mesh on the target body. ``newton.sensors.SensorContact`` reports the
total contact force per collision shape, which yields one force per taxel.

Notes
-----
``SensorContact`` must be constructed before the ``newton.Contacts``
buffer is allocated, and ``solver.update_contacts(contacts)`` must run
before each :meth:`ContactPatchBackend.update`.

When the sensing body is kinematic, construct the
``newton.CollisionPipeline`` with ``include_static_kinematic_pairs=False``.
Adjacent patches on a kinematic body otherwise generate collision work
for every touching pair, which can dominate the cost of ``collide()``.
Contacts between dynamic bodies and the patches are unaffected by this
flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import newton
import numpy as np
import trimesh
import warp as wp
from newton.sensors import SensorContact

from halltaxel._collision import inherit_collision_filters
from halltaxel.backends.base import TactileBackend, TactileHandle
from halltaxel.layout import TaxelLayout
from halltaxel.reading import TactileReading
from halltaxel.segmentation import SurfaceSegmentation, segment_surface
from halltaxel.spec import TactileSurfaceSpec

__all__ = [
    "ContactPatchBackend",
    "ContactPatchHandle",
    "ContactPatchShapes",
    "attach_contact_patches",
]


@dataclass(frozen=True)
class ContactPatchShapes:
    """Collision shapes added by :func:`attach_contact_patches`.

    Both lists are in patch-id order, so entry ``i`` belongs to patch
    ``i`` of the segmentation.

    Attributes
    ----------
    shape_indices : list of int
        Index of each patch's shape in the builder it was added to.
    labels : list of str
        Label of each patch's shape, ``f"{label_prefix}_{i:03d}"``.
    """

    shape_indices: list[int]
    labels: list[str]


def attach_contact_patches(
    builder: newton.ModelBuilder,
    body: int,
    mesh: trimesh.Trimesh,
    segmentation: SurfaceSegmentation,
    *,
    label_prefix: str,
    xform: Any | None = None,
    cfg: newton.ModelBuilder.ShapeConfig | None = None,
) -> ContactPatchShapes:
    """Add one collision mesh per taxel patch to a body.

    Each patch of ``segmentation`` is added with
    ``builder.add_shape_mesh()``, using the same ``xform`` and ``cfg``
    for every patch. The new shapes inherit the collision filters of the
    shapes already on ``body``. Must be called before
    ``builder.finalize()``.

    Parameters
    ----------
    builder : newton.ModelBuilder
        Builder to modify.
    body : int
        Index of the body the patches are attached to. Must be a body,
        not static world geometry: ``SensorContact`` does not accept
        global shapes.
    mesh : trimesh.Trimesh
        The sensing surface.
    segmentation : SurfaceSegmentation
        Partition of ``mesh``'s faces into patches.
    label_prefix : str
        Prefix of the shape labels. Patch ``i`` is labeled
        ``f"{label_prefix}_{i:03d}"``.
    xform : wp.transform, optional
        Transform of the patches in ``body``'s frame. Defaults to
        identity.
    cfg : newton.ModelBuilder.ShapeConfig, optional
        Shape configuration for every patch. Defaults to
        ``builder.default_shape_cfg``.

    Returns
    -------
    ContactPatchShapes
        Shape index and label of each patch, in patch-id order.

    Raises
    ------
    ValueError
        If ``segmentation.labels`` does not have one entry per face of
        ``mesh``, or if a patch has no faces.

    Notes
    -----
    The patches replace a whole-mesh collision shape for the same
    surface. If the body also carries the full mesh, disable its
    collisions and set its density to zero to avoid duplicate contacts
    and mass.
    """
    segmentation.check_mesh(mesh)

    shape_indices: list[int] = []
    labels: list[str] = []
    for patch_id in range(segmentation.num_regions):
        face_idx = np.flatnonzero(segmentation.labels == patch_id)
        if len(face_idx) == 0:
            raise ValueError(f"patch {patch_id} has no faces - malformed segmentation")

        patch_mesh = mesh.submesh([face_idx], append=True)
        newton_mesh = newton.Mesh(patch_mesh.vertices, patch_mesh.faces.flatten())
        label = f"{label_prefix}_{patch_id:03d}"
        shape_idx = builder.add_shape_mesh(
            body, xform=xform, mesh=newton_mesh, cfg=cfg, label=label
        )

        shape_indices.append(shape_idx)
        labels.append(label)

    inherit_collision_filters(builder, body, shape_indices)
    return ContactPatchShapes(shape_indices=shape_indices, labels=labels)


@dataclass(frozen=True)
class ContactPatchHandle(TactileHandle):
    """Handle for one surface attached by :class:`ContactPatchBackend`.

    Attributes
    ----------
    patches : ContactPatchShapes
        Shape indices and labels of the surface's patches. The shape
        indices refer to the builder the patches were added to; use the
        labels to identify the shapes in a model replicated across
        worlds.
    """

    patches: ContactPatchShapes


class ContactPatchBackend(TactileBackend):
    """Tactile backend that senses force per taxel-patch collision shape.

    :meth:`attach` segments the surface, adds one collision mesh per
    patch, and records the patch labels. :meth:`finalize` builds one
    ``SensorContact`` over all recorded labels. :meth:`update` returns
    the force on every patch of every surface in one reading, keyed by
    label prefix.

    Parameters
    ----------
    cfg : newton.ModelBuilder.ShapeConfig, optional
        Shape configuration used by :meth:`attach` calls that pass no
        ``cfg``. Defaults to ``builder.default_shape_cfg``.
    collide_between_surfaces : bool, default True
        If False, each :meth:`attach` filters collisions between its
        patches and all previously attached patches, so separate
        surfaces never collide with each other. All :meth:`attach` calls
        must then use the same builder.
    relax_iters : int, default 0
        ``relax_iters`` argument of :func:`segment_surface`, used for
        every surface.
    smooth_iters : int, default 1
        ``smooth_iters`` argument of :func:`segment_surface`, used for
        every surface.
    """

    def __init__(
        self,
        *,
        cfg: newton.ModelBuilder.ShapeConfig | None = None,
        collide_between_surfaces: bool = True,
        relax_iters: int = 0,
        smooth_iters: int = 1,
    ):
        self._default_cfg = cfg
        self._collide_between_surfaces = collide_between_surfaces
        self._relax_iters = relax_iters
        self._smooth_iters = smooth_iters
        self._surfaces: dict[str, slice] = {}
        self._labels: list[str] = []
        self._shape_indices: list[int] = []
        self._sensor_rotations: list[np.ndarray] = []
        self._sensor: SensorContact | None = None

    def attach(
        self,
        builder: newton.ModelBuilder,
        body: int,
        spec: TactileSurfaceSpec,
        *,
        xform: Any | None = None,
        cfg: newton.ModelBuilder.ShapeConfig | None = None,
    ) -> ContactPatchHandle:
        """Segment a surface and add one collision mesh per patch to a body.

        The surface's ``sensing_mesh`` is segmented around
        ``spec.taxel_positions`` with :func:`segment_surface`, and the
        patches are added with :func:`attach_contact_patches`.

        Parameters
        ----------
        builder : newton.ModelBuilder
            Builder to modify, before ``finalize()``.
        body : int
            Index of the body the surface is attached to.
        spec : TactileSurfaceSpec
            The surface to attach.
        xform : wp.transform, optional
            Transform of the surface in ``body``'s frame. Defaults to
            identity.
        cfg : newton.ModelBuilder.ShapeConfig, optional
            Shape configuration of the patches. Defaults to the
            backend's ``cfg``.

        Returns
        -------
        ContactPatchHandle
            Handle with the surface's layout and patch shapes.

        Raises
        ------
        RuntimeError
            If :meth:`finalize` has already been called.
        ValueError
            If ``spec.label_prefix`` is already used by a surface
            attached to this backend.
        """
        if self._sensor is not None:
            raise RuntimeError(
                "attach() called after finalize() - attach every surface before finalize()"
            )
        if spec.label_prefix in self._surfaces:
            raise ValueError(
                f"label_prefix {spec.label_prefix!r} is already attached to this backend - "
                "every surface needs a distinct prefix (see attach_contact_patches())"
            )

        mesh = spec.sensing_mesh
        segmentation = segment_surface(
            mesh,
            seeds=spec.taxel_positions,
            relax_iters=self._relax_iters,
            smooth_iters=self._smooth_iters,
        )
        patches = attach_contact_patches(
            builder,
            body,
            mesh,
            segmentation,
            label_prefix=spec.label_prefix,
            xform=xform,
            cfg=cfg if cfg is not None else self._default_cfg,
        )
        mount = spec.sensor_xform if spec.sensor_xform is not None else wp.transform_identity()
        # Each taxel sits at its patch's seed: its position projected
        # onto the surface, and moved by relaxation if relax_iters > 0.
        layout = TaxelLayout(positions=segmentation.seeds.copy(), sensor_xform=mount)
        if xform is not None:
            layout = _transform_layout(layout, xform)

        if not self._collide_between_surfaces:
            for new_shape in patches.shape_indices:
                for old_shape in self._shape_indices:
                    builder.add_shape_collision_filter_pair(old_shape, new_shape)

        start = len(self._labels)
        taxels = slice(start, start + len(patches.labels))
        self._surfaces[spec.label_prefix] = taxels
        self._labels.extend(patches.labels)
        self._shape_indices.extend(patches.shape_indices)
        # Each patch shape's world pose already includes ``xform``, so the
        # sensor frame is the mount alone relative to the shape.
        self._sensor_rotations.append(np.tile(np.array(mount.q), (len(patches.labels), 1)))
        return ContactPatchHandle(
            label_prefix=spec.label_prefix, layout=layout, taxels=taxels, patches=patches
        )

    def finalize(self, model: newton.Model) -> None:
        """Build the ``SensorContact`` over all attached patches.

        Parameters
        ----------
        model : newton.Model
            The finalized model.

        Raises
        ------
        RuntimeError
            If called more than once, or if no surface has been attached.
        """
        if self._sensor is not None:
            raise RuntimeError("finalize() already called on this backend")
        if not self._labels:
            raise RuntimeError(
                "no surfaces attached - call attach() for every surface before finalize()"
            )

        self._sensor = SensorContact(model, sensing_shapes=self._labels)
        self._surfaces = MappingProxyType(self._surfaces)

    def update(self, model: newton.Model, state: Any, contacts: Any) -> TactileReading:
        """Read the force on every patch for the current step.

        Parameters
        ----------
        model : newton.Model
            The finalized model.
        state : newton.State
            Current simulation state.
        contacts : newton.Contacts
            Contacts for this step, after ``solver.update_contacts()``.

        Returns
        -------
        TactileReading
            Reaction force of every patch in its surface's sensor frame,
            shape ``(model.world_count, num_patches, 3)``, with each
            surface's slice in ``surfaces``.

        Raises
        ------
        RuntimeError
            If :meth:`finalize` has not been called.
        """
        if self._sensor is None:
            raise RuntimeError("finalize() must run before update() - build a TactileRuntime first")

        self._sensor.update(state, contacts)
        shape = (model.world_count, len(self._labels))
        forces = self._sensor.total_force.numpy().reshape(*shape, 3)
        # sensing_transforms is each patch shape's world pose (state.body_q
        # composed with the shape's own xform on its body), computed by
        # _sensor.update() above. Composed with the surface's sensor mount it
        # is the sensor frame's world orientation; rotating by its inverse
        # takes the world-frame ``forces`` into the sensor frame. total_force
        # is the force on the patch, so it is negated to give the reaction.
        shape_rotations = self._sensor.sensing_transforms.numpy()[:, 3:7].reshape(*shape, 4)
        rotations = _quat_multiply(shape_rotations, np.concatenate(self._sensor_rotations))
        forces = _rotate(_quat_conjugate(rotations), -forces)
        return TactileReading(forces=forces, surfaces=self._surfaces)


def _transform_layout(layout: TaxelLayout, xform: Any) -> TaxelLayout:
    """Return ``layout`` moved from its own frame into the frame ``xform`` maps to."""
    xform = wp.transform(*xform)
    rotation = np.array(wp.quat_to_matrix(xform.q)).reshape(3, 3)
    return TaxelLayout(
        positions=layout.positions @ rotation.T + np.array(xform.p),
        sensor_xform=xform * layout.sensor_xform,
    )


def _rotate(quats: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate ``vectors`` (..., 3) by unit quaternions ``quats`` (..., 4) in (x, y, z, w) order."""
    xyz, w = quats[..., :3], quats[..., 3:]
    t = 2.0 * np.cross(xyz, vectors)
    return vectors + w * t + np.cross(xyz, t)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the products ``a * b`` of quaternions (..., 4) in (x, y, z, w) order."""
    a_xyz, a_w = a[..., :3], a[..., 3:]
    b_xyz, b_w = b[..., :3], b[..., 3:]
    xyz = a_w * b_xyz + b_w * a_xyz + np.cross(a_xyz, b_xyz)
    w = a_w * b_w - np.sum(a_xyz * b_xyz, axis=-1, keepdims=True)
    return np.concatenate([xyz, w], axis=-1)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Return the conjugates of quaternions (..., 4) in (x, y, z, w) order."""
    return np.concatenate([-q[..., :3], q[..., 3:]], axis=-1)
