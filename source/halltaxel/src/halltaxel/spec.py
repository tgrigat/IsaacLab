"""Declarative description of a tactile sensing surface.

:class:`TactileSurfaceSpec` holds the sensing mesh, an optional sensed
sub-region of it, the taxel positions, a label prefix, and an optional
sensor frame the taxel forces are reported in. It can be
built directly, from files (:meth:`TactileSurfaceSpec.from_files`), or
from a marker link in a URDF (:meth:`TactileSurfaceSpec.from_urdf`). The
module also provides builder helpers for URDF-based setups.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import newton
import numpy as np
import trimesh
import warp as wp

from halltaxel._collision import inherit_collision_filters
from halltaxel.segmentation import load_surface_mesh, select_region

__all__ = ["TactileSurfaceSpec", "body_by_link", "replace_body_mesh"]


@dataclass(frozen=True)
class TactileSurfaceSpec:
    """Description of one tactile sensing surface.

    Attributes
    ----------
    mesh : trimesh.Trimesh
        Surface mesh. If ``sensor_region`` is given, only the part of
        ``mesh`` under it is sensed.
    taxel_positions : ndarray, shape (N, 3)
        Taxel positions, in the frame of ``mesh``.
    label_prefix : str
        Prefix of the surface's shape labels. Must be unique among the
        surfaces attached to one model.
    sensor_region : trimesh.Trimesh or None, default None
        Mesh covering the sensed part of ``mesh``'s surface, in the same
        frame (see :func:`select_region`). If None, all of ``mesh`` is
        sensed.
    sensor_xform : wp.transform or None, default None
        Pose of the sensor frame in the frame of ``mesh``. Taxel forces
        are reported in this frame, as a real sensor reports them in its
        own. If None, forces are reported in the frame of ``mesh``.
    """

    mesh: trimesh.Trimesh
    taxel_positions: np.ndarray
    label_prefix: str
    sensor_region: trimesh.Trimesh | None = None
    sensor_xform: wp.transform | None = None

    @cached_property
    def sensor_mask(self) -> np.ndarray | None:
        """ndarray of bool or None: Faces of ``mesh`` under ``sensor_region``.

        Shape ``(len(mesh.faces),)``, computed with :func:`select_region`
        on first access and cached. None if ``sensor_region`` is None.
        """
        if self.sensor_region is None:
            return None
        return select_region(self.mesh, self.sensor_region)

    @cached_property
    def sensing_mesh(self) -> trimesh.Trimesh:
        """trimesh.Trimesh: The sensed part of ``mesh``.

        ``mesh`` restricted to the faces in ``sensor_mask``, or ``mesh``
        itself if ``sensor_region`` is None. Cached.
        """
        mask = self.sensor_mask
        if mask is None:
            return self.mesh
        return self.mesh.submesh([np.flatnonzero(mask)], append=True)

    @cached_property
    def unsensed_mesh(self) -> trimesh.Trimesh | None:
        """trimesh.Trimesh or None: The part of ``mesh`` outside ``sensor_region``.

        None if ``sensor_region`` is None or covers every face. Cached.
        """
        mask = self.sensor_mask
        if mask is None or mask.all():
            return None
        return self.mesh.submesh([np.flatnonzero(~mask)], append=True)

    @classmethod
    def from_files(
        cls,
        mesh_path: str | Path,
        taxel_positions_path: str | Path,
        *,
        label_prefix: str,
        sensor_region_path: str | Path | None = None,
        scale: float = 1.0,
        sensor_xform: wp.transform | None = None,
    ) -> TactileSurfaceSpec:
        """Create a spec from a mesh file and a taxel-position file.

        Parameters
        ----------
        mesh_path : str or Path
            Mesh file of the surface, in any format trimesh can load.
        taxel_positions_path : str or Path
            ``.npy`` file with an (N, 3) array of taxel positions, in the
            frame of the scaled mesh. Not scaled.
        label_prefix : str
            See :class:`TactileSurfaceSpec`.
        sensor_region_path : str or Path, optional
            Mesh file of the sensed region. See ``sensor_region``.
        scale : float, default 1.0
            Uniform scale applied to both meshes after loading.
        sensor_xform : wp.transform, optional
            Pose of the sensor frame in the frame of the scaled mesh. See
            :class:`TactileSurfaceSpec`.

        Returns
        -------
        TactileSurfaceSpec
        """
        mesh = load_surface_mesh(mesh_path, scale=scale)
        taxel_positions = np.load(taxel_positions_path)
        sensor_region = (
            load_surface_mesh(sensor_region_path, scale=scale)
            if sensor_region_path is not None
            else None
        )
        return cls(
            mesh=mesh,
            taxel_positions=taxel_positions,
            label_prefix=label_prefix,
            sensor_region=sensor_region,
            sensor_xform=sensor_xform,
        )

    @classmethod
    def from_urdf(
        cls,
        builder: newton.ModelBuilder,
        fixture_link: str,
        taxel_positions: np.ndarray,
        *,
        label_prefix: str,
        source_link: str | None = None,
        sensor_xform: wp.transform | None = None,
    ) -> TactileSurfaceSpec:
        """Create a spec from a sensor-region marker link of an imported URDF.

        ``fixture_link`` is a link attached by a fixed joint to the body
        that carries the sensor. Its mesh marks the sensed region of that
        body's mesh. The spec's ``mesh`` is the parent body's first mesh
        shape, and ``sensor_region`` and ``taxel_positions`` are
        transformed into the parent body's frame through the fixed
        joint. The fixture's shapes are made invisible and non-colliding
        in ``builder``.

        Parameters
        ----------
        builder : newton.ModelBuilder
            Builder on which ``add_urdf()`` has been called, before
            ``finalize()``.
        fixture_link : str
            URDF link name of the sensor-region marker.
        taxel_positions : ndarray, shape (N, 3)
            Taxel positions in ``fixture_link``'s frame.
        label_prefix : str
            See :class:`TactileSurfaceSpec`.
        source_link : str, optional
            URDF link name of the body carrying the sensor. If given, it
            must be the parent of ``fixture_link``'s fixed joint.
        sensor_xform : wp.transform, optional
            Pose of the sensor frame in the parent body's frame, not
            ``fixture_link``'s. See :class:`TactileSurfaceSpec`.

        Returns
        -------
        TactileSurfaceSpec
            Spec in the parent body's frame; attach it to that body with
            ``xform=None``.

        Raises
        ------
        ValueError
            If a link name does not match exactly one body, if
            ``fixture_link``'s body is not the child of exactly one fixed
            joint, if ``source_link`` is not that joint's parent, or if a
            body has no mesh shape.

        See Also
        --------
        replace_body_mesh : Remove the sensed part from the parent body's
            own mesh.
        """
        fixture_body = body_by_link(builder, fixture_link)
        parent_body, fixture_pose = _fixed_joint_pose(builder, fixture_body)
        if source_link is not None:
            expected_body = body_by_link(builder, source_link)
            if expected_body != parent_body:
                raise ValueError(
                    f"source_link {source_link!r} (body {expected_body}) doesn't match "
                    f"fixture_link {fixture_link!r}'s actual fixed-joint parent (body {parent_body})"
                )

        source_mesh = _body_mesh(builder, parent_body)
        fixture_mesh = _body_mesh(builder, fixture_body)
        for shape in _body_shapes(builder, fixture_body):
            builder.shape_flags[shape] &= ~_RENDER_AND_COLLIDE

        matrix = _transform_matrix(fixture_pose)
        sensor_region = fixture_mesh.copy()
        sensor_region.apply_transform(matrix)
        positions = trimesh.transformations.transform_points(np.asarray(taxel_positions), matrix)

        return cls(
            mesh=source_mesh,
            taxel_positions=positions,
            label_prefix=label_prefix,
            sensor_region=sensor_region,
            sensor_xform=sensor_xform,
        )


def body_by_link(builder: newton.ModelBuilder, link_name: str) -> int:
    """Return the index of the body created by ``add_urdf()`` for a URDF link.

    Bodies are matched by the label suffix ``f"/{link_name}"``, so the
    articulation name does not need to be known.

    Parameters
    ----------
    builder : newton.ModelBuilder
        Builder the URDF was imported into.
    link_name : str
        URDF link name.

    Returns
    -------
    int
        Body index.

    Raises
    ------
    ValueError
        If not exactly one body matches.
    """
    matches = [i for i, label in enumerate(builder.body_label) if label.endswith(f"/{link_name}")]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one body labeled .../{link_name!r}, found {len(matches)} - "
            "check the URDF was imported into this builder and the link name is exact"
        )
    return matches[0]


_RENDER_AND_COLLIDE = int(
    newton.ShapeFlags.VISIBLE
    | newton.ShapeFlags.COLLIDE_SHAPES
    | newton.ShapeFlags.COLLIDE_PARTICLES
)


def replace_body_mesh(
    builder: newton.ModelBuilder, body: int, spec: TactileSurfaceSpec
) -> int | None:
    """Replace a body's mesh shapes with the unsensed part of a spec's mesh.

    Makes every mesh shape currently on ``body`` invisible and
    non-colliding, then adds ``spec.unsensed_mesh`` as a zero-density
    shape with the color of the first replaced shape and the label
    ``f"{spec.label_prefix}_unsensed"``. The new shape inherits the
    collision filters of the body's existing shapes. Together with the
    tactile patches, it covers the body's surface once.

    Call before :func:`attach_tactile`, since every mesh shape on
    ``body`` is disabled.

    Parameters
    ----------
    builder : newton.ModelBuilder
        Builder to modify, before ``finalize()``.
    body : int
        Index of the body whose mesh ``spec.mesh`` was taken from.
    spec : TactileSurfaceSpec
        The surface to be attached to ``body``.

    Returns
    -------
    int or None
        Index of the added shape, or None if ``spec.unsensed_mesh`` is
        None.
    """
    mesh_shapes = [
        s for s in _body_shapes(builder, body) if builder.shape_type[s] == newton.GeoType.MESH
    ]
    for shape in mesh_shapes:
        builder.shape_flags[shape] &= ~_RENDER_AND_COLLIDE

    unsensed = spec.unsensed_mesh
    if unsensed is None:
        return None
    cfg = builder.default_shape_cfg.copy()
    cfg.density = 0.0  # the body's mass already comes from its own inertia
    shape = builder.add_shape_mesh(
        body,
        mesh=newton.Mesh(unsensed.vertices, unsensed.faces.flatten()),
        cfg=cfg,
        # Keep the replaced mesh's own color (e.g. its URDF material) rather
        # than newton's per-shape palette fallback.
        color=builder.shape_color[mesh_shapes[0]] if mesh_shapes else None,
        label=f"{spec.label_prefix}_unsensed",
    )
    inherit_collision_filters(builder, body, [shape])
    return shape


def _body_shapes(builder: newton.ModelBuilder, body: int) -> list[int]:
    """Return the indices of all shapes on ``body``."""
    return [s for s, b in enumerate(builder.shape_body) if b == body]


def _fixed_joint_pose(builder: newton.ModelBuilder, child_body: int) -> tuple[int, wp.transform]:
    """Return the parent body and parent-from-child transform of a body's fixed joint.

    Raises
    ------
    ValueError
        If ``child_body`` is not the child of exactly one joint, or that
        joint is not fixed.
    """
    joints = [j for j in range(builder.joint_count) if builder.joint_child[j] == child_body]
    if len(joints) != 1:
        raise ValueError(
            f"expected exactly one joint with body {child_body} as its child, found {len(joints)}"
        )
    (joint,) = joints
    if builder.joint_type[joint] != newton.JointType.FIXED:
        raise ValueError(
            f"body {child_body}'s joint (type {builder.joint_type[joint]!r}) isn't FIXED - "
            "a tactile-region fixture must be a fixed-joint child, since its pose relative "
            "to its parent has to be constant, independent of the rest of the articulation"
        )
    x_p = wp.transform(*builder.joint_X_p[joint])
    x_c = wp.transform(*builder.joint_X_c[joint])
    return builder.joint_parent[joint], x_p * wp.transform_inverse(x_c)


def _body_mesh(builder: newton.ModelBuilder, body: int) -> trimesh.Trimesh:
    """Return the first mesh shape of ``body`` as a trimesh in the body's frame.

    Raises
    ------
    ValueError
        If ``body`` has no mesh shape.
    """
    mesh_shapes = [
        s for s in _body_shapes(builder, body) if builder.shape_type[s] == newton.GeoType.MESH
    ]
    if not mesh_shapes:
        raise ValueError(f"body {body} has no MESH-type shape to use as a sensing surface")
    shape = mesh_shapes[0]
    source = builder.shape_source[shape]
    mesh = trimesh.Trimesh(
        vertices=np.array(source.vertices), faces=np.array(source.indices).reshape(-1, 3)
    )
    mesh.apply_transform(_transform_matrix(wp.transform(*builder.shape_transform[shape])))
    return mesh


def _transform_matrix(t: wp.transform) -> np.ndarray:
    """Return the 4x4 homogeneous matrix of a ``wp.transform``."""
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(wp.quat_to_matrix(t.q)).reshape(3, 3)
    matrix[:3, 3] = np.array(t.p)
    return matrix
