"""Scene helpers used by the halltaxel examples.

Provides test objects (a ball, a drop cube), posing and camera helpers,
and a per-patch force heatmap. Not required to use the library.
"""

from __future__ import annotations

from collections.abc import Sequence

import newton
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from halltaxel.segmentation import project_to_surface
from halltaxel.spec import TactileSurfaceSpec
from halltaxel.viz import force_heat_colors, patch_colors

__all__ = [
    "CUBE_DENSITY",
    "CUBE_HALF_EXTENT",
    "CUBE_WEIGHT",
    "SKIN_COLOR",
    "PatchHeatmap",
    "add_ball",
    "add_drop_cube",
    "frame_camera",
    "press_point",
    "to_world",
    "upright_xform",
]

CUBE_HALF_EXTENT = 0.004  # m, 8 mm cube
CUBE_DENSITY = 2000.0  # kg/m^3
CUBE_WEIGHT = (2 * CUBE_HALF_EXTENT) ** 3 * CUBE_DENSITY * 9.81  # N, ~0.01 N
DROP_GAP = 0.004  # m, cube's starting clearance above the surface
SKIN_COLOR = (0.82, 0.68, 0.6)

Vec3 = np.ndarray | Sequence[float]

_UP = np.array([0.0, 0.0, 1.0])


def press_point(spec: TactileSurfaceSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return a point at the center of a surface's taxels, on the sensing mesh.

    The centroid of ``spec.taxel_positions`` is projected onto the faces
    of ``spec.sensing_mesh`` that face outward, i.e. whose normal points
    the same way as the direction from the centroid of ``spec.mesh`` to
    the taxel centroid. This keeps the point off the inner wall of a
    hollow skin shell whose inner surface is also sensed: the taxel
    centroid can lie inside the shell, closer to that inner wall than to
    the pad.

    Parameters
    ----------
    spec : TactileSurfaceSpec
        The surface.

    Returns
    -------
    point : ndarray, shape (3,)
        Projected centroid, in the spec's frame.
    normal : ndarray, shape (3,)
        Unit normal of the mesh face containing ``point``.
    """
    centroid = spec.taxel_positions.mean(axis=0)
    mesh = spec.sensing_mesh
    outward = mesh.face_normals @ (centroid - spec.mesh.centroid) > 0
    if outward.any():
        mesh = mesh.submesh([np.flatnonzero(outward)], append=True)
    projected = project_to_surface(mesh, centroid[None, :])
    return projected.points[0], projected.normals[0]


def upright_xform(
    spec: TactileSurfaceSpec, point: Vec3, normal: Vec3, *, clearance: float = 0.01
) -> wp.transform:
    """Return a body pose that turns ``normal`` to +Z and places ``point`` above the origin.

    The pose is also offset along Z so that the lowest vertex of
    ``spec.mesh`` lies ``clearance`` above the plane Z = 0.

    Parameters
    ----------
    spec : TactileSurfaceSpec
        The surface whose mesh is posed.
    point : array_like, shape (3,)
        Point in the spec's frame to place on the world Z axis.
    normal : array_like, shape (3,)
        Unit direction in the spec's frame to rotate onto world +Z.
    clearance : float, default 0.01
        Height of the mesh's lowest vertex above Z = 0, in metres.

    Returns
    -------
    wp.transform
        Body-to-world transform.
    """
    q = wp.quat_between_vectors(wp.vec3(*normal), wp.vec3(*_UP))
    rotation = Rotation.from_quat([q[0], q[1], q[2], q[3]])
    lowest_z = rotation.apply(spec.mesh.vertices)[:, 2].min()
    rotated_point = rotation.apply(point)
    return wp.transform(p=wp.vec3(-rotated_point[0], -rotated_point[1], -lowest_z + clearance), q=q)


def to_world(xform: wp.transform, point: Vec3, normal: Vec3) -> tuple[np.ndarray, np.ndarray]:
    """Transform a point and a normal from a body frame to world frame.

    Parameters
    ----------
    xform : wp.transform
        Body-to-world transform.
    point : array_like, shape (3,)
        Point in the body frame.
    normal : array_like, shape (3,)
        Direction in the body frame.

    Returns
    -------
    point : ndarray, shape (3,)
        Transformed point.
    normal : ndarray, shape (3,)
        Rotated direction, normalized to unit length.
    """
    world_point = wp.transform_point(xform, wp.vec3(*point))
    world_normal = wp.normalize(wp.transform_vector(xform, wp.vec3(*normal)))
    return np.array(world_point), np.array(world_normal)


def add_ball(
    builder: newton.ModelBuilder,
    position: Vec3,
    radius: float,
    *,
    density: float = CUBE_DENSITY,
    label: str = "ball",
) -> int:
    """Add a free-floating sphere body.

    Parameters
    ----------
    builder : newton.ModelBuilder
        Builder to modify.
    position : array_like, shape (3,)
        Initial center of the sphere, in world frame.
    radius : float
        Sphere radius in metres.
    density : float, default CUBE_DENSITY
        Density in kg/m^3.
    label : str, default "ball"
        Body label.

    Returns
    -------
    int
        Index of the new body.
    """
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(*position), q=wp.quat_identity()), label=label
    )
    cfg = builder.default_shape_cfg.copy()
    cfg.density = density
    builder.add_shape_sphere(body, radius=radius, cfg=cfg)
    return body


def _drop_cube_position(point: Vec3, normal: Vec3 = _UP) -> np.ndarray:
    """Return a drop cube's initial center, ``DROP_GAP`` clear of ``point`` along ``normal``."""
    return np.asarray(point) + (CUBE_HALF_EXTENT + DROP_GAP) * np.asarray(normal)


def add_drop_cube(
    builder: newton.ModelBuilder,
    point: Vec3 = (0.0, 0.0, 0.0),
    normal: Vec3 = _UP,
    *,
    label: str = "cube",
) -> int:
    """Add a free-floating cube body just above a point.

    The cube has half-extent ``CUBE_HALF_EXTENT`` and density
    ``CUBE_DENSITY``. Its center starts at
    ``point + (CUBE_HALF_EXTENT + DROP_GAP) * normal``, with identity
    orientation.

    Parameters
    ----------
    builder : newton.ModelBuilder
        Builder to modify.
    point : array_like, shape (3,), default (0, 0, 0)
        Point above which the cube starts, in world frame.
    normal : array_like, shape (3,), default (0, 0, 1)
        Direction in which the cube is offset from ``point``.
    label : str, default "cube"
        Body label.

    Returns
    -------
    int
        Index of the new body.
    """
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(*_drop_cube_position(point, normal)), q=wp.quat_identity()),
        label=label,
    )
    cfg = builder.default_shape_cfg.copy()
    cfg.density = CUBE_DENSITY
    builder.add_shape_box(
        body, hx=CUBE_HALF_EXTENT, hy=CUBE_HALF_EXTENT, hz=CUBE_HALF_EXTENT, cfg=cfg
    )
    return body


def frame_camera(viewer, target: Vec3, offset: Vec3, *, near: float = 0.001) -> None:
    """Place the viewer camera at ``target + offset``, looking at ``target``.

    Parameters
    ----------
    viewer : newton.viewer.ViewerBase
        Viewer whose camera is set.
    target : array_like, shape (3,)
        Point to look at, in world frame.
    offset : array_like, shape (3,)
        Camera position relative to ``target``.
    near : float, default 0.001
        Near clipping distance in metres. Only applied to viewers that
        have a ``camera`` attribute.
    """
    front = -np.asarray(offset, dtype=float) / np.linalg.norm(offset)
    pitch = np.degrees(np.arcsin(front[2]))  # newton's convention, for Z-up
    yaw = np.degrees(np.arctan2(front[1], front[0]))
    target = wp.vec3(*target)
    viewer.set_camera(pos=target + wp.vec3(*offset), pitch=float(pitch), yaw=float(yaw))
    if hasattr(viewer, "camera"):
        viewer.camera.look_at(target)
        viewer.camera.near = near


class PatchHeatmap:
    """Color patch shapes by the force on them.

    Each surface's patches get distinct base colors from
    :func:`~halltaxel.viz.patch_colors`, blended toward red with the
    force on each patch by :func:`~halltaxel.viz.force_heat_colors`. The
    base colors are written to ``builder.shape_color`` on construction.

    Parameters
    ----------
    builder : newton.ModelBuilder
        Builder the patch shapes were added to, before ``finalize()``.
    shape_groups : Sequence[Sequence[int]]
        Patch shape indices of each surface, in the order their forces
        appear in the forces passed to :meth:`update`.
    saturation_force : float, default CUBE_WEIGHT / 2
        Force magnitude at which a patch is fully red, in N.
    """

    def __init__(
        self,
        builder: newton.ModelBuilder,
        shape_groups: Sequence[Sequence[int]],
        *,
        saturation_force: float = CUBE_WEIGHT / 2,
    ):
        self._shapes = np.concatenate([np.asarray(g, dtype=int) for g in shape_groups])
        self._group_sizes = [len(g) for g in shape_groups]
        self._base = np.concatenate([patch_colors(len(g)) for g in shape_groups])
        self._saturation = saturation_force
        self._model_colors: np.ndarray | None = None
        for shape, color in zip(self._shapes, self._base):
            builder.shape_color[shape] = tuple(color)

    def update(self, model: newton.Model, forces: np.ndarray) -> list[float]:
        """Update the patch colors in ``model`` from per-patch forces.

        Parameters
        ----------
        model : newton.Model
            The finalized model; ``model.shape_color`` is updated.
        forces : ndarray, shape (num_patches, 3)
            Force per patch, over all groups in order.

        Returns
        -------
        list of float
            Sum of the patch force magnitudes of each surface.
        """
        if self._model_colors is None:
            self._model_colors = model.shape_color.numpy()
        magnitudes = np.linalg.norm(forces, axis=1)
        self._model_colors[self._shapes] = force_heat_colors(
            self._base, magnitudes, self._saturation
        )
        model.shape_color.assign(self._model_colors)
        per_surface = np.split(magnitudes, np.cumsum(self._group_sizes)[:-1])
        return [float(m.sum()) for m in per_surface]
