"""Visualization helpers for tactile surfaces.

Provides color utilities and overlays that draw through the
``newton.viewer`` logging interface (``log_points``, ``log_lines``,
``log_arrows``), so they work with any Newton viewer backend.
"""

from __future__ import annotations

import colorsys
from collections.abc import Sequence

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from halltaxel.layout import TaxelLayout
from halltaxel.segmentation import project_to_surface
from halltaxel.spec import TactileSurfaceSpec

__all__ = [
    "FORCE_COLOR",
    "TAXEL_COLOR",
    "ForceArrows",
    "TaxelOverlay",
    "force_heat_colors",
    "patch_colors",
]

_HOT = np.array([1.0, 0.15, 0.1])
TAXEL_COLOR = (0.9, 0.1, 0.1)
FORCE_COLOR = (1.0, 0.75, 0.0)
_PROJECTED_COLOR = (0.1, 0.3, 0.9)
_LINE_COLOR = (0.05, 0.05, 0.05)


def patch_colors(num_patches: int) -> np.ndarray:
    """Return one distinct color per patch.

    Hues are evenly spaced; saturation and value are fixed.

    Parameters
    ----------
    num_patches : int
        Number of colors.

    Returns
    -------
    ndarray, shape (num_patches, 3)
        RGB colors in [0, 1].
    """
    return np.array(
        [colorsys.hsv_to_rgb(p / max(num_patches, 1), 0.55, 0.95) for p in range(num_patches)]
    ).reshape(num_patches, 3)


def force_heat_colors(
    base_colors: np.ndarray, magnitudes: np.ndarray, saturation_force: float
) -> np.ndarray:
    """Blend colors toward red in proportion to force magnitude.

    Parameters
    ----------
    base_colors : ndarray, shape (N, 3)
        RGB color at zero force.
    magnitudes : array_like, shape (N,)
        Force magnitude per color.
    saturation_force : float
        Magnitude at and above which the color is fully red.

    Returns
    -------
    ndarray, shape (N, 3)
        Blended RGB colors.
    """
    t = np.minimum(np.asarray(magnitudes) / saturation_force, 1.0)[:, None]
    return base_colors + t * (_HOT - base_colors)


class TaxelOverlay:
    """Draw a surface's taxel positions in a Newton viewer.

    Taxels are drawn as red points. Optionally, each taxel's closest
    point on ``spec.sensing_mesh`` is drawn as a blue point connected to
    the taxel by a line.

    Parameters
    ----------
    spec : TactileSurfaceSpec
        Surface whose ``taxel_positions`` are drawn.
    body : int, default -1
        Body the surface is attached to. The points follow its pose. If
        -1, the positions are drawn as given, in world frame.
    show_projection : bool, default False
        Also draw each taxel's projection onto ``spec.sensing_mesh``.
    radius : float, default 0.0001
        Point radius in metres.
    """

    def __init__(
        self,
        spec: TactileSurfaceSpec,
        *,
        body: int = -1,
        show_projection: bool = False,
        radius: float = 0.0001,
    ):
        self.body = body
        self.radius = radius
        self._name = f"/halltaxel/{spec.label_prefix}"
        self._taxels = np.asarray(spec.taxel_positions, dtype=float)
        self._projected = None
        if show_projection:
            self._projected = project_to_surface(spec.sensing_mesh, self._taxels).points

    def log(self, viewer, state=None) -> None:
        """Log the overlay for the current frame.

        Call between ``viewer.begin_frame()`` and ``viewer.end_frame()``.

        Parameters
        ----------
        viewer : newton.viewer.ViewerBase
            Viewer to log to.
        state : newton.State, optional
            State providing the body pose. Required if ``body`` is not -1.

        Raises
        ------
        ValueError
            If the overlay follows a body and ``state`` is None.
        """
        pose = self._body_pose(state)
        taxels = self._vec3(self._to_world(pose, self._taxels))
        viewer.log_points(f"{self._name}/taxels", taxels, self.radius, _fill(taxels, TAXEL_COLOR))
        if self._projected is None:
            return
        projected = self._vec3(self._to_world(pose, self._projected))
        viewer.log_points(
            f"{self._name}/projected", projected, self.radius, _fill(projected, _PROJECTED_COLOR)
        )
        viewer.log_lines(f"{self._name}/projection", taxels, projected, _LINE_COLOR)

    def _body_pose(self, state) -> np.ndarray | None:
        """Return the followed body's pose from ``state``, or None in world-frame mode."""
        if self.body < 0:
            return None
        if state is None:
            raise ValueError("this overlay follows a body - pass the state to log()")
        return state.body_q.numpy()[self.body]

    @staticmethod
    def _to_world(pose: np.ndarray | None, points: np.ndarray) -> np.ndarray:
        """Transform ``points`` by a (px, py, pz, qx, qy, qz, qw) pose; None is identity."""
        if pose is None:
            return points
        return Rotation.from_quat(pose[3:]).apply(points) + pose[:3]  # body_q is xyzw

    @staticmethod
    def _vec3(points: np.ndarray) -> wp.array:
        """Convert an (N, 3) array to a ``wp.array`` of ``wp.vec3``."""
        return wp.array(points.astype(np.float32), dtype=wp.vec3)


class ForceArrows:
    """Draw per-taxel forces as arrows in a Newton viewer.

    Each arrow ends at its taxel position and points into the surface,
    opposite to the taxel's reaction force (i.e. along the force pressing
    on it), with length proportional to the force magnitude. Forces are
    given in each surface's sensor frame, as in :class:`TactileReading`,
    and rotated into the world frame for drawing.

    Parameters
    ----------
    layouts : Sequence[TaxelLayout]
        Layout of each surface, in the order their taxels appear in the
        forces passed to :meth:`log`. Its ``sensor_xform`` gives the
        frame of the surface's forces.
    bodies : Sequence[int]
        Body each layout is attached to. Layouts are in that body's
        frame.
    scale : float
        Arrow length per unit force, in m/N.
    min_force : float, default 1e-6
        Forces with magnitude at or below this value are not drawn, in N.
    name : str, default "/halltaxel/forces"
        Viewer path the arrows are logged under.

    Raises
    ------
    ValueError
        If ``layouts`` and ``bodies`` differ in length.
    """

    def __init__(
        self,
        layouts: Sequence[TaxelLayout],
        bodies: Sequence[int],
        *,
        scale: float,
        min_force: float = 1e-6,
        name: str = "/halltaxel/forces",
    ):
        if len(layouts) != len(bodies):
            raise ValueError(f"got {len(layouts)} layouts but {len(bodies)} bodies")
        self._positions = np.concatenate([layout.positions for layout in layouts])
        self._sensor_rotations = Rotation.from_quat(
            np.concatenate(
                [
                    np.tile(np.array(layout.sensor_xform.q), (layout.num_taxels, 1))
                    for layout in layouts
                ]
            )
        )
        self._bodies = np.concatenate(
            [np.full(layout.num_taxels, body) for layout, body in zip(layouts, bodies)]
        )
        self.scale = scale
        self.min_force = min_force
        self._name = name

    def log(self, viewer, state, forces: np.ndarray) -> None:
        """Log the arrows for the current frame.

        Call between ``viewer.begin_frame()`` and ``viewer.end_frame()``.

        Parameters
        ----------
        viewer : newton.viewer.ViewerBase
            Viewer to log to.
        state : newton.State
            State providing the body poses.
        forces : ndarray, shape (num_taxels, 3)
            Reaction force per taxel in its surface's sensor frame, e.g.
            ``reading.forces[0]``.
        """
        pose = state.body_q.numpy()[self._bodies]
        body_rotations = Rotation.from_quat(pose[:, 3:])  # body_q is xyzw
        anchors = body_rotations.apply(self._positions) + pose[:, :3]
        active = np.linalg.norm(forces, axis=1) > self.min_force
        if not active.any():
            viewer.log_arrows(self._name, None, None, None)
            return
        world_forces = (body_rotations * self._sensor_rotations)[active].apply(forces[active])
        ends = anchors[active]
        starts = ends + self.scale * world_forces
        ends = TaxelOverlay._vec3(ends)
        viewer.log_arrows(self._name, TaxelOverlay._vec3(starts), ends, _fill(ends, FORCE_COLOR))


def _fill(points: wp.array, color: tuple[float, float, float]) -> wp.array:
    """Return a per-point color array of ``color``, on the device of ``points``."""
    # One color per point: ViewerGL.log_points rejects a single RGB tuple,
    # despite the base viewer's signature allowing one.
    return wp.full(len(points), wp.vec3(*color), dtype=wp.vec3, device=points.device)
