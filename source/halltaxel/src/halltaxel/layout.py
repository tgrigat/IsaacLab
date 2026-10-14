"""Per-taxel positions and the sensor frame of a tactile surface."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

__all__ = ["TaxelLayout"]


@dataclass(frozen=True)
class TaxelLayout:
    """Position of each taxel on a surface, and the frame of its forces.

    Row ``i`` describes taxel ``i``, in the order the surface's taxels
    appear in a :class:`TactileReading`. On a :class:`TactileHandle`,
    all values are in the attached body's frame.

    Attributes
    ----------
    positions : ndarray, shape (N, 3)
        Position of each taxel as the backend models it. This can differ
        from ``TactileSurfaceSpec.taxel_positions``, e.g. when a backend
        moves the taxels onto the surface.
    sensor_xform : wp.transform, default identity
        Pose of the sensor frame in the layout's frame. The surface's
        taxel forces are expressed in this frame.
    """

    positions: np.ndarray
    sensor_xform: wp.transform = field(default_factory=wp.transform_identity)

    @property
    def num_taxels(self) -> int:
        """int: Number of taxels, ``len(positions)``."""
        return len(self.positions)
