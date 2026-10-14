"""Simulation of tactile taxel sensors on top of Newton/Warp.

halltaxel attaches tactile sensing surfaces to bodies of a
``newton.ModelBuilder`` and reads per-taxel forces back each simulation
step. A surface is described once, independently of how it is sensed
(:class:`TactileSurfaceSpec`); the sensing method is pluggable through
:class:`TactileBackend`. Concrete backends live in
:mod:`halltaxel.backends` and are imported from there directly.

The package never imports torch at module scope, and the number of worlds
is always taken from the ``newton.Model``, so the same code serves a
single-world viewer and a batched multi-world simulation.
"""

from halltaxel.backends.base import TactileBackend, TactileHandle, attach_tactile
from halltaxel.layout import TaxelLayout
from halltaxel.reading import TactileReading
from halltaxel.runtime import TactileRuntime
from halltaxel.segmentation import (
    SurfaceProjection,
    SurfaceSegmentation,
    load_surface_mesh,
    project_to_surface,
    segment_surface,
    select_region,
)
from halltaxel.spec import TactileSurfaceSpec

__all__ = [
    "SurfaceProjection",
    "SurfaceSegmentation",
    "TactileBackend",
    "TactileHandle",
    "TactileReading",
    "TactileRuntime",
    "TactileSurfaceSpec",
    "TaxelLayout",
    "attach_tactile",
    "load_surface_mesh",
    "project_to_surface",
    "segment_surface",
    "select_region",
]
