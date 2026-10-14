"""Per-step tactile output shared by all backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["TactileReading"]


@dataclass(frozen=True)
class TactileReading:
    """Tactile forces for one simulation step, across all worlds and taxels.

    The leading world dimension is always present, including when there
    is only one world.

    Each surface's taxels form one contiguous run along the taxel axis,
    recorded in ``surfaces``. Index the reading by a surface's label
    prefix or handle to get that surface alone, as views into the same
    arrays::

        thumb = reading["thumb"]  # or reading[thumb_handle]
        thumb.forces  # (num_worlds, n_thumb, 3)

    Attributes
    ----------
    forces : ndarray, shape (num_worlds, num_taxels, 3)
        Reaction force of each taxel in newtons: the force the taxel
        exerts on whatever touches it, i.e. the negated contact force on
        the taxel. A pressed taxel's force points out of the surface.
        Each surface's forces are in its sensor frame
        (:attr:`TaxelLayout.sensor_xform`), the frame of the surface's
        mesh if its spec has no ``sensor_xform``.
    surfaces : Mapping[str, slice]
        Taxel slice of each surface, keyed by label prefix. The slices
        are in taxel order and together cover every taxel exactly once.

    Raises
    ------
    ValueError
        If the slices in ``surfaces`` do not tile ``range(num_taxels)``
        in order.
    """

    forces: np.ndarray
    surfaces: Mapping[str, slice]

    def __post_init__(self):
        end = 0
        for name, taxels in self.surfaces.items():
            if taxels.start != end or taxels.stop < end or taxels.step not in (None, 1):
                raise ValueError(
                    f"surface {name!r} has taxels {taxels}, expected a contiguous slice "
                    f"starting at {end} - surfaces must tile the taxel axis in order"
                )
            end = taxels.stop
        if end != self.forces.shape[1]:
            raise ValueError(f"surfaces cover {end} taxels but forces has {self.forces.shape[1]}")

    def __getitem__(self, surface: Any) -> TactileReading:
        """Return the reading of one surface.

        Parameters
        ----------
        surface : str or TactileHandle
            The surface's label prefix, or the handle returned when it
            was attached.

        Returns
        -------
        TactileReading
            The surface's forces, shape
            ``(num_worlds, n, 3)``, as views into this reading's arrays.

        Raises
        ------
        KeyError
            If no surface has that label prefix.
        ValueError
            If a handle is given whose taxel slice differs from the
            surface's in this reading, e.g. a handle from another
            backend.
        """
        name = surface if isinstance(surface, str) else surface.label_prefix
        try:
            taxels = self.surfaces[name]
        except KeyError:
            raise KeyError(
                f"no surface {name!r} in this reading - surfaces are {list(self.surfaces)}"
            ) from None
        if not isinstance(surface, str) and surface.taxels != taxels:
            raise ValueError(
                f"handle {name!r} has taxels {surface.taxels} but this reading has {taxels} - "
                "the handle belongs to a different backend"
            )
        return TactileReading(
            forces=self.forces[:, taxels],
            surfaces={name: slice(0, taxels.stop - taxels.start)},
        )

    def surface_ids(self) -> np.ndarray:
        """Return the surface index of each taxel.

        Useful for per-surface reductions over the flat arrays without a
        Python loop, e.g. ``np.add.at(totals, (slice(None), ids), f)``.

        Returns
        -------
        ndarray of int, shape (num_taxels,)
            Entry ``i`` is the position of taxel ``i``'s surface in
            ``list(self.surfaces)``.
        """
        sizes = [taxels.stop - taxels.start for taxels in self.surfaces.values()]
        return np.repeat(np.arange(len(sizes)), sizes)

    def numpy(self) -> np.ndarray:
        """Return the forces as a NumPy array.

        Returns
        -------
        ndarray, shape (num_worlds, num_taxels, 3)
            ``self.forces``, not a copy.
        """
        return self.forces

    def torch(self) -> Any:
        """Return the forces as a torch tensor.

        torch is imported on first call; it is not a dependency of
        halltaxel.

        Returns
        -------
        torch.Tensor, shape (num_worlds, num_taxels, 3)
            Tensor sharing memory with ``self.forces``.

        Raises
        ------
        ImportError
            If torch is not installed.
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "TactileReading.torch() needs torch installed - halltaxel itself never "
                "requires it (see this module's docstring)"
            ) from e
        return torch.from_numpy(self.forces)
