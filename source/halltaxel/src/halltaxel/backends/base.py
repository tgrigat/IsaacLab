"""Abstract interface for tactile sensing backends.

A backend is used in three stages:

1. :meth:`TactileBackend.attach` adds the shapes (or other elements) for
   one sensing surface to a ``newton.ModelBuilder``, before
   ``builder.finalize()``.
2. :meth:`TactileBackend.finalize` builds model-wide sensing state once,
   after ``builder.finalize()``.
3. :meth:`TactileBackend.update` returns a :class:`TactileReading` each
   simulation step.

:class:`~halltaxel.runtime.TactileRuntime` calls stages 2 and 3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from halltaxel.layout import TaxelLayout
from halltaxel.reading import TactileReading
from halltaxel.spec import TactileSurfaceSpec

__all__ = ["TactileBackend", "TactileHandle", "attach_tactile"]


@dataclass(frozen=True)
class TactileHandle:
    """Handle for one surface attached by :meth:`TactileBackend.attach`.

    Backends subclass this to add backend-specific fields.

    Attributes
    ----------
    label_prefix : str
        The surface's label prefix, from its :class:`TactileSurfaceSpec`.
    layout : TaxelLayout
        The surface's taxel positions and sensor frame, in the attached
        body's frame.
    taxels : slice
        The surface's taxels along the taxel axis of every
        :class:`TactileReading` from the backend it was attached with.
    """

    label_prefix: str
    layout: TaxelLayout
    taxels: slice


class TactileBackend(ABC):
    """Abstract base class for tactile sensing methods.

    One instance is shared by every surface attached to a given model;
    :meth:`update` returns the taxels of all of them in a single
    :class:`TactileReading`, with each surface's taxel slice recorded in
    :attr:`TactileReading.surfaces` and on its :class:`TactileHandle`.
    """

    @abstractmethod
    def attach(
        self,
        builder: Any,
        body: int,
        spec: TactileSurfaceSpec,
        *,
        xform: Any | None = None,
        cfg: Any | None = None,
    ) -> TactileHandle:
        """Add one sensing surface to a body of ``builder``.

        Must be called before ``builder.finalize()``.

        Parameters
        ----------
        builder : newton.ModelBuilder
            Builder to modify.
        body : int
            Index of the body the surface is attached to.
        spec : TactileSurfaceSpec
            The surface to attach.
        xform : wp.transform, optional
            Transform of the surface in ``body``'s frame. Defaults to the
            backend's default, typically identity.
        cfg : object, optional
            Backend-specific configuration. Defaults to the backend's
            default.

        Returns
        -------
        TactileHandle
            Handle describing the attached surface.
        """

    @abstractmethod
    def finalize(self, model: Any) -> None:
        """Build the model-wide sensing state.

        Called once, after all :meth:`attach` calls and
        ``builder.finalize()``, and before the ``newton.Contacts`` buffer
        is allocated. :class:`TactileRuntime` calls this method.

        Parameters
        ----------
        model : newton.Model
            The finalized model.
        """

    @abstractmethod
    def update(self, model: Any, state: Any, contacts: Any) -> TactileReading:
        """Read the tactile forces of all attached surfaces for the current step.

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
            Taxel forces of every attached surface, keyed by
            label prefix in :attr:`TactileReading.surfaces`.
        """


def attach_tactile(
    builder: Any,
    body: int,
    spec: TactileSurfaceSpec,
    *,
    backend: TactileBackend,
    xform: Any | None = None,
    cfg: Any | None = None,
) -> TactileHandle:
    """Attach a tactile sensing surface to a body.

    Equivalent to ``backend.attach(builder, body, spec, xform=xform,
    cfg=cfg)``.

    Parameters
    ----------
    builder : newton.ModelBuilder
        Builder to modify, before ``finalize()``.
    body : int
        Index of the body the surface is attached to.
    spec : TactileSurfaceSpec
        The surface to attach.
    backend : TactileBackend
        Backend that builds the surface. Use the same instance for every
        surface of a model.
    xform : wp.transform, optional
        Transform of the surface in ``body``'s frame.
    cfg : object, optional
        Backend-specific configuration.

    Returns
    -------
    TactileHandle
        Handle describing the attached surface.
    """
    return backend.attach(builder, body, spec, xform=xform, cfg=cfg)
