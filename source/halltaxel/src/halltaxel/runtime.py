"""Per-step driver that turns simulation state into tactile readings."""

from __future__ import annotations

from typing import Any

from halltaxel.backends.base import TactileBackend, TactileHandle
from halltaxel.reading import TactileReading

__all__ = ["TactileRuntime"]


class TactileRuntime:
    """Produce a :class:`TactileReading` from a finalized model each step.

    The constructor calls ``backend.finalize(model)``, so it must run
    after ``builder.finalize()`` and before the ``newton.Contacts`` buffer
    is allocated.

    Parameters
    ----------
    model : newton.Model
        The finalized model the surfaces in ``handles`` belong to.
    backend : TactileBackend
        The backend every surface in ``handles`` was attached with.
    handles : list of TactileHandle
        The handles returned by :func:`attach_tactile` for this model.

    Raises
    ------
    ValueError
        If ``handles`` is empty.
    """

    def __init__(self, model: Any, backend: TactileBackend, handles: list[TactileHandle]):
        if not handles:
            raise ValueError(
                "handles is empty - attach at least one surface before building a TactileRuntime"
            )

        self.model = model
        self.backend = backend
        self.handles = handles
        backend.finalize(model)

    def update(self, state: Any, contacts: Any) -> TactileReading:
        """Read the tactile forces for the current step.

        Parameters
        ----------
        state : newton.State
            Current simulation state.
        contacts : newton.Contacts
            Contacts for this step, after ``solver.update_contacts()``.

        Returns
        -------
        TactileReading
            Taxel forces of every attached surface. Index it by
            a handle or label prefix to get one surface.
        """
        return self.backend.update(self.model, state, contacts)
